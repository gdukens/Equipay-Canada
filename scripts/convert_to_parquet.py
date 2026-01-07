#!/usr/bin/env python3
"""
Memory-Efficient Parquet Converter for EquiPay Canada
======================================================

Converts raw LFS CSV files to optimized Parquet format with minimal memory usage.

Strategy for 8GB RAM systems:
1. Process ONE file at a time using DuckDB streaming
2. Append to partitioned Parquet structure
3. Never load full dataset into memory
4. Peak memory: ~500MB regardless of total data size

Usage:
    python scripts/convert_to_parquet.py
    
    # Or with options:
    python scripts/convert_to_parquet.py --workers 2 --compression zstd
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_CSV_PATH = Path("data/raw/lfs")
PARQUET_OUTPUT_PATH = Path("data/parquet")
MEMORY_LIMIT = "3GB"  # Conservative for 8GB system (leaves room for OS + other apps)


# Optimized dtypes for LFS data - reduces memory and storage
DTYPE_OPTIMIZATIONS = """
    -- Cast columns to efficient types
    CAST(REC_NUM AS INTEGER) AS REC_NUM,
    CAST(SURVYEAR AS SMALLINT) AS SURVYEAR,
    CAST(SURVMNTH AS TINYINT) AS SURVMNTH,
    CAST(LFSSTAT AS TINYINT) AS LFSSTAT,
    CAST(PROV AS TINYINT) AS PROV,
    
    -- Handle GENDER column (might be SEX in older files)
    CAST(COALESCE(GENDER, SEX) AS TINYINT) AS GENDER,
    
    -- Demographics
    CAST(AGE_6 AS TINYINT) AS AGE_6,
    CAST(AGE_12 AS TINYINT) AS AGE_12,
    CAST(MARSTAT AS TINYINT) AS MARSTAT,
    CAST(EDUC AS TINYINT) AS EDUC,
    
    -- Employment
    CAST(NOC_10 AS TINYINT) AS NOC_10,
    CAST(NOC_40 AS TINYINT) AS NOC_40,
    CAST(NAICS_21 AS TINYINT) AS NAICS_21,
    CAST(COWMAIN AS TINYINT) AS COWMAIN,
    CAST(FTPTMAIN AS TINYINT) AS FTPTMAIN,
    CAST(UNION AS TINYINT) AS UNION,
    CAST(PERMTEMP AS TINYINT) AS PERMTEMP,
    CAST(ESTSIZE AS TINYINT) AS ESTSIZE,
    
    -- Hours and earnings (keep as float for precision)
    CAST(HRLYEARN AS FLOAT) AS HRLYEARN,
    CAST(UHRSMAIN AS FLOAT) AS UHRSMAIN,
    CAST(UTOTHRS AS FLOAT) AS UTOTHRS,
    CAST(AHRSMAIN AS FLOAT) AS AHRSMAIN,
    CAST(ATOTHRS AS FLOAT) AS ATOTHRS,
    
    -- Weight (important for analysis)
    CAST(FINALWT AS FLOAT) AS FINALWT
"""

# Columns we absolutely need (for fallback if schema varies)
CORE_COLUMNS = [
    'REC_NUM', 'SURVYEAR', 'SURVMNTH', 'PROV', 'GENDER',
    'AGE_6', 'AGE_12', 'EDUC', 'NOC_10', 'NAICS_21',
    'FTPTMAIN', 'HRLYEARN', 'FINALWT'
]


def get_csv_files(raw_path: Path) -> List[Path]:
    """Get all LFS CSV files sorted by year/month."""
    files = sorted(raw_path.glob("lfs_*.csv"))
    logger.info(f"Found {len(files)} CSV files in {raw_path}")
    return files


def extract_year_from_filename(filepath: Path) -> int:
    """Extract survey year from filename like lfs_2023_pub0123.csv."""
    name = filepath.stem  # e.g., "lfs_2023_pub0123"
    parts = name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def convert_single_file(
    conn: duckdb.DuckDBPyConnection,
    csv_file: Path,
    output_dir: Path,
    compression: str = "zstd"
) -> Dict[str, Any]:
    """
    Convert a single CSV file to Parquet.
    
    Uses DuckDB's streaming COPY which processes data in chunks,
    never loading the entire file into memory.
    """
    year = extract_year_from_filename(csv_file)
    output_file = output_dir / f"year={year}" / f"{csv_file.stem}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_file": str(csv_file),
        "output_file": str(output_file),
        "year": year,
        "success": False,
        "error": None,
        "rows": 0,
        "input_size_mb": csv_file.stat().st_size / (1024 * 1024)
    }
    
    try:
        # First, check what columns exist in this file
        sample = conn.execute(f"""
            SELECT * FROM read_csv_auto('{csv_file}', header=true) LIMIT 1
        """).fetchdf()
        available_cols = set(sample.columns)
        
        # Build SELECT clause based on available columns
        select_parts = []
        
        # Handle GENDER vs SEX naming
        if 'GENDER' in available_cols:
            select_parts.append("CAST(GENDER AS TINYINT) AS GENDER")
        elif 'SEX' in available_cols:
            select_parts.append("CAST(SEX AS TINYINT) AS GENDER")  # Rename to GENDER
        
        # ALL 60 LFS PUMF columns with optimized type casting
        # Note: Column names that are SQL reserved words must be quoted
        column_types = {
            # =================================================================
            # IDENTIFICATION & TIME (3 cols)
            # =================================================================
            'REC_NUM': 'INTEGER',
            'SURVYEAR': 'SMALLINT',
            'SURVMNTH': 'TINYINT',
            
            # =================================================================
            # GEOGRAPHY (2 cols)
            # =================================================================
            'PROV': 'TINYINT',
            'CMA': 'SMALLINT',  # Census Metropolitan Area
            
            # =================================================================
            # DEMOGRAPHICS (6 cols)
            # =================================================================
            'AGE_12': 'TINYINT',
            'AGE_6': 'TINYINT',
            'MARSTAT': 'TINYINT',
            'IMMIG': 'TINYINT',
            'EFAMTYPE': 'TINYINT',  # Economic family type
            'AGYOWNK': 'TINYINT',   # Age of youngest child
            
            # =================================================================
            # HUMAN CAPITAL (2 cols)
            # =================================================================
            'EDUC': 'TINYINT',
            'SCHOOLN': 'TINYINT',   # Attending school
            
            # =================================================================
            # EMPLOYMENT STATUS (3 cols)
            # =================================================================
            'LFSSTAT': 'TINYINT',
            'EVERWORK': 'TINYINT',  # Ever worked at a job
            'PRIORACT': 'TINYINT',  # Prior main activity
            
            # =================================================================
            # JOB CHARACTERISTICS (14 cols)
            # =================================================================
            'NOC_10': 'TINYINT',
            'NOC_40': 'TINYINT',
            'NOC_43': 'TINYINT',
            'NAICS_21': 'TINYINT',
            'COWMAIN': 'TINYINT',
            'FTPTMAIN': 'TINYINT',
            'FTPTLAST': 'TINYINT',  # Full/part-time status at last job
            '"UNION"': 'TINYINT',   # UNION is a SQL reserved word - must quote
            'PERMTEMP': 'TINYINT',
            'ESTSIZE': 'TINYINT',
            'FIRMSIZE': 'TINYINT',  # Firm size
            'MJH': 'TINYINT',       # Multiple job holder
            'WHYPT': 'TINYINT',     # Reason for part-time work
            'TENURE': 'SMALLINT',   # Can exceed 127 months
            'PREVTEN': 'SMALLINT',  # Previous job tenure
            
            # =================================================================
            # WORK HOURS & EARNINGS (12 cols)
            # =================================================================
            'UHRSMAIN': 'FLOAT',
            'AHRSMAIN': 'FLOAT',
            'UTOTHRS': 'FLOAT',
            'ATOTHRS': 'FLOAT',
            'HRSAWAY': 'FLOAT',     # Hours away from work
            'PAIDOT': 'SMALLINT',   # Paid overtime (can be >127)
            'UNPAIDOT': 'SMALLINT', # Unpaid overtime (can be >127)
            'XTRAHRS': 'SMALLINT',  # Extra hours worked (can be >127)
            # HRLYEARN handled specially below (cents -> dollars conversion)
            'YABSENT': 'TINYINT',   # Reason for absence
            'WKSAWAY': 'TINYINT',   # Weeks away from work
            'PAYAWAY': 'TINYINT',   # Paid while away
            'YAWAY': 'TINYINT',     # Year of absence
            
            # =================================================================
            # UNEMPLOYMENT (7 cols)
            # =================================================================
            'DURUNEMP': 'SMALLINT', # Duration of unemployment (weeks)
            'FLOWUNEM': 'TINYINT',  # Unemployment flow
            'UNEMFTPT': 'TINYINT',  # Seeking FT/PT while unemployed
            'WHYLEFTO': 'TINYINT',  # Why left last job (objective)
            'WHYLEFTN': 'TINYINT',  # Why left last job (subjective)
            'DURJLESS': 'SMALLINT', # Duration jobless
            'AVAILABL': 'TINYINT',  # Availability for work
            
            # =================================================================
            # JOB SEARCH (8 cols)
            # =================================================================
            'LKPUBAG': 'TINYINT',   # Looked at public employment agency
            'LKEMPLOY': 'TINYINT',  # Looked at employers directly
            'LKRELS': 'TINYINT',    # Looked through friends/relatives
            'LKATADS': 'TINYINT',   # Looked at ads
            'LKANSADS': 'TINYINT',  # Answered ads
            'LKOTHERN': 'TINYINT',  # Other job search methods
            'YNOLOOK': 'TINYINT',   # Reason not looking
            'TLOLOOK': 'TINYINT',   # Time since last looked
            
            # =================================================================
            # SURVEY WEIGHT (1 col)
            # =================================================================
            'FINALWT': 'FLOAT',
        }
        
        for col, dtype in column_types.items():
            # Get the unquoted column name for checking availability
            check_col = col.replace('"', '')
            if check_col in available_cols and check_col not in ['GENDER', 'SEX']:
                # For output, use unquoted alias
                alias = check_col
                select_parts.append(f"CAST({col} AS {dtype}) AS {alias}")
        
        # Handle HRLYEARN specially: convert from cents to dollars
        # LFS PUMF stores HRLYEARN in cents (e.g., 2500 = $25.00/hour)
        if 'HRLYEARN' in available_cols:
            select_parts.append("CAST(HRLYEARN / 100.0 AS FLOAT) AS HRLYEARN")
        
        select_clause = ", ".join(select_parts)
        
        # Stream convert to Parquet (memory-efficient)
        conn.execute(f"""
            COPY (
                SELECT {select_clause}
                FROM read_csv_auto('{csv_file}', 
                    header=true, 
                    ignore_errors=true,
                    parallel=false
                )
            ) TO '{output_file}' 
            (FORMAT PARQUET, COMPRESSION {compression})
        """)
        
        # Get row count
        result = conn.execute(f"""
            SELECT COUNT(*) as n FROM read_parquet('{output_file}')
        """).fetchone()
        stats["rows"] = result[0]
        
        # Get output size
        if output_file.exists():
            stats["output_size_mb"] = output_file.stat().st_size / (1024 * 1024)
            stats["compression_ratio"] = stats["input_size_mb"] / max(stats["output_size_mb"], 0.001)
        
        stats["success"] = True
        
    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Failed to convert {csv_file.name}: {e}")
    
    return stats


def convert_all_files(
    raw_path: Path = RAW_CSV_PATH,
    output_path: Path = PARQUET_OUTPUT_PATH,
    compression: str = "zstd",
    memory_limit: str = MEMORY_LIMIT,
    workers: int = 1
) -> Dict[str, Any]:
    """
    Convert all CSV files to Parquet format.
    
    Processes files one at a time to minimize memory usage.
    Each file is streamed through DuckDB without loading into memory.
    """
    logger.info("=" * 60)
    logger.info("EquiPay Canada - CSV to Parquet Conversion")
    logger.info("=" * 60)
    logger.info(f"Memory limit: {memory_limit}")
    logger.info(f"Compression: {compression}")
    logger.info(f"Input: {raw_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize DuckDB with memory limit
    conn = duckdb.connect(":memory:")
    conn.execute(f"SET memory_limit = '{memory_limit}'")
    conn.execute(f"SET threads = {workers}")
    conn.execute("SET preserve_insertion_order = false")  # Faster writes
    
    # Get all CSV files
    csv_files = get_csv_files(raw_path)
    
    if not csv_files:
        logger.error(f"No CSV files found in {raw_path}")
        return {"success": False, "error": "No CSV files found"}
    
    # Track results
    results = {
        "started_at": datetime.now().isoformat(),
        "input_path": str(raw_path),
        "output_path": str(output_path),
        "compression": compression,
        "files": [],
        "summary": {
            "total_files": len(csv_files),
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
            "total_input_mb": 0,
            "total_output_mb": 0
        }
    }
    
    # Process each file
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{len(csv_files)}] Converting {csv_file.name}...")
        
        stats = convert_single_file(conn, csv_file, output_path, compression)
        results["files"].append(stats)
        
        if stats["success"]:
            results["summary"]["successful"] += 1
            results["summary"]["total_rows"] += stats["rows"]
            results["summary"]["total_input_mb"] += stats["input_size_mb"]
            results["summary"]["total_output_mb"] += stats.get("output_size_mb", 0)
            logger.info(f"    ✓ {stats['rows']:,} rows, "
                       f"{stats['input_size_mb']:.1f}MB → {stats.get('output_size_mb', 0):.1f}MB "
                       f"({stats.get('compression_ratio', 0):.1f}x)")
        else:
            results["summary"]["failed"] += 1
            logger.error(f"    ✗ Failed: {stats['error']}")
    
    # Calculate overall compression ratio
    if results["summary"]["total_output_mb"] > 0:
        results["summary"]["overall_compression_ratio"] = (
            results["summary"]["total_input_mb"] / results["summary"]["total_output_mb"]
        )
    
    results["completed_at"] = datetime.now().isoformat()
    
    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(results, f, indent=2)
    
    conn.close()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files converted: {results['summary']['successful']}/{results['summary']['total_files']}")
    logger.info(f"Total rows: {results['summary']['total_rows']:,}")
    logger.info(f"Size: {results['summary']['total_input_mb']:.1f}MB → {results['summary']['total_output_mb']:.1f}MB")
    logger.info(f"Compression ratio: {results['summary'].get('overall_compression_ratio', 0):.1f}x")
    logger.info(f"Metadata saved to: {metadata_file}")
    
    return results


def verify_parquet(output_path: Path = PARQUET_OUTPUT_PATH):
    """Verify the Parquet files are readable and have expected structure."""
    logger.info("\nVerifying Parquet files...")
    
    conn = duckdb.connect(":memory:")
    
    try:
        # Try to read all parquet files
        parquet_glob = str(output_path / "**/*.parquet")
        
        result = conn.execute(f"""
            SELECT 
                COUNT(*) as total_rows,
                MIN(SURVYEAR) as min_year,
                MAX(SURVYEAR) as max_year,
                COUNT(DISTINCT SURVYEAR) as n_years,
                COUNT(DISTINCT PROV) as n_provinces
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        """).fetchdf()
        
        logger.info(f"✓ Total rows: {result['total_rows'].iloc[0]:,}")
        logger.info(f"✓ Years: {result['min_year'].iloc[0]} - {result['max_year'].iloc[0]} ({result['n_years'].iloc[0]} years)")
        logger.info(f"✓ Provinces: {result['n_provinces'].iloc[0]}")
        
        # Test a sample query
        result = conn.execute(f"""
            SELECT SURVYEAR, GENDER, COUNT(*) as n, AVG(HRLYEARN) as avg_wage
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            WHERE HRLYEARN > 0
            GROUP BY SURVYEAR, GENDER
            ORDER BY SURVYEAR, GENDER
            LIMIT 10
        """).fetchdf()
        
        logger.info("\n✓ Sample query successful:")
        print(result.to_string(index=False))
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFS CSV files to Parquet format (memory-efficient)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=RAW_CSV_PATH,
        help="Input directory containing CSV files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PARQUET_OUTPUT_PATH,
        help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--compression", "-c",
        choices=["zstd", "snappy", "gzip", "lz4"],
        default="zstd",
        help="Compression codec (default: zstd)"
    )
    parser.add_argument(
        "--memory",
        default=MEMORY_LIMIT,
        help="Memory limit for DuckDB (default: 3GB)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker threads (default: 1 for low memory)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing Parquet files"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_parquet(args.output)
    else:
        results = convert_all_files(
            raw_path=args.input,
            output_path=args.output,
            compression=args.compression,
            memory_limit=args.memory,
            workers=args.workers
        )
        
        if results["summary"]["successful"] > 0:
            verify_parquet(args.output)


if __name__ == "__main__":
    main()
