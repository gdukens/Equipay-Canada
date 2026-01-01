#!/usr/bin/env python3
"""
Download LFS PUMF Data from Statistics Canada
==============================================

This script downloads Labour Force Survey Public Use Microdata Files (PUMF)
directly from Statistics Canada.

Data Source: Statistics Canada, Catalogue 71M0001X
https://www150.statcan.gc.ca/n1/en/catalogue/71M0001X

Usage:
    python scripts/download_lfs_data.py
    python scripts/download_lfs_data.py --years 2020 2021 2022
    python scripts/download_lfs_data.py --all
"""

import os
import sys
import argparse
import requests
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# STATISTICS CANADA LFS PUMF DOWNLOAD URLS
# =============================================================================

# Historical Annual Files (2010-2023)
# Source: Statistics Canada, Labour Force Survey PUMF, Catalogue 71M0001X
LFS_HISTORICAL_URLS: Dict[int, str] = {
    2010: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2010-CSV.zip?st=4m-1u8Qj",
    2011: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2011-CSV.zip?st=vJuu4LXD",
    2012: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2012-CSV.zip?st=3ITzqtP9",
    2013: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2013-CSV.zip?st=Vi4QJe2u",
    2014: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2014-CSV.zip?st=WEexF6nK",
    2015: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2015-CSV.zip?st=rxlb9Knp",
    2016: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2016-CSV.zip?st=okrla9pH",
    2017: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2017-CSV.zip?st=40RZoXTR",
    2018: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2018-CSV.zip?st=z3ReIgSx",
    2019: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2019-CSV.zip?st=GU3YE_Ud",
    2020: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2020-CSV.zip?st=yEkc2bIl",
    2021: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2021-CSV.zip?st=CTZvSwXP",
    2022: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2022-CSV.zip?st=gaOjoGz6",
    2023: "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/hist/2023-CSV.zip?st=T-tYCepz",
}

# Monthly Files for 2025
# Source: Statistics Canada, Labour Force Survey PUMF, Catalogue 71M0001X
LFS_2025_MONTHLY_URLS: Dict[str, str] = {
    "2025-01": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-01-CSV.zip?st=GmUiF6LE",
    "2025-02": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-02-CSV.zip?st=utsZ8KAl",
    "2025-03": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-03-CSV.zip?st=4XmM6T6L",
    "2025-04": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-04-CSV.zip?st=N9BEoGTB",
    "2025-05": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-05-CSV.zip?st=vMwFvCq9",
    "2025-06": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-06-CSV.zip?st=JHs8QJRc",
    "2025-07": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-07-CSV.zip?st=qMC23vCC",
    "2025-08": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-08-CSV.zip?st=zEqnk591",
    "2025-09": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-09-CSV.zip?st=wBCK60py",
    "2025-10": "https://www150.statcan.gc.ca/n1/en/pub/71m0001x/2021001/2025-10-CSV.zip?st=OOpTDsDq",
}


class LFSDataDownloader:
    """
    Downloads and extracts LFS PUMF data from Statistics Canada.
    """
    
    def __init__(self, output_dir: str = "data/raw/lfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EquiPay-Canada/1.0 (Research Project)',
            'Accept': 'application/zip,application/octet-stream,*/*',
        })
    
    def download_file(self, url: str, label: str) -> List[Path]:
        """
        Download a single ZIP file and extract ALL CSV contents.
        
        Args:
            url: Statistics Canada download URL
            label: Label for the file (e.g., "2020" or "2025-01")
            
        Returns:
            List of paths to extracted CSV files
        """
        logger.info(f"Downloading {label}...")
        extracted_files = []
        
        try:
            response = self.session.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            # Extract ZIP contents
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
                
                if not csv_files:
                    logger.warning(f"  No CSV files found in ZIP for {label}")
                    return []
                
                logger.info(f"  Found {len(csv_files)} CSV files in ZIP")
                
                # Extract ALL CSV files (skip codebooks)
                for csv_file in csv_files:
                    # Read content first to check if it's a codebook
                    with z.open(csv_file) as source:
                        content = source.read()
                        # Check first line for codebook indicators
                        first_line = content[:500].decode('utf-8', errors='ignore').upper()
                        if 'FIELD_CHAMP' in first_line or 'POSITION_POSITION' in first_line:
                            logger.info(f"    Skipping codebook: {csv_file}")
                            continue
                    
                    # Determine output filename based on content or ZIP structure
                    # For yearly ZIPs, each CSV is a monthly file
                    base_name = Path(csv_file).stem.lower()
                    
                    if "-" in label:  # Monthly file (e.g., 2025-01)
                        output_name = f"lfs_{label.replace('-', '_')}.csv"
                    else:
                        # Annual ZIP - extract month from filename
                        # Common patterns: pub202001.csv, LFS_2020_01.csv, lfs202001.csv
                        import re
                        # Try to find YYYYMM or YYYY_MM pattern
                        match = re.search(r'(\d{4})[-_]?(\d{2})', base_name)
                        if match:
                            year = match.group(1)
                            month = match.group(2)
                            output_name = f"lfs_{year}_{month}.csv"
                        else:
                            # Fallback: use original name
                            output_name = f"lfs_{label}_{base_name}.csv"
                    
                    output_path = self.output_dir / output_name
                    
                    # Write file
                    with open(output_path, 'wb') as target:
                        target.write(content)
                    
                    logger.info(f"    ✓ Extracted: {output_name} ({len(content):,} bytes)")
                    extracted_files.append(output_path)
                
                return extracted_files
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"  ✗ Failed to download {label}: {e}")
            return []
        except zipfile.BadZipFile as e:
            logger.error(f"  ✗ Invalid ZIP file for {label}: {e}")
            return []
        except Exception as e:
            logger.error(f"  ✗ Unexpected error for {label}: {e}")
            return []
    
    def download_historical(self, years: Optional[List[int]] = None) -> List[Path]:
        """
        Download historical annual LFS PUMF files (2010-2023).
        Each ZIP contains 12 monthly CSV files.
        
        Args:
            years: List of years to download. If None, downloads all available.
            
        Returns:
            List of paths to downloaded files
        """
        years = years or list(LFS_HISTORICAL_URLS.keys())
        downloaded = []
        
        logger.info("=" * 60)
        logger.info("DOWNLOADING HISTORICAL LFS PUMF DATA")
        logger.info("Source: Statistics Canada, Catalogue 71M0001X")
        logger.info("Each ZIP contains 12 monthly CSV files")
        logger.info("=" * 60)
        
        for year in sorted(years):
            if year not in LFS_HISTORICAL_URLS:
                logger.warning(f"No URL available for {year}")
                continue
                
            # Check if monthly files already exist for this year
            existing_months = list(self.output_dir.glob(f"lfs_{year}_*.csv"))
            if len(existing_months) >= 12:
                logger.info(f"  ⏭ Skipping {year} ({len(existing_months)} monthly files already exist)")
                downloaded.extend(existing_months)
                continue
            
            paths = self.download_file(LFS_HISTORICAL_URLS[year], str(year))
            if paths:
                downloaded.extend(paths)
        
        return downloaded
    
    def download_2025_monthly(self, months: Optional[List[str]] = None) -> List[Path]:
        """
        Download 2025 monthly LFS PUMF files.
        
        Args:
            months: List of months in format "2025-MM". If None, downloads all available.
            
        Returns:
            List of paths to downloaded files
        """
        months = months or list(LFS_2025_MONTHLY_URLS.keys())
        downloaded = []
        
        logger.info("=" * 60)
        logger.info("DOWNLOADING 2025 MONTHLY LFS PUMF DATA")
        logger.info("Source: Statistics Canada, Catalogue 71M0001X")
        logger.info("=" * 60)
        
        for month in sorted(months):
            if month not in LFS_2025_MONTHLY_URLS:
                logger.warning(f"No URL available for {month}")
                continue
            
            # Check if already downloaded
            existing = self.output_dir / f"lfs_{month.replace('-', '_')}.csv"
            if existing.exists():
                logger.info(f"  ⏭ Skipping {month} (already exists)")
                downloaded.append(existing)
                continue
            
            paths = self.download_file(LFS_2025_MONTHLY_URLS[month], month)
            if paths:
                downloaded.extend(paths)
        
        return downloaded
    
    def download_all(self) -> List[Path]:
        """
        Download all available LFS PUMF data.
        
        Returns:
            List of all downloaded file paths
        """
        downloaded = []
        downloaded.extend(self.download_historical())
        downloaded.extend(self.download_2025_monthly())
        
        logger.info("=" * 60)
        logger.info(f"DOWNLOAD COMPLETE: {len(downloaded)} files")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("=" * 60)
        
        return downloaded
    
    def get_download_summary(self) -> str:
        """Get a summary of available data files."""
        existing_files = list(self.output_dir.glob("lfs_*.csv"))
        
        summary = [
            "=" * 60,
            "LFS PUMF DATA SUMMARY",
            "=" * 60,
            f"Data directory: {self.output_dir.absolute()}",
            f"Files available: {len(existing_files)}",
            "",
        ]
        
        if existing_files:
            summary.append("Available files:")
            for f in sorted(existing_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                summary.append(f"  - {f.name} ({size_mb:.1f} MB)")
        else:
            summary.append("No data files found. Run download_all() to fetch data.")
        
        summary.append("=" * 60)
        return "\n".join(summary)


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download LFS PUMF data from Statistics Canada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_lfs_data.py --all
    python scripts/download_lfs_data.py --years 2020 2021 2022 2023
    python scripts/download_lfs_data.py --months 2025-01 2025-02 2025-03
    python scripts/download_lfs_data.py --summary
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Download all available LFS PUMF data (2010-2023 + 2025 monthly)'
    )
    parser.add_argument(
        '--years', type=int, nargs='+',
        help='Specific years to download (e.g., --years 2020 2021 2022)'
    )
    parser.add_argument(
        '--months', type=str, nargs='+',
        help='Specific 2025 months to download (e.g., --months 2025-01 2025-02)'
    )
    parser.add_argument(
        '--output', type=str, default='data/raw/lfs',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Show summary of available data files'
    )
    
    args = parser.parse_args()
    
    downloader = LFSDataDownloader(output_dir=args.output)
    
    if args.summary:
        print(downloader.get_download_summary())
        return
    
    if args.all:
        downloader.download_all()
    elif args.years:
        downloader.download_historical(years=args.years)
    elif args.months:
        downloader.download_2025_monthly(months=args.months)
    else:
        # Default: download all
        print("No specific options provided. Downloading all available data...")
        downloader.download_all()
    
    print(downloader.get_download_summary())


if __name__ == '__main__':
    main()
