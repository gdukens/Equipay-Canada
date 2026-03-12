#!/usr/bin/env python3
"""
Railway deployment startup script
Creates sample data and starts appropriate service based on environment
"""
import os
import sys
from pathlib import Path

# Ensure proper Python path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

def setup_railway_environment():
    """Set up Railway-specific environment and data."""
    print("🚄 Setting up Railway environment...")
    
    # Set environment variables
    os.environ['EQUIPAY_MODE'] = 'RAILWAY'
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['PYTHONPATH'] = '/app'
    
    # Create sample data if it doesn't exist
    sample_csv = Path('data/processed/lfs_processed.csv')
    if not sample_csv.exists():
        print("📊 Creating sample data...")
        try:
            from scripts.create_sample_data import create_sample_data
            create_sample_data()
            print("✅ Sample data created successfully")
        except Exception as e:
            print(f"⚠️ Could not create sample data: {e}")
    
    print("🟢 Railway environment ready!")

def start_service():
    """Start the appropriate service based on SERVICE_TYPE environment variable."""
    service_type = os.environ.get('SERVICE_TYPE', 'api')
    port = os.environ.get('PORT', '8000')
    
    if service_type == 'dashboard':
        print(f"🎛️ Starting Streamlit dashboard on port {port}...")
        os.system(f'streamlit run app/dashboard.py --server.port {port} --server.address 0.0.0.0 --server.headless true')
    else:
        print(f"🔗 Starting FastAPI server on port {port}...")
        os.system(f'uvicorn api.main:app --host 0.0.0.0 --port {port}')

if __name__ == "__main__":
    setup_railway_environment()
    start_service()