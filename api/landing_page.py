"""
Enhanced API with custom landing page
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Add this to your existing api/main.py

@app.get("/", response_class=HTMLResponse)
async def api_home():
    """Custom API landing page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EquiPay Canada API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: #1f77b4; color: white; padding: 20px; border-radius: 8px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .get { background: #007bff; }
            .post { background: #28a745; }
            a { color: #1f77b4; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏢 EquiPay Canada API</h1>
            <p>Salary Prediction & Pay Equity Analysis for Canadian Workers</p>
        </div>
        
        <h2>🚀 Quick Links</h2>
        <ul>
            <li><a href="/docs">📋 Interactive API Documentation (Swagger)</a></li>
            <li><a href="/redoc">📖 Beautiful Documentation (ReDoc)</a></li>
            <li><a href="/health">❤️ API Health Check</a></li>
        </ul>
        
        <h2>🔗 Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> 
            <strong>/predict</strong> - Predict salary for individual worker
            <p>Submit worker characteristics, get hourly wage prediction with confidence intervals.</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> 
            <strong>/predict/batch</strong> - Predict salaries for multiple workers
            <p>Submit array of worker profiles, get batch predictions with summary statistics.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> 
            <strong>/wage-gap</strong> - Analyze gender wage gap
            <p>Get current wage gap statistics with statistical significance testing.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> 
            <strong>/fairness</strong> - Model fairness evaluation
            <p>Bias metrics and fairness evaluation for salary predictions.</p>
        </div>
        
        <h2>💡 Example Usage</h2>
        <pre>
# Predict salary
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sex": 1,
    "age": 35,
    "education": 3,
    "occupation": 1,
    "province": 35,
    "full_time": 1
  }'

# Check wage gap
curl "http://localhost:8000/wage-gap"
        </pre>
        
        <h2>🔧 Integration</h2>
        <p>
            This API integrates with the <strong>EquiPay Dashboard</strong> and can be embedded in:
        </p>
        <ul>
            <li>HR systems for real-time salary benchmarking</li>
            <li>Job boards for salary estimates</li>
            <li>Compensation planning tools</li>
            <li>Research applications</li>
        </ul>
        
        <p><em>Data Source: Statistics Canada Labour Force Survey (2010-2025)</em></p>
    </body>
    </html>
    """