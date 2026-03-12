# Deployment Guide for EquiPay Canada

This guide covers multiple deployment options for the EquiPay Canada dashboard and API.

---

## 🚀 Option 1: Docker (Recommended for Production)

Your project already includes Docker configuration. This is the most portable and reliable option.

### Quick Start with Docker

```bash
# Build and run both dashboard and API
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
```

### Deploy to Cloud with Docker

**Any cloud provider supporting Docker containers:**
- Azure Container Instances
- AWS ECS/Fargate
- Google Cloud Run
- DigitalOcean App Platform
- Fly.io

**Example: Deploy to Azure Container Instances**

```bash
# Login to Azure
az login

# Create resource group
az group create --name equipay-rg --location eastus

# Create container registry
az acr create --resource-group equipay-rg --name equipayregistry --sku Basic

# Build and push image
az acr build --registry equipayregistry --image equipay-dashboard:latest .

# Deploy container
az container create \
  --resource-group equipay-rg \
  --name equipay-dashboard \
  --image equipayregistry.azurecr.io/equipay-dashboard:latest \
  --dns-name-label equipay-canada \
  --ports 8501 \
  --cpu 2 \
  --memory 4
```

---

## ☁️ Option 2: Streamlit Community Cloud (Free & Easy)

**Best for:** Quick sharing, demos, small projects

### Setup Steps:

1. **Prepare your repository:**
   ```bash
   # Create .streamlit/config.toml
   mkdir -p .streamlit
   cat > .streamlit/config.toml << EOF
   [server]
   headless = true
   port = 8501
   enableCORS = false
   
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#262730"
   EOF
   ```

2. **Create packages file:**
   ```bash
   # Streamlit Cloud uses packages.txt for system deps (if needed)
   cat > packages.txt << EOF
   build-essential
   EOF
   ```

3. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push
   ```

4. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `equipay-canada`
   - Main file: `app/dashboard.py`
   - Click "Deploy"

**Limitations:**
- Free tier: 1GB RAM, limited CPU
- May be slow with large datasets
- Set `EQUIPAY_MODE=FAST` in settings

---

## 🔵 Option 3: Azure App Service

**Best for:** Enterprise deployment, Azure ecosystem integration

### Deploy via Azure CLI:

```bash
# Login
az login

# Create resource group
az group create --name equipay-rg --location eastus

# Create App Service plan (Linux)
az appservice plan create \
  --name equipay-plan \
  --resource-group equipay-rg \
  --is-linux \
  --sku B2

# Create web app
az webapp create \
  --resource-group equipay-rg \
  --plan equipay-plan \
  --name equipay-canada-dashboard \
  --runtime "PYTHON|3.10"

# Configure startup command
az webapp config set \
  --resource-group equipay-rg \
  --name equipay-canada-dashboard \
  --startup-file "streamlit run app/dashboard.py --server.port=8000 --server.address=0.0.0.0"

# Deploy code
zip -r deploy.zip . -x "*.git*" "venv/*" "__pycache__/*"
az webapp deployment source config-zip \
  --resource-group equipay-rg \
  --name equipay-canada-dashboard \
  --src deploy.zip

# Set environment variables
az webapp config appsettings set \
  --resource-group equipay-rg \
  --name equipay-canada-dashboard \
  --settings EQUIPAY_MODE=FULL

# Access: https://equipay-canada-dashboard.azurewebsites.net
```

### Deploy via Azure Portal:

1. Go to [Azure Portal](https://portal.azure.com)
2. Create **Web App** resource
3. Settings:
   - Runtime: Python 3.10
   - Operating System: Linux
   - Region: Choose closest to your users
4. Upload your code via FTP or GitHub Actions
5. Set startup command in Configuration → General settings

---

## 🟣 Option 4: Heroku

**Best for:** Quick prototypes, simple deployment

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create equipay-canada

# Add Procfile
cat > Procfile << EOF
web: streamlit run app/dashboard.py --server.port=\$PORT --server.address=0.0.0.0
EOF

# Deploy
git add .
git commit -m "Add Procfile for Heroku"
git push heroku main

# Set environment variables
heroku config:set EQUIPAY_MODE=FAST

# Open app
heroku open
```

**Create setup.sh for Heroku:**
```bash
cat > setup.sh << EOF
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF
```

---

## 🟠 Option 5: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 equipay-canada --region us-east-1

# Create environment
eb create equipay-env

# Deploy
eb deploy

# Open app
eb open
```

**Create .ebextensions/01_streamlit.config:**
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app/dashboard.py
  aws:elasticbeanstalk:application:environment:
    STREAMLIT_SERVER_PORT: 8501
    EQUIPAY_MODE: FAST
```

---

## 🔴 Option 6: Google Cloud Run (Serverless)

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/equipay-dashboard

# Deploy
gcloud run deploy equipay-dashboard \
  --image gcr.io/YOUR_PROJECT_ID/equipay-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

---

## 📦 Pre-Deployment Checklist

### 1. Data Preparation
```bash
# Ensure processed data exists
python scripts/convert_to_parquet.py

# Precompute aggregates for faster dashboard
python scripts/precompute_aggregates.py

# Verify data
ls -lh data/processed/
```

### 2. Environment Configuration
```bash
# Set production mode
export EQUIPAY_MODE=FULL

# Or for memory-constrained environments
export EQUIPAY_MODE=FAST
```

### 3. Test Locally First
```bash
# Test dashboard
streamlit run app/dashboard.py

# Test API
uvicorn api.main:app --reload

# Test Docker
docker-compose up
```

### 4. Security Hardening
```yaml
# Add to .streamlit/config.toml
[server]
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```

---

## 🔒 Environment Variables

Set these in your deployment platform:

| Variable | Default | Description |
|----------|---------|-------------|
| `EQUIPAY_MODE` | `FAST` | Data loading mode (`FAST` or `FULL`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`) |
| `PYTHONPATH` | `/app` | Python module search path |
| `STREAMLIT_SERVER_PORT` | `8501` | Dashboard port |
| `API_PORT` | `8000` | API port |

---

## 📊 Resource Requirements

### Minimum (FAST mode):
- **RAM:** 2GB
- **CPU:** 1 core
- **Storage:** 1GB
- **Cost:** ~$10-20/month

### Recommended (FULL mode):
- **RAM:** 4GB
- **CPU:** 2 cores
- **Storage:** 5GB
- **Cost:** ~$30-50/month

### Production (FULL mode + caching):
- **RAM:** 8GB
- **CPU:** 4 cores
- **Storage:** 10GB
- **Cost:** ~$100-150/month

---

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
# Check logs
docker logs equipay-dashboard

# Verify Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Out of Memory Errors
```bash
# Use FAST mode
export EQUIPAY_MODE=FAST

# Or increase container memory
docker run -m 4g equipay-dashboard
```

### Data Loading Issues
```bash
# Verify data exists
ls data/processed/lfs_processed.csv

# Regenerate if needed
python scripts/process_data_lowmem.py
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app/dashboard.py --server.port 8502
```

---

## 🚀 Quick Deployment Commands

### Deploy to Streamlit Cloud (Fastest)
```bash
git push
# Then deploy via web interface
```

### Deploy with Docker (Most Reliable)
```bash
docker-compose up -d
```

### Deploy to Azure (Enterprise)
```bash
az webapp up --name equipay-canada --runtime "PYTHON|3.10"
```

### Deploy to Heroku (Simple)
```bash
git push heroku main
```

---

## 📚 Additional Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Docker Documentation](https://docs.docker.com/)
- [Azure App Service Python](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
- [Heroku Python Support](https://devcenter.heroku.com/articles/python-support)

---

## 💡 Recommendations by Use Case

| Use Case | Best Option | Why |
|----------|-------------|-----|
| **Demo/POC** | Streamlit Cloud | Free, fast setup, no infrastructure |
| **Internal Tool** | Docker + VM | Full control, secure, cost-effective |
| **Public Dashboard** | Azure App Service | Scalable, enterprise-grade, monitoring |
| **Research Project** | Docker Compose | Reproducible, portable, includes API |
| **Client Delivery** | Cloud Run | Serverless, auto-scaling, pay-per-use |

---

## 🆘 Need Help?

- Check `logs/` directory for error messages
- Review `config.yaml` for configuration issues
- Ensure all data files are in `data/processed/`
- Set `EQUIPAY_MODE=FAST` for testing
- Verify Python version is 3.10+

**For production deployments, always:**
1. Test locally first with `docker-compose up`
2. Use `EQUIPAY_MODE=FULL` for complete analysis
3. Enable SSL/HTTPS for public access
4. Set up monitoring and logging
5. Configure automatic backups for data
