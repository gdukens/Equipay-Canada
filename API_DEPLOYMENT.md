# FastAPI Deployment Guide

## 🔥 Railway (Easiest for FastAPI)

### 1-Click Deploy:
```bash
# Connect GitHub repo to Railway
# Auto-detects FastAPI and deploys
# Get: https://your-app.railway.app
```

## 🟢 Render (Free tier)

### Create `render.yaml`:
```yaml
services:
  - type: web
    name: equipay-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: EQUIPAY_MODE
        value: FAST
```

## 🔵 Azure Container Instances

```bash
# Deploy API container
az container create \
  --resource-group equipay-rg \
  --name equipay-api \
  --image equipayregistry.azurecr.io/equipay-api:latest \
  --dns-name-label equipay-api \
  --ports 8000 \
  --cpu 1 \
  --memory 2
```

## 🟠 Heroku

```bash
# Create Procfile for API
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile.api

# Deploy
heroku create equipay-api
git subtree push --prefix=api heroku main
```

## 🟣 Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/equipay-api .
gcloud run deploy equipay-api \
  --image gcr.io/PROJECT_ID/equipay-api \
  --platform managed \
  --allow-unauthenticated
```