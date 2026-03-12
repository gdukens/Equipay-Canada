# Quick Deployment Guide

## ✅ Prerequisites Verified
- ✓ Docker installed (v28.5.1)
- ✓ Docker Compose available (v2.40.0)
- ✓ Data processed and ready
- ✓ Streamlit config created

---

## 🐳 Deploy 1: Docker (Local/Production)

### Start the Dashboard:

```bash
# Build and start (first time)
docker compose up --build -d

# Or just start (after first build)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f dashboard

# Stop when done
docker compose down
```

**Access:**
- Dashboard: http://localhost:8501
- API: http://localhost:8000

### Troubleshooting:
```bash
# Rebuild from scratch
docker compose down --volumes --rmi all
docker compose up --build -d

# Check container logs
docker logs equipay-canada-dashboard-1

# Access container shell
docker compose exec dashboard bash
```

---

## ☁️ Deploy 2: Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
# Make sure you're on main/master branch
git status

# Add deployment files
git add .streamlit/ Procfile packages.txt DEPLOYMENT.md
git add -A

# Commit
git commit -m "Add Streamlit Cloud deployment config"

# Push to GitHub
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to: https://share.streamlit.io
2. Click **"New app"**
3. Connect GitHub account (if not already)
4. Select:
   - **Repository**: `your-username/equipay-canada`
   - **Branch**: `main` or `master`
   - **Main file path**: `app/dashboard.py`
5. **Advanced settings** (optional):
   - Python version: `3.10`
   - Environment variables:
     ```
     EQUIPAY_MODE=FAST
     ```
6. Click **"Deploy!"**

Wait 2-5 minutes for deployment.

### Step 3: Access Your App

You'll get a URL like:
```
https://your-username-equipay-canada-app-dashboard-xyz123.streamlit.app
```

Share this URL with anyone!

### Updating Streamlit Cloud:
```bash
# Just push changes to GitHub
git add .
git commit -m "Update dashboard"
git push

# Streamlit Cloud auto-deploys in ~1 minute
```

---

## 🎯 Quick Comparison

| Feature | Docker | Streamlit Cloud |
|---------|--------|-----------------|
| **Cost** | Free (your machine) | Free (1GB RAM) |
| **Speed** | Fast (local) | Depends on location |
| **Setup Time** | 30 seconds | 2 minutes |
| **Access** | localhost only | Public URL |
| **Data** | Full dataset | Use FAST mode |
| **Updates** | Rebuild container | Git push |
| **Best For** | Development/Testing | Sharing/Demos |

---

## 🚀 Recommended Workflow

1. **Develop locally** with Docker:
   ```bash
   docker compose up -d
   # Edit code, test at localhost:8501
   ```

2. **Share publicly** with Streamlit Cloud:
   ```bash
   git push
   # Share URL with stakeholders
   ```

3. **Full production** (if needed):
   - Deploy Docker image to Azure/AWS
   - See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📊 Current Status

✅ Ready to deploy to Docker right now
✅ Ready to deploy to Streamlit Cloud (need GitHub push first)

### Next Steps:

**For Docker:**
```bash
docker compose up -d
```

**For Streamlit Cloud:**
1. `git add -A`
2. `git commit -m "Ready for deployment"`
3. `git push`
4. Go to share.streamlit.io

---

## 🆘 Common Issues

### Docker: Port already in use
```bash
# Stop existing Streamlit
Get-Process -Name streamlit | Stop-Process

# Or change port in docker-compose.yml
ports:
  - "8502:8501"  # Changed from 8501
```

### Streamlit Cloud: Out of memory
Set in app settings:
```
EQUIPAY_MODE=FAST
```

### Docker: Data not loading
```bash
# Verify data volume
docker compose exec dashboard ls -lh data/processed/
```

### Streamlit Cloud: Import errors
Check `requirements.txt` has all dependencies.

---

## 💡 Pro Tips

1. **Use FAST mode for Streamlit Cloud** (free tier has 1GB RAM limit)
2. **Use FULL mode for Docker** (you have plenty of resources)
3. **Test Docker locally before cloud deployment**
4. **Keep sensitive data out of GitHub** (your processed data is already gitignored)
5. **Monitor Streamlit Cloud usage** at your app dashboard

---

## 🎉 You're All Set!

Both deployment options are configured and ready. Pick your use case:

- **Want to test now?** → Docker
- **Want to share with others?** → Streamlit Cloud  
- **Want both?** → Do both! They don't conflict.
