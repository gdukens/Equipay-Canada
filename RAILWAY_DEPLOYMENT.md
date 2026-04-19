# Railway Deployment Guide - EquiPay Canada

Railway deployment with size optimizations to stay under 4GB limit.

## 🚄 Railway Deployment (Size-Optimized)

Railway has a **4GB image size limit** on the free tier. Our optimized deployment excludes large data files and uses sample data for demonstration.

### Quick Deploy to Railway

1. **Push optimized code to GitHub:**
   ```bash
   git add railway_start.py Dockerfile.railway railway.yml
   git commit -m "Add Railway deployment optimization"
   git push
   ```

2. **Deploy API Service:**
   - Visit [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Set build configuration:
     ```
     Dockerfile: Dockerfile.railway
     ```
   - Set environment variables:
     ```
     EQUIPAY_MODE=RAILWAY
     SERVICE_TYPE=api
     LOG_LEVEL=INFO
     ```

3. **Deploy Dashboard Service (separate deployment):**
   - Create second Railway service from same repo
   - Set environment variables:
     ```
     EQUIPAY_MODE=RAILWAY
     SERVICE_TYPE=dashboard
     LOG_LEVEL=INFO
     ```

### Size Optimization Features

 **Excluded from Railway build:**
- Large CSV data files (2GB+)
- Parquet cache files (300MB+) 
- Model artifacts (when large)
- Development artifacts

 **Included sample data:**
- 10,000 synthetic records (~1MB)
- Maintains data structure & relationships
- Demonstrates full application functionality
- Realistic wage gaps & demographics

 **Smart data handling:**
- Falls back to sample data when large files unavailable
- Maintains all analysis capabilities
- Railway-specific environment detection

### Environment Variables

| Variable | Value | Purpose |
|----------|--------|---------|
| `EQUIPAY_MODE` | `RAILWAY` | Enables sample data mode |
| `SERVICE_TYPE` | `api` or `dashboard` | Which service to start |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | (auto-set by Railway) | Service port |

### Architecture on Railway

```
┌─────────────────────┐    ┌─────────────────────┐
│   API Service       │    │  Dashboard Service  │
│   (Dockerfile.railway) │    │  (Dockerfile.railway) │
│                     │    │                     │
│   • FastAPI         │    │   • Streamlit       │
│   • Sample data     │───▶│   • Sample data     │
│   • Swagger UI      │    │   • Full analysis   │
└─────────────────────┘    └─────────────────────┘
```

### Deployment Commands

```bash
# Test locally first
docker build -f Dockerfile.railway -t equipay-railway .
docker run -p 8000:8000 -e SERVICE_TYPE=api equipay-railway
docker run -p 8501:8501 -e SERVICE_TYPE=dashboard equipay-railway

# Deploy to Railway
railway login
railway link [your-project]
railway up
```

### Cost & Scaling

- **Free tier**: 500 hours/month, 4GB image limit
- **Pro tier**: $5/month, unlimited usage, larger images
- **Automatic scaling**: Railway handles traffic spikes
- **Custom domains**: Available on all plans

### Troubleshooting

**Build fails with size error:**
- Check Dockerfile.railway excludes large files
- Verify .dockerignore is not too restrictive
- Consider Pro tier for larger images

**Sample data issues:**
- Check `railway_start.py` runs successfully
- Verify `EQUIPAY_MODE=RAILWAY` is set
- Review logs for data creation errors

**Service won't start:**
- Check `SERVICE_TYPE` environment variable
- Verify port configuration (Railway auto-assigns PORT)
- Check application logs in Railway dashboard

### Alternative: Data Upload Strategy

For production deployment with full dataset on Railway Pro:

1. **Upload data to Railway Volume:**
   ```bash
   railway volume create data-volume
   railway volume mount data-volume:/app/data
   ```

2. **Use standard Dockerfile:**
   ```yaml
   # railway.yml
   build:
     dockerfile: Dockerfile  # Use full Dockerfile
   
   volumes:
     - data-volume:/app/data
   ```

3. **Upload data separately:**
   ```bash
   railway volume upload data-volume ./data/
   ```

This allows full dataset deployment on Railway Pro tier with persistent storage.

## Production Notes

- Sample data demonstrates full functionality
- Upgrade to Railway Pro for production workloads
- Consider external database for large datasets
- Monitor usage and costs through Railway dashboard

The Railway deployment provides a quick way to demonstrate EquiPay Canada's capabilities without requiring large data files, making it perfect for showcases and development environments.
