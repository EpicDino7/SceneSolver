# SceneSolver Backend Deployment Guide

## 🚀 Updated to Use Hugging Face Space Model

Your backend has been updated to use the deployed Hugging Face Space model instead of the local Flask API.

## Changes Made

### ✅ **New Dependencies**

- Added `form-data` for API calls to Hugging Face

### ✅ **New Files**

- `utils/huggingface.js` - Handles all HF Space communication
- `deploy.md` - This deployment guide

### ✅ **Updated Files**

- `routes/upload.js` - Now uses HF Space API instead of Flask
- `index.js` - Added model health check endpoints
- `package.json` - Added form-data dependency
- `.env` - Added HF Space configuration

## 🔧 Deployment Steps

### 1. Install Dependencies

```bash
cd backend
npm install
```

### 2. Environment Variables

Make sure your `.env` file includes:

```env
# Existing variables...
MONGODB_URI=your_mongodb_uri
GEMINI_API_KEY=your_gemini_key
# ... other vars

# NEW: Hugging Face Space Configuration
HF_SPACE_URL=https://EpicDino-scenesolver.hf.space
HF_SPACE_NAME=EpicDino/SceneSolverModels
```

### 3. Test Locally

```bash
npm run dev
```

### 4. Test Endpoints

- Health Check: `GET /health`
- Model Health: `GET /model-health`
- Upload: `POST /api/upload`

### 5. Deploy to Production

Update your production environment variables to include the HF configuration.

## 🔌 API Changes

### **Upload Endpoint**

- **Before**: Connected to local Flask API at `127.0.0.1:5000`
- **After**: Connects to HF Space at `https://EpicDino-scenesolver.hf.space`

### **New Endpoints**

- `GET /model-health` - Check HF model status
- `GET /api/upload/model-health` - Same but under upload route

### **Response Changes**

Upload responses now include:

```json
{
  "message": "Files uploaded and analyzed successfully",
  "files": [...],
  "caseResult": {
    "predicted_class": "Robbery",
    "crime_confidence": 0.89,
    "extracted_evidence": [...],
    "model_type": "few-shot-finetuned"
  },
  "modelSource": "huggingface_space"
}
```

## 🐛 Troubleshooting

### Model Not Available

```json
{
  "error": "Model not available: Space is starting up"
}
```

**Solution**: Wait 2-3 minutes for HF Space to load models

### Analysis Timeout

```json
{
  "error": "Analysis timeout - results not received"
}
```

**Solution**: Check HF Space status, may need GPU upgrade

### Connection Errors

- Verify HF_SPACE_URL in environment
- Check if Space is running at the URL
- Ensure internet connectivity

## 📊 Performance Notes

- **First Request**: May take 30-60 seconds (cold start)
- **Subsequent Requests**: 5-15 seconds
- **Large Files**: May take longer, max 50MB per file
- **Timeouts**: 3 minutes for analysis, 30 seconds for health

## 🔄 Rollback Plan

If issues occur, you can quickly rollback by:

1. Reverting to the Flask API endpoints in `upload.js`
2. Commenting out HF imports
3. Restarting your local Flask server

## 🌐 Production Deployment

For Railway/Heroku/other platforms:

1. Add HF environment variables
2. Ensure `form-data` dependency is installed
3. Update CORS settings if needed
4. Monitor logs for HF API responses

## 📈 Monitoring

Monitor these endpoints for health:

- `/health` - Overall system health
- `/model-health` - HF Space model status
- Check logs for HF API response times
