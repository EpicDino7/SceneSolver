# SceneSolver Backend Flask - Deployment Guide

## Hosting Options

### Option 1: Hugging Face Spaces (Recommended)

Best for ML models with large files:

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space with Gradio SDK
3. Upload your files:
   - `app_hf.py` → rename to `app.py`
   - `requirements_hf.txt` → rename to `requirements.txt`
   - All model files (`.pth` files)
   - All Python modules (`crime_pipeline_few_shot.py`, etc.)

### Option 2: Railway/Render

For regular Flask deployment:

1. Use `app.py` (original file)
2. Use `requirements.txt` (original file)
3. Set environment variables in hosting dashboard

## Environment Variables

Required for all hosting platforms:

```
MONGODB_URL=your_mongodb_connection_string
MONGODB_DB_NAME=crime_db
MONGODB_COLLECTION_NAME=predictions
FLASK_ENV=production
UPLOAD_FOLDER=uploads
FRAMES_FOLDER=frames
```

## Model Files

Large model files (total ~1GB):

- `clip_finetuned_few_shot.pth` (578MB)
- `vit_pretrained_evidence.pth` (332MB)
- `clip_model_weights.pth` (74MB)
- `vit_model.pth` (12MB)

**Important:** Upload these to your hosting platform's file storage or use Git LFS.

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - File upload and analysis

## Usage

After deployment, your Flask API will be available at:

- Hugging Face: `https://your-username-your-space.hf.space`
- Railway: `https://your-app.railway.app`
- Render: `https://your-app.onrender.com`
