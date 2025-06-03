# 🚀 HuggingFace Spaces Deployment Guide

Complete step-by-step guide to deploy SceneSolver on Hugging Face Spaces.

## 📋 Pre-Deployment Checklist

### ✅ Required Files

- [x] `app.py` (Gradio application)
- [x] `requirements.txt` (Dependencies)
- [x] `crime_pipeline_few_shot.py` (Core pipeline)
- [x] `vit_pretrained_model.py` (ViT implementation)
- [x] `clip_finetune_few_shot.py` (CLIP implementation and training)
- [x] `vit_pretrained_evidence.pth` (332MB - ViT weights)
- [x] `clip_finetuned_few_shot.pth` (578MB - CLIP weights)
- [x] `.gitattributes` (Git LFS configuration)

### 📦 Model Files Summary

```
vit_pretrained_evidence.pth         332MB   ✅ Ready
clip_finetuned_few_shot.pth         578MB   ✅ Ready
Total Size:                         ~910MB
```

## 🔧 Step-by-Step Deployment

### Step 1: Create HuggingFace Space

1. **Go to HuggingFace**: Visit [huggingface.co/new-space](https://huggingface.co/new-space)

2. **Configure Space**:

   ```
   Space name: scenesolver-crime-analysis
   License: MIT
   SDK: Gradio
   Hardware: CPU Basic (Free) or GPU T4 small (Paid)
   Visibility: Public
   ```

3. **Click "Create Space"**

### Step 2: Local Setup

1. **Clone Your New Space**:

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/scenesolver-crime-analysis
   cd scenesolver-crime-analysis
   ```

2. **Initialize Git LFS**:
   ```bash
   git lfs install
   ```

### Step 3: File Upload

1. **Copy Core Files**:

   ```bash
   # Copy from your SceneSolver directory to the space directory:
   cp /path/to/SceneSolver/backendflask/app.py .
   cp /path/to/SceneSolver/backendflask/requirements.txt .
   cp /path/to/SceneSolver/backendflask/crime_pipeline_few_shot.py .
   cp /path/to/SceneSolver/backendflask/vit_pretrained_model.py .
   cp /path/to/SceneSolver/backendflask/clip_finetune_few_shot.py .
   cp /path/to/SceneSolver/backendflask/.gitattributes .
   ```

2. **Copy Model Files** (Large files - use Git LFS):

   ```bash
   cp /path/to/SceneSolver/backendflask/vit_pretrained_evidence.pth .
   cp /path/to/SceneSolver/backendflask/clip_finetuned_few_shot.pth .
   ```

3. **Optional Environment File**:
   ```bash
   cp /path/to/SceneSolver/backendflask/.env .
   ```

### Step 4: Git LFS Setup

1. **Track Large Files**:

   ```bash
   git lfs track "*.pth"
   git add .gitattributes
   ```

2. **Verify LFS Tracking**:
   ```bash
   git lfs ls-files
   # Should show your .pth files
   ```

### Step 5: Deploy

1. **Add All Files**:

   ```bash
   git add .
   ```

2. **Commit Changes**:

   ```bash
   git commit -m "🚀 Deploy SceneSolver with ViT and CLIP models

   - Add Gradio interface with crime scene analysis
   - Include VIT model (332MB) for image classification
   - Include CLIP model (578MB) for few-shot learning
   - Support multi-file upload (images/videos)
   - Add system health monitoring"
   ```

3. **Push to HuggingFace**:
   ```bash
   git push
   ```

### Step 6: Monitor Deployment

1. **Check Build Logs**: Visit your space URL and monitor the build process
2. **Verify Model Loading**: Check the logs for successful model initialization
3. **Test Interface**: Upload sample images to verify functionality

## 🎯 Expected URLs

After successful deployment:

- **Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/scenesolver-crime-analysis`
- **App URL**: `https://YOUR_USERNAME-scenesolver-crime-analysis.hf.space`
- **API Docs**: `https://YOUR_USERNAME-scenesolver-crime-analysis.hf.space/docs`

## 🔍 Troubleshooting

### Common Issues

1. **Large File Upload Issues**:

   ```bash
   # If Git LFS fails, try:
   git lfs push --all origin main
   ```

2. **Model Loading Errors**:

   - Check that all `.pth` files are uploaded correctly
   - Verify file paths in the code match uploaded files
   - Check HuggingFace logs for specific error messages

3. **Memory Issues**:

   - Consider upgrading to GPU T4 hardware
   - Check if models can be loaded with CPU-only inference

4. **Dependency Issues**:
   - Verify all packages in `requirements.txt` are compatible
   - Check for version conflicts in build logs

### Build Time Expectations

- **Without GPU**: 5-10 minutes
- **With GPU**: 3-7 minutes
- **Large files**: Additional 2-5 minutes for LFS

## 📊 Performance Optimization

### Hardware Recommendations

1. **CPU Basic (Free)**:

   - Suitable for testing
   - Slower inference (~10-15 seconds per image)
   - May timeout on large uploads

2. **GPU T4 Small (Paid)**:
   - Recommended for production
   - Fast inference (~2-3 seconds per image)
   - Better handling of multiple files

### Code Optimizations

- Models are loaded once at startup
- Temporary files are automatically cleaned
- Memory-efficient processing pipeline

## 🔐 Environment Variables (Optional)

Set in Space Settings → Variables and secrets:

```env
MONGODB_URL=mongodb+srv://...
MONGODB_DB_NAME=crime_db
MONGODB_COLLECTION_NAME=predictions
```

## ✅ Deployment Verification

### Test Checklist

1. **Basic Health Check**:

   - [ ] Space loads without errors
   - [ ] Health tab shows models available
   - [ ] System status is operational

2. **Single Image Analysis**:

   - [ ] Upload single JPG/PNG image
   - [ ] Click analyze button
   - [ ] Receive analysis results

3. **Multiple File Analysis**:

   - [ ] Upload multiple images
   - [ ] Successful batch processing
   - [ ] Individual results for each file

4. **Video Analysis**:
   - [ ] Upload MP4/MOV video
   - [ ] Frame extraction works
   - [ ] Video analysis completes

## 📞 Support

If you encounter issues:

1. **Check HuggingFace Logs**: Look for error messages in the space
2. **Verify File Upload**: Ensure all files are properly committed
3. **Test Locally**: Run `python app.py` locally first
4. **Community Help**: Ask on HuggingFace forums or Discord

---

**🎉 Congratulations!** Your SceneSolver application should now be deployed and accessible worldwide on HuggingFace Spaces!
