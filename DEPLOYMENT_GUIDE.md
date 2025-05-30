# 🚀 SceneSolver Deployment Guide

Complete guide to deploy your SceneSolver application with free hosting services.

## 📋 Overview

- **Frontend (React)**: Vercel
- **Backend (Node.js)**: Railway
- **Backend Flask (ML)**: Hugging Face Spaces

## 🏗️ **Repository Structure Options**

### **Option 1: Monorepo (Recommended) ✅**

Keep all folders in one repository and deploy from subfolders:

```
SceneSolver-clean/
├── frontend/          # Deploy to Vercel from this folder
├── backend/           # Deploy to Railway from this folder
├── backendflask/      # Deploy to Hugging Face from this folder
└── DEPLOYMENT_GUIDE.md
```

**Advantages:**

- ✅ Single repository to manage
- ✅ Synchronized version control
- ✅ Shared documentation
- ✅ All platforms support subfolder deployment

### **Option 2: Separate Repositories**

Split into three repositories if preferred:

```
├── scenesolver-frontend/     # Separate repo for Vercel
├── scenesolver-backend/      # Separate repo for Railway
└── scenesolver-ml/          # Separate repo for Hugging Face
```

## 🎯 Step-by-Step Deployment

### 1. Frontend Deployment (Vercel)

#### Prerequisites

- GitHub account
- Vercel account (free)

#### Steps (Monorepo)

1. Push your entire repository to GitHub
2. Connect Vercel to GitHub
3. Import your repository
4. **Set Root Directory to `frontend`** in project settings
5. Set environment variables in Vercel dashboard:
   ```
   VITE_API_URL=https://your-railway-app.railway.app
   VITE_FLASK_API_URL=https://your-hf-space.hf.space
   ```
6. Deploy!

#### Steps (Separate Repo)

1. Copy `frontend/` contents to new repository
2. Push to GitHub
3. Connect Vercel (no root directory needed)
4. Set environment variables and deploy

### 2. Backend Node.js Deployment (Railway)

#### Prerequisites

- GitHub account
- Railway account (free $5/month credit)

#### Steps (Monorepo)

1. Push your entire repository to GitHub (if not done already)
2. Connect Railway to GitHub
3. Create new project from GitHub repo
4. **Set Root Directory to `backend`** in project settings
5. Set environment variables in Railway dashboard:
   ```
   NODE_ENV=production
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   SESSION_SECRET=your_session_secret
   MONGODB_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   GEMINI_API_KEY=your_gemini_api_key
   EMAIL_USER=your_email
   EMAIL_PASSWORD=your_email_password
   FRONTEND_URL=https://your-vercel-app.vercel.app
   ```
6. Railway will auto-deploy!

#### Steps (Separate Repo)

1. Copy `backend/` contents to new repository
2. Push to GitHub
3. Connect Railway (no root directory needed)
4. Set environment variables and deploy

#### Alternative: Render

- Similar process to Railway
- Free tier with 512MB RAM
- Sleeps after 15 minutes of inactivity

### 3. Backend Flask Deployment (Hugging Face Spaces)

#### Prerequisites

- Hugging Face account (free)

#### Steps (Monorepo or Separate Repo)

1. Create new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Gradio** as SDK
3. Upload files from `backendflask/` folder to Space:
   - Rename `app_hf.py` → `app.py`
   - Rename `requirements_hf.txt` → `requirements.txt`
   - Upload all `.pth` model files
   - Upload all Python modules (`crime_pipeline_few_shot.py`, etc.)
4. Set environment variables in Space settings:
   ```
   MONGODB_URL=your_mongodb_connection_string
   MONGODB_DB_NAME=crime_db
   MONGODB_COLLECTION_NAME=predictions
   FLASK_ENV=production
   ```
5. Space will auto-deploy!

**Note:** Hugging Face Spaces work best with manual file upload due to large model files.

#### Alternative: Railway (without large models)

If you can reduce model sizes or use external model storage:

1. Use original `app.py` file
2. Set up same as Node.js backend

## 🔗 Connecting the Services

### Update Frontend URLs

After deployment, update your frontend code to use production URLs:

```javascript
// In your React app
const API_URL = process.env.VITE_API_URL || "http://localhost:5000";
const FLASK_API_URL = process.env.VITE_FLASK_API_URL || "http://localhost:5001";
```

### Update Backend CORS

Update the `allowedOrigins` in `backend/index.js`:

```javascript
const allowedOrigins = [
  "http://localhost:5173",
  "https://your-actual-vercel-url.vercel.app", // Replace with actual URL
  process.env.FRONTEND_URL,
].filter(Boolean);
```

## 💾 Database Setup

### MongoDB Atlas (Free)

1. Create account at [mongodb.com](https://mongodb.com)
2. Create free cluster
3. Create database user
4. Whitelist all IPs (0.0.0.0/0) for production
5. Get connection string for environment variables

## 🔑 Environment Variables Summary

### Frontend (Vercel)

```
VITE_API_URL=https://your-railway-app.railway.app
VITE_FLASK_API_URL=https://your-hf-space.hf.space
```

### Backend Node.js (Railway)

```
NODE_ENV=production
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
SESSION_SECRET=your_session_secret
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database
JWT_SECRET=your_jwt_secret
GEMINI_API_KEY=your_gemini_api_key
EMAIL_USER=your_email
EMAIL_PASSWORD=your_email_app_password
FRONTEND_URL=https://your-vercel-app.vercel.app
```

### Backend Flask (Hugging Face)

```
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/database
MONGODB_DB_NAME=crime_db
MONGODB_COLLECTION_NAME=predictions
FLASK_ENV=production
```

## 🏗️ Build Commands

All configured in deployment files:

- **Frontend**: `npm run build` (automatic on Vercel)
- **Backend Node.js**: `npm start` (configured in railway.toml)
- **Backend Flask**: Python app auto-detected

## 📊 Monitoring & Health Checks

### Health Check Endpoints

- Node.js Backend: `GET /health`
- Flask Backend: Built into Gradio interface

### Monitoring

- Vercel: Built-in analytics
- Railway: Built-in metrics
- Hugging Face: Space analytics

## 🚨 Common Issues & Solutions

### CORS Issues

- Update `allowedOrigins` in backend
- Ensure environment variables are set correctly

### Model Loading Issues (Flask)

- Ensure all `.pth` files are uploaded
- Check file paths in code
- Use Git LFS for large files on GitHub

### Session Issues (Node.js)

- Update cookie settings for production
- Ensure `SESSION_SECRET` is set
- Update `sameSite` settings for cross-domain

### MongoDB Connection Issues

- Whitelist all IPs in MongoDB Atlas
- Verify connection string format
- Check network restrictions

## 💡 Tips for Success

1. **Test locally first** - Ensure everything works before deployment
2. **Environment variables** - Never commit secrets to git
3. **Model files** - Use Git LFS or direct upload for large files
4. **CORS configuration** - Update for production domains
5. **Health checks** - Monitor all services regularly

## 📱 Mobile Compatibility

All services are mobile-friendly:

- Vercel: Automatic responsive design
- Railway: API works on all devices
- Hugging Face: Gradio responsive interface

## 💰 Cost Breakdown (Free Tiers)

- **Vercel**: Free (100GB bandwidth/month)
- **Railway**: Free ($5 credit/month)
- **Hugging Face**: Free (with some usage limits)
- **MongoDB Atlas**: Free (512MB storage)

**Total Monthly Cost: $0** 🎉

## 🆘 Need Help?

- Check service status pages
- Review logs in respective dashboards
- Test health endpoints
- Verify environment variables

Ready to deploy! 🚀
