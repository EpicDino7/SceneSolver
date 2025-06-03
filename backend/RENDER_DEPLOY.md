.# 🚀 SceneSolver Backend - Render Deployment Guide

## Why Render?

- ✅ More reliable than Railway
- ✅ Better error messages and debugging
- ✅ Excellent free tier
- ✅ Auto-deployments from GitHub
- ✅ Built-in SSL certificates

## 🔧 Step-by-Step Deployment

### 1. **Commit Your Changes**

First, commit all the latest changes to your repository:

```bash
git add .
git commit -m "Switch to Render deployment configuration"
git push origin main
```

### 2. **Sign Up for Render**

- Go to [render.com](https://render.com)
- Sign up with your GitHub account
- Connect your GitHub repository

### 3. **Create a New Web Service**

1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repository: `SceneSolver-clean`
3. Select the `backend` folder (if prompted)

### 4. **Configuration Settings**

Use these **exact** settings:

| Field              | Value                      |
| ------------------ | -------------------------- |
| **Name**           | `scenesolver-backend`      |
| **Environment**    | `Node`                     |
| **Build Command**  | `npm install --production` |
| **Start Command**  | `npm start`                |
| **Branch**         | `main`                     |
| **Root Directory** | `backend`                  |

### 5. **Advanced Settings**

Click **"Advanced"** and set:

- **Health Check Path**: `/ping`
- **Auto-Deploy**: `Yes`

### 6. **Environment Variables**

Add these environment variables in Render (copy values from your `.env` file):

```
NODE_ENV=production
MONGODB_URI=[copy from your .env file]
GEMINI_API_KEY=[copy from your .env file]
GOOGLE_CLIENT_ID=[copy from your .env file]
GOOGLE_CLIENT_SECRET=[copy from your .env file]
SESSION_SECRET=[copy from your .env file]
JWT_SECRET=[copy from your .env file]
EMAIL_USER=[copy from your .env file]
EMAIL_PASSWORD=[copy from your .env file]
HF_SPACE_URL=https://epicdino-scenesolvermodels.hf.space
HF_SPACE_NAME=EpicDino/SceneSolverModels
FRONTEND_URL=https://your-vercel-app.vercel.app
```

⚠️ **Important:** Replace the bracketed placeholders with actual values from your `.env` file.

### 7. **Deploy!**

Click **"Create Web Service"** and watch the deployment!

## 📊 What to Expect

### **Build Process:**

```
==> Building...
==> Installing dependencies with npm install --production
==> Starting with npm start
==> Health check at /ping: ✅ SUCCESS
==> Your service is live!
```

### **Deployment Time:**

- **First Deploy**: 2-3 minutes
- **Subsequent Deploys**: 30-60 seconds

### **Your URLs:**

- **API Base**: `https://scenesolver-backend.onrender.com`
- **Health Check**: `https://scenesolver-backend.onrender.com/ping`
- **Full Health**: `https://scenesolver-backend.onrender.com/health`

## 🔍 Testing Your Deployment

### 1. **Test Health Endpoints**

```bash
# Simple health check
curl https://scenesolver-backend.onrender.com/ping

# Detailed health check
curl https://scenesolver-backend.onrender.com/health
```

## 🎉 You're Done!

Your SceneSolver backend is now deployed on Render with:

- ✅ Automatic deployments
- ✅ Health monitoring
- ✅ SSL certificates
- ✅ Better reliability than Railway

Need help? Check the Render logs or reach out!
