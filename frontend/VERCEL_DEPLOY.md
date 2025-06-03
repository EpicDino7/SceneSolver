# 🚀 SceneSolver Frontend - Vercel Deployment Guide

## 🎯 Quick Deploy to Vercel

Your backend is live at: [https://scenesolver-backend-2wb1.onrender.com/](https://scenesolver-backend-2wb1.onrender.com/)

Now let's deploy your frontend!

## 🔧 Step-by-Step Deployment

### 1. **Commit Your Changes**

```bash
cd frontend
git add .
git commit -m "Configure frontend for Vercel deployment with Render backend"
git push origin main
```

### 2. **Deploy to Vercel**

#### Option A: **One-Click Deploy (Recommended)**

1. Go to [vercel.com](https://vercel.com)
2. Sign up/login with your GitHub account
3. Click **"Add New Project"**
4. Import your `SceneSolver-clean` repository
5. **Root Directory**: Select `frontend`
6. **Framework Preset**: Vite
7. Click **"Deploy"**

#### Option B: **Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from frontend directory
cd frontend
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? [Your account]
# - Link to existing project? No
# - Project name: scenesolver or scene-solver
# - Directory: ./ (current directory)
# - Want to override settings? Yes
# - Output directory: dist
# - Build command: npm run build
```

### 3. **Configuration During Deploy**

Vercel will auto-detect your Vite React app. Ensure these settings:

| Setting              | Value           |
| -------------------- | --------------- |
| **Framework**        | Vite            |
| **Root Directory**   | `frontend`      |
| **Build Command**    | `npm run build` |
| **Output Directory** | `dist`          |
| **Install Command**  | `npm install`   |

### 4. **Environment Variables**

Vercel will automatically use the environment variable from `vercel.json`:

```json
"env": {
  "VITE_API_URL": "https://scenesolver-backend-2wb1.onrender.com"
}
```

If needed, you can also set it manually in Vercel dashboard:

- Go to Project Settings → Environment Variables
- Add: `VITE_API_URL` = `https://scenesolver-backend-2wb1.onrender.com`

## 📊 What to Expect

### **Build Process:**

```
✅ Installing dependencies...
✅ Building with Vite...
✅ Optimizing assets...
✅ Deploying to Vercel Edge Network...
✅ Your app is live!
```

### **Deployment Time:**

- **First Deploy**: 1-2 minutes
- **Subsequent Deploys**: 30-45 seconds

### **Your URLs:**

- **Production**: `https://scenesolver.vercel.app` (or similar)
- **Preview**: Unique URL for each commit

## 🔍 Testing Your Deployment

### 1. **Test Frontend**

Visit your Vercel URL and verify:

- ✅ App loads successfully
- ✅ UI components render correctly
- ✅ No console errors

### 2. **Test Backend Connection**

Test the API connection:

- ✅ Upload functionality works
- ✅ Authentication flows work
- ✅ No CORS errors in browser console

### 3. **Test Full Flow**

- ✅ Upload crime scene images
- ✅ Get analysis results
- ✅ Generate reports

## 🚨 Troubleshooting

### **Build Fails:**

- Check the build logs in Vercel dashboard
- Ensure all dependencies are in `package.json`
- Verify no TypeScript errors

### **CORS Errors:**

- Your backend already includes Vercel URLs in CORS
- If you get a different Vercel URL, update the backend CORS settings

### **API Connection Issues:**

- Verify `VITE_API_URL` is set correctly
- Check browser network tab for failed requests
- Ensure backend is still running on Render

## 🔄 Auto-Deployments

Vercel will automatically deploy when you:

- Push to `main` branch
- Make changes to the `frontend` folder
- Update environment variables

## 🎉 Success!

After deployment, you'll have:

- ✅ **Frontend**: Live on Vercel with global CDN
- ✅ **Backend**: Live on Render with health monitoring
- ✅ **Full Integration**: Frontend talking to backend seamlessly
- ✅ **Auto-deployments**: Both services update automatically

## 🔗 Final URLs

- **Frontend (Vercel)**: `https://[your-app].vercel.app`
- **Backend (Render)**: `https://scenesolver-backend-2wb1.onrender.com`
- **API Health**: `https://scenesolver-backend-2wb1.onrender.com/health`

## 💡 Pro Tips

1. **Custom Domain**: Easy to add in Vercel dashboard
2. **Analytics**: Enable Vercel Analytics for usage insights
3. **Preview Deployments**: Every branch gets a preview URL
4. **Rollbacks**: One-click rollback to previous versions

Need help? Check the deployment logs or reach out! 🚀
