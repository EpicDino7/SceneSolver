import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import session from "express-session";
import passport from "passport";
import gauthRoutes from "./routes/gauth.js";
import authRoutes from "./routes/auth.js";
import uploadRoutes from "./routes/upload.js";
import "./config/passport.js";
import bodyParser from "body-parser";
import { checkSpaceHealth } from "./utils/huggingface.js";

dotenv.config();

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.urlencoded({ extended: true }));

// Updated CORS configuration for production
const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:3000",
  "https://scenesolver-backend.onrender.com", // Render backend URL
  "https://scenesolver-backend-2wb1.onrender.com", // Current Render backend URL
  "https://scenesolver.vercel.app", // Vercel frontend URL
  "https://scene-solver.vercel.app", // Alternative Vercel URL
  process.env.FRONTEND_URL,
].filter(Boolean);

app.use(
  cors({
    origin: allowedOrigins,
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "Accept"],
    exposedHeaders: ["Access-Control-Allow-Credentials"],
  })
);

app.use(function (req, res, next) {
  res.header("Access-Control-Allow-Credentials", "true");
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.header("Access-Control-Allow-Origin", origin);
  }
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
  );
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  next();
});

app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  console.log("Headers:", req.headers);
  console.log("Body:", req.body);
  next();
});

app.use(
  session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: false,
    cookie: {
      secure: process.env.NODE_ENV === "production",
      httpOnly: true,
      maxAge: 24 * 60 * 60 * 1000,
      sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
    },
    name: "sessionId",
  })
);

app.use(passport.initialize());
app.use(passport.session());

app.use((req, res, next) => {
  console.log("Session:", req.session);
  console.log("User:", req.user);
  next();
});

// Root endpoint
app.get("/", (req, res) => {
  res.status(200).json({
    message: "SceneSolver Backend API",
    status: "running",
    version: "1.0.0",
    timestamp: new Date().toISOString(),
    endpoints: {
      health: "/health",
      modelHealth: "/model-health",
      auth: "/api/auth",
      upload: "/api/upload",
    },
  });
});

// Health check endpoint
app.get("/health", (req, res) => {
  try {
    console.log("🏥 Health check requested");

    // More comprehensive health check for Railway
    const healthData = {
      status: "healthy",
      timestamp: new Date().toISOString(),
      server: "running",
      port: PORT,
      environment: process.env.NODE_ENV || "development",
      database:
        mongoose.connection.readyState === 1 ? "connected" : "disconnected",
      uptime: process.uptime(),
      memory: {
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      },
    };

    console.log("✅ Health check successful:", healthData);
    res.status(200).json(healthData);
  } catch (error) {
    console.error("❌ Health check error:", error);
    res.status(500).json({
      status: "unhealthy",
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

// Simplified health endpoint for Railway
app.get("/ping", (req, res) => {
  console.log("🏓 Ping requested");
  res.status(200).send("OK");
});

// Dedicated model health endpoint (kept separate for detailed model checking)
app.get("/model-health", async (req, res) => {
  try {
    console.log("🏥 Checking Hugging Face model health...");
    const healthCheck = await checkSpaceHealth();

    res.status(200).json({
      status: healthCheck.healthy ? "healthy" : "unhealthy",
      model_source: "huggingface_space",
      space_url: process.env.HF_SPACE_URL,
      details: healthCheck.details,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("❌ Model health check failed:", error);
    res.status(500).json({
      status: "error",
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

app.use("/api/auth", gauthRoutes);
app.use("/api/auth/local", authRoutes);
app.use("/api/upload", uploadRoutes);

const PORT = process.env.PORT || 5000;
const HOST = process.env.HOST || "0.0.0.0";

console.log("🚀 Starting SceneSolver Backend...");
console.log(`📍 Port: ${PORT}`);
console.log(`📍 Host: ${HOST}`);
console.log(`📍 Environment: ${process.env.NODE_ENV || "development"}`);
console.log(
  `📍 MongoDB URI: ${process.env.MONGODB_URI ? "✅ Set" : "❌ Missing"}`
);

// Start server first, then connect to MongoDB
const server = app.listen(PORT, HOST, () => {
  console.log(`🚀 Server running on ${HOST}:${PORT}`);
  console.log(`📊 Health check available at: http://${HOST}:${PORT}/health`);
  console.log(`🏓 Ping endpoint available at: http://${HOST}:${PORT}/ping`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || "development"}`);

  // Connect to MongoDB after server starts
  if (process.env.MONGODB_URI) {
    mongoose
      .connect(process.env.MONGODB_URI, {
        serverSelectionTimeoutMS: 5000, // Timeout after 5s instead of 30s
        socketTimeoutMS: 45000, // Close sockets after 45s of inactivity
      })
      .then(() => console.log("✅ Connected to MongoDB"))
      .catch((err) => {
        console.error("❌ MongoDB connection error:", err);
        // Don't exit process, let health check handle it
      });
  } else {
    console.warn("⚠️  MONGODB_URI not provided, skipping database connection");
  }
});

server.on("error", (err) => {
  console.error("❌ Server startup error:", err);
  if (err.code === "EADDRINUSE") {
    console.error(`Port ${PORT} is already in use`);
    process.exit(1);
  }
});

// Railway-specific: Handle SIGINT and SIGTERM for graceful shutdown
const gracefulShutdown = (signal) => {
  console.log(`\n${signal} received, shutting down gracefully...`);
  server.close(() => {
    console.log("🔄 HTTP server closed");
    mongoose.connection.close(false, () => {
      console.log("🔄 MongoDB connection closed");
      process.exit(0);
    });
  });
};

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));
