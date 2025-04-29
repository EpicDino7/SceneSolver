// server.js or app.js
const express = require("express");
const multer = require("multer");
const mongoose = require("mongoose");
const path = require("path");

// Set up MongoDB connection
mongoose.connect("mongodb://localhost:27017/crimeEvidenceDB", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Define a schema for files
const fileSchema = new mongoose.Schema({
  email: { type: String, required: true },
  filename: { type: String, required: true },
  fileType: { type: String, required: true },
  filePath: { type: String, required: true },
  uploadedAt: { type: Date, default: Date.now },
});

// Create a model for file storage
const File = mongoose.model("File", fileSchema);

// Set up file storage using Multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage });

const app = express();
app.use(express.json());

// Define POST route for file upload
app.post("/api/files", upload.array("file"), async (req, res) => {
  const { email } = req.body;

  if (!email || !req.files) {
    return res.status(400).json({ message: "Email and files are required" });
  }

  try {
    const filesToSave = req.files.map((file) => ({
      email: email,
      filename: file.filename,
      fileType: file.mimetype,
      filePath: file.path,
    }));

    // Save files to MongoDB
    await File.insertMany(filesToSave);

    res.status(200).json({ message: "Files uploaded successfully!" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error uploading files" });
  }
});

// Serve uploaded files (optional, for debugging or downloading)
app.use("/uploads", express.static("uploads"));

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
