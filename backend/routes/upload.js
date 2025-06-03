import express from "express";
import multer from "multer";
import mongoose from "mongoose";
import dotenv from "dotenv";
import { Readable } from "stream";
import User from "../models/user.js";
import Guser from "../models/Guser.js";
import { MongoClient, ObjectId } from "mongodb";
import axios from "axios";
import FormData from "form-data";
import {
  analyzeCrimeScene,
  checkSpaceHealth,
  validateFiles,
} from "../utils/huggingface.js";

dotenv.config();

const router = express.Router();
const mongoURI = process.env.MONGODB_URI;

const conn = mongoose.createConnection(mongoURI);

let gridfsBucket;
conn.once("open", () => {
  gridfsBucket = new mongoose.mongo.GridFSBucket(conn.db, {
    bucketName: "uploads",
  });
  console.log("GridFS bucket initialized");
});

const storage = multer.memoryStorage();
const upload = multer({ storage });

const streamToGridFS = (fileBuffer, options) => {
  return new Promise((resolve, reject) => {
    const readableStream = new Readable();
    readableStream.push(fileBuffer);
    readableStream.push(null);

    const uploadStream = gridfsBucket.openUploadStream(options.filename, {
      contentType: options.contentType,
      metadata: options.metadata,
    });

    let fileId = uploadStream.id;

    readableStream.pipe(uploadStream);

    uploadStream.on("error", (error) => {
      reject(error);
    });

    uploadStream.on("finish", () => {
      resolve({
        _id: fileId,
        filename: options.filename,
        contentType: options.contentType,
        metadata: options.metadata,
      });
    });
  });
};

// Health check endpoint for Hugging Face model
router.get("/model-health", async (req, res) => {
  try {
    console.log("🏥 Checking Hugging Face model health...");
    const healthCheck = await checkSpaceHealth();

    res.status(200).json({
      status: healthCheck.healthy ? "healthy" : "unhealthy",
      model_source: "huggingface_space",
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

router.post("/", upload.array("files"), async (req, res) => {
  console.log("upload post route reached");

  try {
    if (!gridfsBucket) {
      return res.status(500).json({ error: "GridFS not initialized" });
    }

    const files = req.files;
    const { email, caseName, location, date, crimeTime } = req.body;

    console.log("Files received:", files.length);
    console.log("Uploaded by user:", email);

    const user =
      (await User.findOne({ email })) || (await Guser.findOne({ email }));

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const uId = user._id;

    if (!files || files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    // Validate files before processing
    try {
      validateFiles(files);
    } catch (validationError) {
      return res.status(400).json({ error: validationError.message });
    }

    const images = files.filter((f) => f.mimetype.startsWith("image"));
    const videos = files.filter((f) => f.mimetype.startsWith("video"));

    if (
      (images.length > 0 && images.length < 4) ||
      (images.length === 0 && videos.length !== 1)
    ) {
      return res
        .status(400)
        .json({ error: "Must upload at least 4 images or 1 video" });
    }

    // Analyze crime scene using Hugging Face model
    console.log("🔍 Starting crime scene analysis with HF model...");
    let caseResult = null;
    let analysisError = null;

    try {
      const analysisResponse = await analyzeCrimeScene(files);
      caseResult = analysisResponse.result;
      console.log("✅ HF Analysis completed:", caseResult);
    } catch (error) {
      console.error("❌ HF Analysis failed:", error);
      analysisError = error.message || "Analysis failed";

      // Continue with upload but note the analysis failure
      caseResult = {
        error: "Analysis failed",
        details: analysisError,
        predicted_class: "Unknown",
        crime_confidence: 0,
        extracted_evidence: [],
        model_type: "huggingface_space_error",
      };
    }

    // Generate Gemini report if analysis was successful
    let geminiResponse = null;
    if (caseResult && !caseResult.error) {
      try {
        const geminiRes = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: `You are a professional crime investigation expert. Analyze the given crime scene data and generate a structured and formal crime scene report. Use the details to explain what likely happened, in clear and straightforward language.

                      Crime Scene Details:
                      - Location: ${location || "Location not provided"}
                      - Date: ${date || "Date not provided"}
                      - Time: ${crimeTime || "Time not provided"}
                      - Analysis Results: ${JSON.stringify(caseResult)}

                      The analysis results include:
                      - predicted_class: the predicted type of crime (${
                        caseResult.predicted_class
                      })
                      - crime_confidence: the model's confidence in the prediction (${
                        caseResult.crime_confidence
                      })
                      - extracted_evidence: a list of visual evidence with description and confidence levels
                      - model_type: ${
                        caseResult.model_type || "huggingface_space"
                      }

                      Your output must follow this exact structure but do not include any markdown formatting or asterisks:

                      CRIME SCENE INVESTIGATION REPORT
                      ===============================

                      **Case Details:**
                      Location: ${location || "Not specified"}
                      Date: ${date || "Not specified"}
                      Time: ${crimeTime || "Not specified"}
                      Crime Type: ${caseResult.predicted_class}
                      Analysis Confidence: ${Math.round(
                        (caseResult.crime_confidence || 0) * 100
                      )}%
                      Analysis Source: AI Crime Scene Analyzer (HuggingFace Space)

                      **Physical Evidence Analysis:**
                      ${
                        caseResult.extracted_evidence
                          ?.map(
                            (evidence) =>
                              `- ${evidence.label} (Confidence: ${Math.round(
                                evidence.confidence * 100
                              )}%)\n  Significance: Evidence type commonly associated with ${
                                caseResult.predicted_class
                              } incidents`
                          )
                          .join("\n") || "No specific evidence extracted"
                      }

                      **Primary Scene Analysis:**
                      Based on the AI analysis indicating ${
                        caseResult.predicted_class
                      } with ${Math.round(
                        (caseResult.crime_confidence || 0) * 100
                      )}% confidence, the scene suggests ${caseResult.predicted_class.toLowerCase()} activity occurred at this location. The detected evidence patterns are consistent with this classification. ${
                        caseResult.extracted_evidence?.length > 0
                          ? `Key evidence includes ${caseResult.extracted_evidence
                              .map((e) => e.label.toLowerCase())
                              .join(", ")}.`
                          : ""
                      }

                      **Alternative Scenario:**
                      While the primary analysis suggests ${
                        caseResult.predicted_class
                      }, investigators should consider alternative explanations for the observed evidence patterns. The confidence level of ${Math.round(
                        (caseResult.crime_confidence || 0) * 100
                      )}% indicates some uncertainty that warrants additional investigation.

                      **Recommendations:**
                      1. Verify AI analysis findings through physical evidence collection
                      2. Interview witnesses and gather additional contextual information
                      3. Cross-reference findings with similar case patterns
                      4. Collect additional photographic evidence if needed

                      ---
                      Note: This report was generated by SceneSolver AI Investigation System (HuggingFace Space).
                      Model: Few-shot trained CLIP + ViT models
                      Generated on: ${new Date().toISOString()}
                      ---`,
                    },
                  ],
                },
              ],
            }),
          }
        );

        geminiResponse = await geminiRes.json();
        console.log("✅ Gemini report generated successfully");
      } catch (geminiErr) {
        console.error("⚠️ Gemini API error:", geminiErr);
      }
    }

    // Upload files to GridFS with analysis results
    const uploadPromises = files.map((file) =>
      streamToGridFS(file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
        metadata: {
          userId: uId,
          caseName: caseName,
          caseResult: caseResult,
          caseSummary: geminiResponse,
          type: file.mimetype.startsWith("video") ? "video" : "image",
          location: location,
          date: date,
          crimeTime: crimeTime,
          modelSource: "huggingface_space",
          analysisTimestamp: new Date().toISOString(),
          analysisError: analysisError,
        },
      })
    );

    const uploadedFiles = await Promise.all(uploadPromises);

    const response = {
      message: "Files uploaded and analyzed successfully",
      files: uploadedFiles,
      caseResult: caseResult,
      geminiOutput: geminiResponse,
      modelSource: "huggingface_space",
    };

    if (analysisError) {
      response.warning = "Analysis partially failed but files were uploaded";
      response.analysisError = analysisError;
    }

    res.status(200).json(response);
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({
      error: "File upload failed",
      details: error.message,
    });
  }
});

router.get("/user", async (req, res) => {
  try {
    if (!gridfsBucket) {
      return res.status(500).json({ error: "GridFS not initialized" });
    }

    const { email } = req.query;

    const user =
      (await User.findOne({ email })) || (await Guser.findOne({ email }));

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const uId = user._id;

    const query = {
      "metadata.userId": uId,
    };

    const cursor = conn.db.collection("uploads.files").find(query);

    const files = await cursor.toArray();

    res.json(files);
  } catch (error) {
    console.error("Error fetching files:", error);
    res.status(500).send("Server error");
  }
});

router.get("/user/case", async (req, res) => {
  try {
    if (!gridfsBucket) {
      return res.status(500).json({ error: "GridFS not initialized" });
    }

    const { email, caseName } = req.query;

    const user =
      (await User.findOne({ email })) || (await Guser.findOne({ email }));

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const uId = user._id;

    const query = {
      "metadata.userId": uId,
      "metadata.caseName": caseName,
    };

    const cursor = conn.db.collection("uploads.files").find(query);

    const files = await cursor.toArray();

    if (!files || files.length === 0) {
      return res.status(404).json({ error: "No files found for this case" });
    }

    res.json(files);
  } catch (error) {
    console.error("Error fetching files:", error);
    res.status(500).send("Server error");
  }
});

router.get("/file/:id", async (req, res) => {
  try {
    const fileId = new mongoose.Types.ObjectId(req.params.id);
    const file = await conn.db
      .collection("uploads.files")
      .findOne({ _id: fileId });

    if (!file) return res.status(404).json({ error: "File not found" });

    const downloadStream = gridfsBucket.openDownloadStream(fileId);
    res.set("Content-Type", file.contentType);
    downloadStream.pipe(res);
  } catch (error) {
    console.error("File fetch error:", error);
    res.status(500).json({ error: "Failed to fetch file" });
  }
});

router.post("/add-to-case", upload.array("files"), async (req, res) => {
  try {
    const { email, caseName } = req.body;
    const files = req.files;

    if (!email || !caseName || !files) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const user =
      (await User.findOne({ email })) || (await Guser.findOne({ email }));
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Validate files
    try {
      validateFiles(files);
    } catch (validationError) {
      return res.status(400).json({ error: validationError.message });
    }

    // Get existing case files to combine with new files for reanalysis
    const existingQuery = {
      "metadata.userId": user._id,
      "metadata.caseName": caseName,
    };

    const existingFiles = await conn.db
      .collection("uploads.files")
      .find(existingQuery)
      .toArray();

    // Combine existing and new files for reanalysis
    const allFilesForAnalysis = [...files]; // For now, just analyze new files
    // TODO: In future, might want to reanalyze all files together

    console.log(
      `🔍 Analyzing ${allFilesForAnalysis.length} new files for case: ${caseName}`
    );

    // Analyze new files using Hugging Face model
    let analysisResult = null;
    let analysisError = null;

    try {
      const analysisResponse = await analyzeCrimeScene(allFilesForAnalysis);
      analysisResult = analysisResponse.result;
      console.log("✅ HF Reanalysis completed:", analysisResult);
    } catch (error) {
      console.error("❌ HF Reanalysis failed:", error);
      analysisError = error.message || "Reanalysis failed";

      analysisResult = {
        error: "Reanalysis failed",
        details: analysisError,
        predicted_class: "Unknown",
        crime_confidence: 0,
        extracted_evidence: [],
        model_type: "huggingface_space_error",
      };
    }

    // Upload new files with analysis results
    const uploadPromises = files.map((file) =>
      streamToGridFS(file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
        metadata: {
          userId: user._id,
          email,
          caseName,
          uploadDate: new Date(),
          caseResult: analysisResult,
          modelSource: "huggingface_space",
          analysisTimestamp: new Date().toISOString(),
          analysisError: analysisError,
          isAdditionalFile: true,
        },
      })
    );

    await Promise.all(uploadPromises);

    const response = {
      message: "Files added and analyzed successfully",
      analysisResult: analysisResult,
      modelSource: "huggingface_space",
      filesAdded: files.length,
    };

    if (analysisError) {
      response.warning = "Analysis partially failed but files were uploaded";
      response.analysisError = analysisError;
    }

    res.json(response);
  } catch (error) {
    console.error("Error in add-to-case:", error);
    res.status(500).json({
      error: error.message || "Error adding files",
    });
  }
});

router.post("/reanalyze", async (req, res) => {
  try {
    const { email, caseName } = req.body;

    if (!email || !caseName) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const user =
      (await User.findOne({ email })) || (await Guser.findOne({ email }));
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Get all files for this case
    const query = {
      "metadata.userId": user._id,
      "metadata.caseName": caseName,
    };

    const caseFiles = await conn.db
      .collection("uploads.files")
      .find(query)
      .toArray();

    if (!caseFiles || caseFiles.length === 0) {
      return res.status(404).json({ error: "No files found for this case" });
    }

    // Download file buffers for reanalysis
    const filesForAnalysis = [];
    for (const file of caseFiles) {
      try {
        const downloadStream = gridfsBucket.openDownloadStream(file._id);
        const chunks = [];

        await new Promise((resolve, reject) => {
          downloadStream.on("data", (chunk) => chunks.push(chunk));
          downloadStream.on("end", () => resolve());
          downloadStream.on("error", reject);
        });

        const buffer = Buffer.concat(chunks);
        filesForAnalysis.push({
          buffer: buffer,
          originalname: file.filename,
          mimetype: file.contentType,
          size: buffer.length,
        });
      } catch (downloadError) {
        console.error(
          `Failed to download file ${file.filename}:`,
          downloadError
        );
      }
    }

    if (filesForAnalysis.length === 0) {
      return res
        .status(500)
        .json({ error: "Failed to download case files for reanalysis" });
    }

    console.log(
      `🔍 Reanalyzing ${filesForAnalysis.length} files for case: ${caseName}`
    );

    // Reanalyze using Hugging Face model
    let reanalysisResult = null;
    let analysisError = null;

    try {
      const analysisResponse = await analyzeCrimeScene(filesForAnalysis);
      reanalysisResult = analysisResponse.result;
      console.log("✅ HF Reanalysis completed:", reanalysisResult);
    } catch (error) {
      console.error("❌ HF Reanalysis failed:", error);
      analysisError = error.message || "Reanalysis failed";

      reanalysisResult = {
        error: "Reanalysis failed",
        details: analysisError,
        predicted_class: "Unknown",
        crime_confidence: 0,
        extracted_evidence: [],
        model_type: "huggingface_space_error",
      };
    }

    // Update all case files with new analysis results
    const client = await MongoClient.connect(process.env.MONGODB_URI);
    const db = client.db();
    const collection = db.collection("uploads.files");

    await collection.updateMany(
      { "metadata.email": email, "metadata.caseName": caseName },
      {
        $set: {
          "metadata.caseResult": reanalysisResult,
          "metadata.lastAnalyzed": new Date(),
          "metadata.modelSource": "huggingface_space",
          "metadata.reanalysisTimestamp": new Date().toISOString(),
          "metadata.analysisError": analysisError,
        },
      }
    );

    client.close();

    const response = {
      message: "Case reanalyzed successfully",
      result: reanalysisResult,
      modelSource: "huggingface_space",
      filesAnalyzed: filesForAnalysis.length,
    };

    if (analysisError) {
      response.warning = "Reanalysis partially failed";
      response.analysisError = analysisError;
    }

    res.json(response);
  } catch (error) {
    console.error("Error reanalyzing case:", error);
    res.status(500).json({ error: "Error reanalyzing case" });
  }
});

export default router;
