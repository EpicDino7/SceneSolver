import React, { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";

const images = [
  "src/assets/crimeSceneImg.jpg",
  "src/assets/cs1.jpg",
  "src/assets/cs2.jpg",
  "src/assets/cs3.jpg",
];

export default function Upload() {
  const [files, setFiles] = useState([]);
  const [index, setIndex] = useState(0);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const validateFiles = (acceptedFiles) => {
    const allowedImageExtensions = [".jpg", ".jpeg", ".png"];
    const allowedVideoExtensions = [".mp4", ".mov"];

    let images = [];
    let videos = [];

    for (const file of acceptedFiles) {
      const name = file.name.toLowerCase();

      if (allowedImageExtensions.some(ext => name.endsWith(ext))) {
        images.push(file);
      } else if (allowedVideoExtensions.some(ext => name.endsWith(ext))) {
        videos.push(file);
      } else {
        setErrorMessage("Invalid file type detected. Only .jpg, .jpeg, .png, .mp4, and .mov are allowed.");
        return false;
      }
    }
    if (images.length > 0 && videos.length > 0) {
      setErrorMessage("Cannot upload images and videos together. Upload either images or videos.");
      return false;
    }

    if (images.length > 0 && images.length < 4) {
      setErrorMessage("Please upload at least 4 images.");
      return false;
    }
    if (videos.length > 1) {
      setErrorMessage("Only one video is allowed.");
      return false;
    }

    setErrorMessage(""); // No errors
    return true;
  };

  const onDrop = (acceptedFiles) => {
    if (validateFiles(acceptedFiles)) {
      setFiles(acceptedFiles);
    } else {
      setFiles([]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setErrorMessage("Please upload files before submitting.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("file", file);
      });

      const response = await fetch("http://localhost:5000/api/files", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction("✅ Files uploaded successfully!");
        setFiles([]);
      } else {
        const errorData = await response.json();
        setPrediction(`Error uploading files: ${errorData.message || 'Something went wrong'}`);
      }
    } catch (err) {
      console.error("Error sending files:", err);
      setPrediction("Error connecting to the server.");
    }
    setLoading(false);
  };

  const clearFiles = () => {
    setFiles([]);
    setErrorMessage(""); // Clear any error messages
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [],
      "image/png": [],
      "video/mp4": [],
      "video/quicktime": [] // .mov files
    }
  });

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6 overflow-hidden">
      <div className="absolute inset-0 w-full h-full overflow-hidden z-0">
        {images.map((img, i) => (
          <motion.img
            key={i}
            src={img}
            className="absolute w-full h-full object-cover blur-lg"
            style={{ opacity: i === index ? 1 : 0 }}
            animate={{ opacity: i === index ? 1 : 0 }}
            transition={{ duration: 2, ease: "easeInOut" }}
          />
        ))}
      </div>

      <div className="absolute inset-0 bg-black/60 backdrop-blur-md"></div>

      <motion.div
        className="relative z-10 bg-white/10 backdrop-blur-lg p-10 rounded-xl shadow-xl max-w-2xl w-full text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <h2 className="text-4xl font-bold text-white">
          Upload Crime Scene Evidence
        </h2>
        <p className="text-gray-300 mt-3 text-lg">
          Drag & drop files here or click to browse.
        </p>

        <div
          {...getRootProps()}
          className={`mt-6 p-12 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
            isDragActive
              ? "border-[#D83A3A] bg-red-50/20"
              : "border-gray-300 bg-white/10"
          }`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p className="text-[#D83A3A] font-semibold text-lg">
              Drop files here...
            </p>
          ) : (
            <p className="text-gray-300 text-lg">
              Drag and drop files or click to upload
            </p>
          )}
        </div>

        {errorMessage && (
          <div className="text-red-500 mt-4 font-semibold">{errorMessage}</div>
        )}

        {files.length > 0 && (
          <div className="mt-6 space-y-2 text-white">
            {files.map((file, index) => (
              <p key={index} className="text-gray-200 text-lg">
                {file.name}
              </p>
            ))}
            <div className="flex justify-center gap-4 mt-4">
              <button
                onClick={handleSubmit}
                disabled={loading || files.length === 0}
                className={`px-6 py-3 rounded-md text-white font-semibold transition-colors duration-300 ${
                  loading || files.length === 0
                    ? "bg-gray-500 cursor-not-allowed"
                    : "bg-[#D83A3A] hover:bg-[#B82E2E]"
                }`}
              >
                {loading ? "Uploading..." : "Submit"}
              </button>
              <button
                onClick={clearFiles}
                disabled={files.length === 0}
                className={`px-6 py-3 rounded-md text-white font-semibold transition-colors duration-300 ${
                  files.length === 0 ? "bg-gray-500 cursor-not-allowed" : "bg-gray-700 hover:bg-gray-600"
                }`}
              >
                Clear
              </button>
            </div>
          </div>
        )}
      </motion.div>

      {prediction && (
        <div style={{ marginTop: 20 }} className="text-white text-lg font-semibold">
          {prediction}
        </div>
      )}
    </section>
  );
}
