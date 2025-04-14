import React, { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";

const images = [
  "src/assets/crimeSceneImg.jpg",
  "src/assets/cs1.jpg",
  "src/assets/cs2.jpg",
  "src/assets/cs3.jpg",
];

export default function Upload() {
  const [files, setFiles] = useState([]);
  const [index, setIndex] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const onDrop = (acceptedFiles) => {
    setFiles(acceptedFiles);
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setPrediction("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error("Error:", err);
      setPrediction("Error contacting the API");
    }
    setLoading(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

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

        {files.length > 0 && (
          <div className="mt-6 space-y-2 text-white">
            {files.map((file, index) => (
              <p key={index} className="text-gray-200 text-lg">
                {file.name}
              </p>
            ))}
          </div>
        )}
      </motion.div>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Submit"}
        </button>
      </form>
      {prediction && (
        <div style={{ marginTop: 20 }}>
          <strong>Prediction:</strong> {prediction}
        </div>
      )}
    </section>
  );
}
