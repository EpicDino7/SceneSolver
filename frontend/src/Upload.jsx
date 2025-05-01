import React, { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";

const images = [
  "src/assets/crimeSceneImg.jpg",
  "src/assets/cs1.jpg",
  "src/assets/cs2.jpg",
  "src/assets/cs3.jpg",
  "src/assets/cs4.jpeg",
  "src/assets/cs5.jpeg",
  "src/assets/cs6.jpeg",
  "src/assets/cs7.jpeg",
  "src/assets/cs8.jpeg",
];

export default function Upload() {
  const [step, setStep] = useState(1);
  const [caseTitle, setCaseTitle] = useState("");
  const [files, setFiles] = useState([]);
  const [index, setIndex] = useState(0);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const onDrop = (acceptedFiles) => {
    const validFiles = acceptedFiles.filter((file) =>
      ["image/png", "image/jpeg", "video/mp4", "video/quicktime"].includes(file.type)
    );

    if (validFiles.some((file) => file.type.startsWith("video"))) {
      if (validFiles.length > 1) {
        setError("You can only upload one video.");
        setFiles([]);
        setSuccessMessage("");
        return;
      }
    } else {
      if (validFiles.length < 4) {
        setError("You must upload at least 4 images.");
        setFiles([]);
        setSuccessMessage("");
        return;
      }
    }

    setError("");
    setFiles(validFiles);
    setSuccessMessage(`${validFiles.length} file(s) uploaded successfully!`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setError("No files uploaded. Please upload files before submitting.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("caseTitle", caseTitle);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPrediction(data.prediction || "No prediction returned.");
      setSuccessMessage("Prediction successful!");
      setError("");
    } catch (err) {
      console.error("Error:", err);
      setError("Error contacting the API.");
      setSuccessMessage("");
      setPrediction("");
    }
    setLoading(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center bg-[#1A1A1A] p-6 overflow-auto font-serif pt-16">
      {/* Background slideshow */}
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

      {/* Dark overlay */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm"></div>

      {/* Main card */}
      <motion.div
        className="relative z-10 bg-white/10 backdrop-blur-lg p-10 rounded-2xl shadow-2xl max-w-2xl w-full text-center border border-gray-300/30 overflow-auto"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        {step === 1 ? (
          <>
            <h2 className="text-4xl font-bold text-white">Enter Case Title</h2>
            <input
              type="text"
              value={caseTitle}
              onChange={(e) => setCaseTitle(e.target.value)}
              className="mt-6 p-3 rounded-2xl border border-gray-400 bg-transparent text-white placeholder-gray-400 w-full focus:outline-none focus:ring-2 focus:ring-[#D83A3A] transition-all"
              placeholder="Enter case title"
            />
            <button
              onClick={() => {
                if (caseTitle.trim() === "") {
                  setError("Case title cannot be empty.");
                  return;
                }
                setStep(2);
                setError("");
              }}
              className="mt-6 px-6 py-3 bg-[#D83A3A] text-white rounded-2xl hover:bg-[#B92B2B] transition-all shadow-md"
            >
              Next
            </button>
            {error && <p className="text-red-400 mt-4 text-sm">{error}</p>}
          </>
        ) : (
          <>
            <h2 className="text-4xl font-bold text-white">{caseTitle}</h2>
            <p className="text-gray-300 mt-3 text-lg">
              Upload <span className="text-[#D83A3A] font-semibold">one video</span> OR at least <span className="text-[#D83A3A] font-semibold">4 images</span>.
            </p>

            <div
              {...getRootProps()}
              className={`mt-6 p-12 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 ${
                isDragActive
                  ? "border-[#D83A3A] bg-red-50/20"
                  : "border-gray-400 bg-white/10"
              }`}
            >
              <input {...getInputProps()} />
              {isDragActive ? (
                <p className="text-[#D83A3A] font-semibold text-lg">Drop files here...</p>
              ) : (
                <p className="text-gray-300 text-lg">Drag and drop files or click to upload</p>
              )}
            </div>

            {error && <p className="text-red-400 mt-4 text-sm">{error}</p>}
            {successMessage && <p className="text-green-400 mt-4 text-sm">{successMessage}</p>}

            {files.length > 0 && (
              <div className="mt-6 max-h-40 overflow-y-auto space-y-2 text-left">
                {files.map((file, index) => (
                  <p
                    key={index}
                    className="text-gray-200 text-sm truncate bg-black/40 px-3 py-2 rounded-xl"
                  >
                    {file.name}
                  </p>
                ))}
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <button
                type="submit"
                disabled={loading}
                className="mt-6 px-6 py-3 bg-[#D83A3A] text-white rounded-2xl hover:bg-[#B92B2B] transition-all shadow-md disabled:opacity-50"
              >
                {loading ? "Predicting..." : "Start Analyzing!"}
              </button>
            </form>

            {prediction && (
              <div className="mt-6 text-white bg-black/40 p-4 rounded-2xl text-left shadow-inner">
                <strong className="text-[#D83A3A]">Prediction:</strong> {prediction}
              </div>
            )}
          </>
        )}
      </motion.div>
    </section>
  );
}
