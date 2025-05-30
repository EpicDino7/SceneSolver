import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom"; // Import useNavigate for navigation
import Header from "./components/Header";
import { useAuth } from "./context/AuthContext";

// Import assets properly for Vite
import crimeSceneImg from "./assets/crimeSceneImg.jpg";
import cs1 from "./assets/cs1.jpg";
import cs2 from "./assets/cs2.jpg";
import cs3 from "./assets/cs3.jpg";
import cs4 from "./assets/cs4.jpeg";
import cs5 from "./assets/cs5.jpeg";
import cs6 from "./assets/cs6.jpeg";
import cs7 from "./assets/cs7.jpeg";
import cs8 from "./assets/cs8.jpeg";

const images = [crimeSceneImg, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8];

export default function Dashboard() {
  const [index, setIndex] = useState(0);
  const navigate = useNavigate(); // Create navigate function

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 5000); // Change image every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const handleAnalyzeClick = () => {
    // Redirect to the upload page after login
    navigate("/upload"); // Replace '/upload' with the path of your upload page
  };

  return (
    <div className="w-full font-serif relative">
      {/* Hero Section */}
      <section className="relative w-full h-screen flex flex-col justify-center items-center text-center px-6 overflow-hidden">
        {/* Background Image Transition */}
        <div className="absolute inset-0 w-full h-full overflow-hidden z-0">
          <AnimatePresence mode="sync">
            {images.map(
              (img, i) =>
                i === index && (
                  <motion.img
                    key={i}
                    src={img}
                    className="absolute top-0 left-0 w-full h-full object-cover"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                )
            )}
          </AnimatePresence>
        </div>

        {/* Dark Overlay */}
        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm"></div>

        {/* Hero Content */}
        <motion.div
          className="relative z-10 max-w-3xl text-white"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <h1 className="text-6xl font-bold transition-transform hover:scale-105 duration-300">
            <span className="text-[#D83A3A]">SceneSolver</span>
            <br />
            AI-Powered Crime Scene Analysis
          </h1>
          <p className="mt-4 text-lg transition-opacity hover:opacity-100 duration-300">
            Upload crime scene images or videos and let AI analyze key evidence
            instantly.
          </p>

          {/* Buttons */}
          <div className="mt-6 flex space-x-10 justify-center">
            <motion.button
              whileHover={{ scale: 1.15 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 bg-[#D83A3A] text-white text-lg font-bold rounded-full shadow-lg transition-all duration-300 hover:bg-[#B92B2B]"
              onClick={handleAnalyzeClick} // Call the handleAnalyzeClick function
            >
              Analyze Crime Scene Now!
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.15 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 bg-white text-[#D83A3A] text-lg font-bold rounded-full shadow-lg transition-all duration-300 hover:bg-gray-200"
              onClick={() => navigate("/about")} // Navigate to the About page
            >
              Learn More
            </motion.button>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="text-center py-12">
        <h2 className="text-3xl font-semibold transition-colors duration-300">
          Key Features
        </h2>
        <p className="text-gray-600 mt-2">
          Enhance your forensic investigations with powerful AI-driven analysis
          tools
        </p>

        {/* Features Grid */}
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {[
            {
              title: "🔍 Crime Type Classification",
              desc: "Uses  OpenAI's CLIP to accurately classify crime scenes through image-text understanding.",
            },
            {
              title: "🕵 Evidence Extraction",
              desc: "Uses Vision Transformer (ViT) to extract and highlight key visual evidence from crime scene images.",
            },
            {
              title: "📝 Scene Summaries",
              desc: "Generate concise crime scene summaries and reports",
            },
          ].map((feature, index) => (
            <motion.div
              key={index}
              className="bg-white p-6 rounded-xl shadow-lg transition-transform hover:scale-105"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.2 }}
            >
              <h3 className="text-xl font-semibold text-[#D83A3A]">
                {feature.title}
              </h3>
              <p className="text-gray-600 mt-2">{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-gray-700 bg-black">
        <p className="text-white">
          © {new Date().getFullYear()} SceneSolver. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
