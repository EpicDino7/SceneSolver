import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

const images = [
  "src/assets/crimeSceneImg.jpg",
  "src/assets/cs1.jpg",
  "src/assets/cs2.jpg",
  "src/assets/cs3.jpg",
];

export default function User() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % images.length);
    }, 5000); // Change image every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center bg-[#1A1A1A] p-6 overflow-auto font-serif">
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

      {/* User Card */}
      <motion.div
        className="relative z-10 bg-white/10 backdrop-blur-lg p-10 rounded-2xl shadow-2xl max-w-lg w-full text-center border border-gray-300/30"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <h2 className="text-4xl font-bold text-white">User Profile</h2>
        <p className="text-gray-300 mt-3 text-lg">Manage your account details here.</p>

        {/* User Details */}
        <div className="mt-6 space-y-4 text-left">
          <div className="flex justify-between items-center">
            <span className="text-gray-300 text-lg">Username:</span>
            <span className="text-white text-lg font-semibold">JohnDoe</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-300 text-lg">Email:</span>
            <span className="text-white text-lg font-semibold">johndoe@example.com</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-300 text-lg">Member Since:</span>
            <span className="text-white text-lg font-semibold">January 2025</span>
          </div>
        </div>

        {/* Buttons */}
        <div className="mt-8 flex justify-center space-x-4">
          <button className="px-6 py-3 bg-[#D83A3A] text-white rounded-lg hover:bg-[#B92B2B] transition-all">
            Edit Profile
          </button>
          <button className="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-all">
            Logout
          </button>
        </div>
      </motion.div>
    </section>
  );
}