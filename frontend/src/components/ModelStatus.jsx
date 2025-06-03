import React, { useState, useEffect } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function ModelStatus() {
  const [status, setStatus] = useState({ healthy: null, details: null });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkModelHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkModelHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkModelHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-health`);
      setStatus({
        healthy: response.data.status === "healthy",
        details: response.data,
      });
    } catch (error) {
      setStatus({
        healthy: false,
        details: { error: error.message },
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center space-x-2 text-gray-400">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
        <span className="text-sm">Checking model status...</span>
      </div>
    );
  }

  return (
    <div
      className={`flex items-center space-x-2 ${
        status.healthy ? "text-green-400" : "text-red-400"
      }`}
    >
      <div
        className={`w-2 h-2 rounded-full ${
          status.healthy ? "bg-green-400" : "bg-red-400"
        }`}
      ></div>
      <span className="text-sm">
        HF Model: {status.healthy ? "Ready" : "Not Available"}
      </span>
      {status.details?.space_url && (
        <a
          href={status.details.space_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-400 hover:text-blue-300 underline"
        >
          View Space
        </a>
      )}
    </div>
  );
}
