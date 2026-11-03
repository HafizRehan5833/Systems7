"use client";

import React, { useState } from "react";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import ImageUploader from "@/components/ImageUploader";
import SampleImages from "@/components/SampleImages";
import PredictionCard from "@/components/PredictionCard";
import ConfidenceBreakdown from "@/components/ConfidenceBreakdown";
import ModelCategories from "@/components/ModelCategories";
import TechnicalSpecs from "@/components/TechnicalSpecs";
import { PredictionResponse } from "@/lib/types";
import { Sparkles, Brain, AlertCircle } from "lucide-react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    setErrorMsg(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data: PredictionResponse = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || "Failed to classify image.");
      }

      setPredictionResult(data);
    } catch (err: any) {
      console.error("Inference Error:", err);
      setErrorMsg(err.message || "An error occurred during prediction.");
      setPredictionResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionResult(null);
    setErrorMsg(null);
  };

  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
          <HeroSection />

          {/* Interactive Workspace: Upload on left, results on right */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mt-4">
            {/* Left Column: Upload and Samples (7 cols) */}
            <div className="lg:col-span-7 space-y-6">
              <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    <span>Input Photograph</span>
                  </h2>
                  <span className="text-xs text-slate-400">Step 1: Upload or Choose Sample</span>
                </div>

                <ImageUploader
                  onImageSelected={handleImageSelected}
                  previewUrl={previewUrl}
                  onClear={handleClear}
                  isLoading={isLoading}
                />
              </div>

              {/* Sample Images Palette */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800">
                <SampleImages
                  onSelectSample={handleImageSelected}
                  disabled={isLoading}
                />
              </div>
            </div>

            {/* Right Column: Prediction Results (5 cols) */}
            <div className="lg:col-span-5 space-y-6">
              {errorMsg && (
                <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-200 text-sm flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <strong className="font-semibold">Classification Error:</strong>
                    <p className="text-xs mt-1 text-rose-300">{errorMsg}</p>
                  </div>
                </div>
              )}

              {predictionResult ? (
                <div className="space-y-6 animate-in fade-in duration-500">
                  <PredictionCard result={predictionResult} />
                  {predictionResult.predictions && (
                    <ConfidenceBreakdown predictions={predictionResult.predictions} />
                  )}
                </div>
              ) : (
                /* Empty state when no prediction has been run */
                <div className="rounded-2xl glass-panel p-8 md:p-12 border border-slate-800 text-center flex flex-col items-center justify-center min-h-[380px]">
                  <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-4">
                    <Brain className="w-8 h-8 animate-pulse" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">
                    Awaiting Image Input
                  </h3>
                  <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                    Upload an image or click any demo sample on the left to trigger the convolutional neural network inference pipeline.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Model Categories Breakdown */}
          <ModelCategories />

          {/* Technical Specifications */}
          <TechnicalSpecs />
        </main>
      </div>

      {/* Footer */}
      <footer className="glass-panel border-t border-slate-800/80 py-6 px-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 VisionAI · System 7 Applied Machine Learning & Deep Learning</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>TensorFlow 2.x</span>
            <span>•</span>
            <span>MobileNetV2</span>
            <span>•</span>
            <span>Next.js 14</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

