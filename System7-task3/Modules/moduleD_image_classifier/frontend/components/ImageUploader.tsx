"use client";

import React, { useRef, useState } from "react";
import { UploadCloud, Image as ImageIcon, X, RefreshCw, CheckCircle2 } from "lucide-react";

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  previewUrl: string | null;
  onClear: () => void;
  isLoading: boolean;
}

export default function ImageUploader({
  onImageSelected,
  previewUrl,
  onClear,
  isLoading,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        onImageSelected(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelected(e.target.files[0]);
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        className="hidden"
        onChange={handleFileChange}
      />

      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 md:p-12 text-center transition-all duration-300 ${
            isDragging
              ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
              : "border-slate-700/80 hover:border-indigo-500/50 bg-slate-900/40 hover:bg-slate-900/60"
          }`}
        >
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
            <UploadCloud className="w-8 h-8" />
          </div>

          <h3 className="text-lg font-semibold text-white mb-1">
            Drop your image here, or{" "}
            <span className="text-indigo-400 underline underline-offset-4">browse</span>
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Supports PNG, JPG, JPEG, and WebP (up to 10MB)
          </p>

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs text-slate-300">
            <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Resized to 32×32 for CIFAR-10 tensor processing</span>
          </div>
        </div>
      ) : (
        <div className="relative rounded-2xl glass-panel p-4 overflow-hidden border border-slate-800">
          <div className="relative w-full aspect-video max-h-[340px] rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center border border-white/5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Uploaded Preview"
              className="w-full h-full object-contain"
            />

            {isLoading && (
              <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin" />
                <p className="text-sm font-medium text-slate-200">
                  Running Neural Inference...
                </p>
                <p className="text-xs text-slate-400">
                  MobileNetV2 feature extraction & forward pass
                </p>
              </div>
            )}
          </div>

          {/* Action bar below preview */}
          <div className="mt-3 flex items-center justify-between px-1">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Image loaded successfully</span>
            </div>

            <button
              onClick={onClear}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 text-xs font-medium border border-rose-500/20 transition-colors disabled:opacity-50"
            >
              <X className="w-3.5 h-3.5" />
              <span>Remove Photo</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

