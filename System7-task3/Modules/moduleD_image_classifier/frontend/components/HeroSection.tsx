import React from "react";
import { Sparkles, ArrowRight, Layers, Zap } from "lucide-react";

export default function HeroSection() {
  return (
    <div className="relative py-12 md:py-16 text-center max-w-4xl mx-auto px-4">
      {/* Background ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[300px] bg-gradient-to-tr from-indigo-600/20 via-purple-600/15 to-cyan-500/10 blur-3xl pointer-events-none -z-10 rounded-full" />

      {/* Pill header */}
      <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium mb-6">
        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
        <span>Deep Convolutional Neural Network & Transfer Learning</span>
      </div>

      {/* Main heading */}
      <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight text-white mb-6 leading-tight">
        Next-Generation <br className="hidden sm:inline" />
        <span className="gradient-text">Image Classification</span> with AI
      </h1>

      {/* Subtitle */}
      <p className="text-base md:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed mb-8">
        Drag and drop any photograph to run real-time inference through{" "}
        <strong className="text-indigo-300 font-semibold">MobileNetV2</strong>, fine-tuned on the
        CIFAR-10 vision benchmark with two-stage transfer learning.
      </p>

      {/* Key Feature Highlights */}
      <div className="flex flex-wrap items-center justify-center gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-1.5">
          <Zap className="w-4 h-4 text-amber-400" />
          <span>Sub-100ms Inference</span>
        </div>
        <span className="text-slate-700">•</span>
        <div className="flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span>Two-Stage Feature Extraction</span>
        </div>
        <span className="text-slate-700">•</span>
        <div className="flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span>Top-3 Probability Distribution</span>
        </div>
      </div>
    </div>
  );
}

