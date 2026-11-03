"use client";

import React from "react";
import { Sparkles, Brain, Cpu, Github, ExternalLink } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-white/10 px-6 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/25 flex items-center justify-center">
            <div className="w-full h-full bg-slate-950/80 rounded-[10px] flex items-center justify-center">
              <Brain className="w-5 h-5 text-indigo-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-white">
                Vision<span className="text-indigo-400">AI</span>
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded-full">
                v2.0 PRO
              </span>
            </div>
            <p className="text-xs text-slate-400">MobileNetV2 · CIFAR-10 Engine</p>
          </div>
        </div>

        {/* Stats Pill & Badges */}
        <div className="hidden md:flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Pretrained: <strong className="text-white">ImageNet</strong></span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Classes: <strong className="text-white">10 Categories</strong></span>
          </div>
        </div>

        {/* Action Link */}
        <div className="flex items-center gap-3">
          <span className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600/10 text-indigo-300 border border-indigo-500/20 font-medium">
            System 7 · Task 3
          </span>
        </div>
      </div>
    </header>
  );
}

