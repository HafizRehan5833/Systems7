"use client";

import React from "react";
import { Sparkles, Gauge, AlertCircle, CheckCircle, Clock } from "lucide-react";
import { PredictionResponse } from "@/lib/types";

interface PredictionCardProps {
  result: PredictionResponse;
}

export default function PredictionCard({ result }: PredictionCardProps) {
  const top = result.top_prediction;
  if (!top) return null;

  const percentage = Math.round(top.confidence * 1000) / 10; // e.g. 94.5%

  // Determine confidence tier
  let tierColor = "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  let tierLabel = "High Confidence";
  if (top.confidence < 0.6) {
    tierColor = "text-amber-400 bg-amber-500/10 border-amber-500/30";
    tierLabel = "Moderate Confidence";
  }
  if (top.confidence < 0.35) {
    tierColor = "text-rose-400 bg-rose-500/10 border-rose-500/30";
    tierLabel = "Low Confidence (Ambiguous)";
  }

  return (
    <div className="w-full rounded-2xl glass-panel p-6 border border-indigo-500/30 relative overflow-hidden">
      {/* Background radial gradient badge */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 blur-3xl pointer-events-none -z-10 rounded-full" />

      {/* Mock notice banner if model is not yet trained */}
      {result.is_mock && (
        <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong className="font-semibold">Demo Mode Active:</strong>{" "}
            {result.message}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-400">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <span>Top Neural Prediction</span>
        </div>

        <div className="flex items-center gap-2">
          {result.latency_ms && (
            <div className="flex items-center gap-1 text-[11px] text-slate-400 px-2 py-1 rounded-md bg-slate-800/80 border border-slate-700">
              <Clock className="w-3 h-3 text-cyan-400" />
              <span>{result.latency_ms} ms</span>
            </div>
          )}
          <span className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${tierColor}`}>
            {tierLabel}
          </span>
        </div>
      </div>

      {/* Main winning class showcase */}
      <div className="flex items-center gap-5 my-4">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-indigo-900/60 to-slate-800/80 border border-indigo-500/30 flex items-center justify-center text-4xl shadow-inner shadow-indigo-500/20">
          {top.icon}
        </div>

        <div className="flex-1">
          <div className="text-3xl md:text-4xl font-extrabold text-white capitalize tracking-tight mb-1">
            {top.label}
          </div>
          <p className="text-xs text-slate-400">
            Probability: <strong className="text-white text-sm font-semibold">{percentage}%</strong>
          </p>
        </div>
      </div>

      {/* Large confidence bar */}
      <div className="mt-5">
        <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden p-0.5 border border-slate-700/50">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-indigo-400 to-cyan-400 transition-all duration-700 ease-out"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}

