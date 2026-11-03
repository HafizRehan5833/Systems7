"use client";

import React from "react";
import { BarChart3, Layers } from "lucide-react";
import { PredictionItem } from "@/lib/types";

interface ConfidenceBreakdownProps {
  predictions: PredictionItem[];
}

export default function ConfidenceBreakdown({
  predictions,
}: ConfidenceBreakdownProps) {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div className="w-full rounded-2xl glass-panel p-6 border border-slate-800">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Probability Distribution
          </h3>
        </div>
        <span className="text-[11px] text-slate-400">Softmax Activation (10 Classes)</span>
      </div>

      <div className="space-y-3.5">
        {predictions.map((item, index) => {
          const pct = Math.round(item.confidence * 1000) / 10;
          const isTop = index === 0;

          return (
            <div key={item.label} className="group">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-base">{item.icon}</span>
                  <span
                    className={`capitalize font-medium ${
                      isTop ? "text-indigo-300 font-semibold" : "text-slate-300"
                    }`}
                  >
                    {item.label}
                  </span>
                  {isTop && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      Rank #1
                    </span>
                  )}
                </div>
                <span className="font-mono text-slate-300 font-medium">
                  {pct}%
                </span>
              </div>

              {/* Progress track */}
              <div className="w-full h-2 rounded-full bg-slate-900 border border-slate-800/80 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isTop
                      ? "bg-gradient-to-r from-indigo-500 to-cyan-400"
                      : "bg-slate-700/60 group-hover:bg-slate-600"
                  }`}
                  style={{ width: `${Math.max(pct, 1.5)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

