"use client";

import React from "react";
import { Sparkles } from "lucide-react";

interface SampleImagesProps {
  onSelectSample: (file: File) => void;
  disabled: boolean;
}

// Inline SVGs converted to Data URLs so sample testing works offline without external assets
const SAMPLES = [
  {
    name: "Cat",
    icon: "🐱",
    color: "#f59e0b",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="#1e1b4b"/><circle cx="48" cy="52" r="28" fill="#fbbf24"/><polygon points="26,30 38,44 24,48" fill="#f59e0b"/><polygon points="70,30 58,44 72,48" fill="#f59e0b"/><circle cx="38" cy="50" r="4" fill="#0f172a"/><circle cx="58" cy="50" r="4" fill="#0f172a"/><polygon points="48,58 44,64 52,64" fill="#ef4444"/></svg>`,
  },
  {
    name: "Automobile",
    icon: "🚗",
    color: "#ef4444",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="#0f172a"/><rect x="18" y="44" width="60" height="22" rx="4" fill="#ef4444"/><polygon points="26,44 36,28 60,28 70,44" fill="#dc2626"/><circle cx="32" cy="68" r="8" fill="#334155"/><circle cx="64" cy="68" r="8" fill="#334155"/></svg>`,
  },
  {
    name: "Airplane",
    icon: "✈️",
    color: "#38bdf8",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="#0369a1"/><polygon points="48,16 54,48 84,62 82,68 52,58 50,76 58,82 56,86 48,82 40,86 38,82 46,76 44,58 14,68 12,62 42,48" fill="#f8fafc"/></svg>`,
  },
  {
    name: "Dog",
    icon: "🐶",
    color: "#a855f7",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="#1e293b"/><circle cx="48" cy="50" r="26" fill="#d97706"/><ellipse cx="28" cy="46" rx="8" ry="16" fill="#92400e"/><ellipse cx="68" cy="46" rx="8" ry="16" fill="#92400e"/><circle cx="40" cy="48" r="3.5" fill="#0f172a"/><circle cx="56" cy="48" r="3.5" fill="#0f172a"/><ellipse cx="48" cy="58" rx="5" ry="3.5" fill="#0f172a"/></svg>`,
  },
  {
    name: "Ship",
    icon: "🚢",
    color: "#06b6d4",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="#0f172a"/><rect x="36" y="26" width="24" height="24" fill="#64748b"/><rect x="42" y="16" width="8" height="10" fill="#ef4444"/><polygon points="12,50 84,50 74,74 22,74" fill="#0284c7"/></svg>`,
  },
];

export default function SampleImages({
  onSelectSample,
  disabled,
}: SampleImagesProps) {
  const handleSampleClick = async (sample: (typeof SAMPLES)[0]) => {
    if (disabled) return;
    const blob = new Blob([sample.svg], { type: "image/svg+xml" });
    const file = new File([blob], `${sample.name.toLowerCase()}_sample.svg`, {
      type: "image/svg+xml",
    });
    onSelectSample(file);
  };

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
        <span>Or Test With One-Click Demo Samples</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {SAMPLES.map((sample) => (
          <button
            key={sample.name}
            onClick={() => handleSampleClick(sample)}
            disabled={disabled}
            className="flex flex-col items-center justify-center p-3 rounded-xl glass-panel glass-panel-hover border border-slate-800 disabled:opacity-50 text-center group cursor-pointer"
          >
            <span className="text-2xl mb-1.5 group-hover:scale-125 transition-transform">
              {sample.icon}
            </span>
            <span className="text-xs font-medium text-slate-200 group-hover:text-indigo-300">
              {sample.name}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

