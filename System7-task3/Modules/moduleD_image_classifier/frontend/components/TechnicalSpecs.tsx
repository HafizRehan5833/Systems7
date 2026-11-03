import React from "react";
import { Cpu, Layers, GitBranch, ShieldCheck } from "lucide-react";

export default function TechnicalSpecs() {
  const specs = [
    {
      title: "Pretrained Base",
      value: "MobileNetV2 (ImageNet)",
      desc: "Trained on 1.4M images across 1,000 classes",
      icon: <Cpu className="w-4 h-4 text-cyan-400" />,
    },
    {
      title: "Input Pipeline",
      value: "32×32 → 96×96 RGB",
      desc: "Normalized to [-1, 1] with random flip augmentation",
      icon: <Layers className="w-4 h-4 text-indigo-400" />,
    },
    {
      title: "Training Paradigm",
      value: "Two-Stage Transfer Learning",
      desc: "Stage 1: Frozen head (lr=1e-3) · Stage 2: Top 30 layers (lr=1e-5)",
      icon: <GitBranch className="w-4 h-4 text-purple-400" />,
    },
    {
      title: "Regularization",
      value: "GlobalAvgPool + 20% Dropout",
      desc: "Prevents co-adaptation and mitigates overfitting",
      icon: <ShieldCheck className="w-4 h-4 text-emerald-400" />,
    },
  ];

  return (
    <div className="w-full mt-10 pt-10 border-t border-slate-800/80">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">
        Model Architecture & Training Specifications
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {specs.map((item) => (
          <div
            key={item.title}
            className="p-4 rounded-xl glass-panel border border-slate-800"
          >
            <div className="flex items-center gap-2 mb-2">
              {item.icon}
              <span className="text-xs text-slate-400">{item.title}</span>
            </div>
            <div className="text-sm font-bold text-white mb-1">{item.value}</div>
            <p className="text-[11px] text-slate-400 leading-normal">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

