import React from "react";
import { Grid } from "lucide-react";

const CATEGORIES = [
  { name: "Airplane", icon: "✈️", type: "Vehicle", desc: "Commercial jets, biplanes, fighters" },
  { name: "Automobile", icon: "🚗", type: "Vehicle", desc: "Sedans, sports cars, hatchbacks" },
  { name: "Bird", icon: "🐦", type: "Animal", desc: "Flying birds, songbirds, waterfowl" },
  { name: "Cat", icon: "🐱", type: "Animal", desc: "Domestic felines, kittens, tabby" },
  { name: "Deer", icon: "🦌", type: "Animal", desc: "Stags, fawns, forest antelopes" },
  { name: "Dog", icon: "🐶", type: "Animal", desc: "Various breeds, puppies, canines" },
  { name: "Frog", icon: "🐸", type: "Animal", desc: "Tree frogs, toads, amphibians" },
  { name: "Horse", icon: "🐴", type: "Animal", desc: "Equines, stallions, ponies" },
  { name: "Ship", icon: "🚢", type: "Vehicle", desc: "Cargo vessels, yachts, boats" },
  { name: "Truck", icon: "🚚", type: "Vehicle", desc: "Heavy transports, pickups, lorries" },
];

export default function ModelCategories() {
  return (
    <div className="w-full mt-12 pt-12 border-t border-slate-800/80">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <Grid className="w-5 h-5 text-indigo-400" />
            <span>Supported Classes (CIFAR-10)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            The neural network has been trained to distinguish between these 10 distinct visual targets
          </p>
        </div>
        <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
          10 Total Classes
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {CATEGORIES.map((cat) => (
          <div
            key={cat.name}
            className="p-3.5 rounded-xl glass-panel border border-slate-800 hover:border-indigo-500/30 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-2xl">{cat.icon}</span>
              <span
                className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                  cat.type === "Animal"
                    ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/20"
                    : "bg-cyan-500/10 text-cyan-300 border border-cyan-500/20"
                }`}
              >
                {cat.type}
              </span>
            </div>
            <div className="font-semibold text-sm text-white mb-0.5">{cat.name}</div>
            <div className="text-[11px] text-slate-400 line-clamp-1">{cat.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

