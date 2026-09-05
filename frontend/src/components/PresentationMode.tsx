import React, { useState, useEffect } from "react";
import type { DepthProcessResponse } from "@/types/depth";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Play, 
  Pause, 
  ChevronRight, 
  ChevronLeft, 
  Clock, 
  CheckCircle2, 
  Sparkles
} from "lucide-react";

interface PresentationModeProps {
  data: DepthProcessResponse | null;
  onSelectHeroScene: (sceneId: string) => void;
  onNavigateTab: (tab: string) => void;
}

const STEPS = [
  {
    step: 1,
    title: "1. Input Scene & View Geometry Analysis",
    subtitle: "Nadir vs Oblique Perspective Classification",
    description: "Evaluates image resolution, sharpness, and classifies view geometry (Near-Nadir 75-90° vs Oblique 25-45° vs Low-Altitude Drone). Flags nadir warnings to prevent uncalibrated height claims.",
    tabTarget: "inspector",
    tag: "Computer Vision Preprocessing"
  },
  {
    step: 2,
    title: "2. Depth Anything V2 Monocular Backbone",
    subtitle: "Structural Edge & Relative Elevation Map",
    description: "Infers continuous floating-point depth representation with multi-scale gradient preservation and computes Depth Quality & Confidence indicators.",
    tabTarget: "inspector",
    tag: "Neural Depth Inference"
  },
  {
    step: 3,
    title: "3. 3D Geometric Surface Reconstruction",
    subtitle: "Back-Projection & Interactive Three.js WebGL",
    description: "Back-projects relative depth into 3D Point Cloud and Heightfield Mesh with vertex colors, statistical outlier removal, and smooth normals.",
    tabTarget: "inspector",
    tag: "3D Geometry Back-Projection"
  },
  {
    step: 4,
    title: "4. Scientific Calibration & Structure Ruler",
    subtitle: "4 Calibration Modes + Peak vs Baseline",
    description: "Distinguishes Relative Depth from Metric Height. Decomposes structure footprints into rooftop peak Z vs surrounding ground baseline Z0 with uncertainty bounds ±σ.",
    tabTarget: "calibration",
    tag: "Metric Height Calibration"
  },
  {
    step: 5,
    title: "5. Validation Framework & Honest Reporting",
    subtitle: "Zero Fabricated Accuracy Claims",
    description: "Shows how error is quantified against an independent reference dataset. No accuracy figure is claimed until verified ground truth is registered - the dashboard states exactly what data is required.",
    tabTarget: "validation",
    tag: "Validation Methodology"
  }
];

export const PresentationMode: React.FC<PresentationModeProps> = ({
  onNavigateTab,
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [secondsRemaining, setSecondsRemaining] = useState<number>(180); // 3 minutes
  const [isTimerRunning, setIsTimerRunning] = useState<boolean>(true);

  // Timer countdown
  useEffect(() => {
    let interval: any = null;
    if (isTimerRunning && secondsRemaining > 0) {
      interval = setInterval(() => {
        setSecondsRemaining((prev) => prev - 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTimerRunning, secondsRemaining]);

  const currentStep = STEPS[currentStepIndex];

  const handleNext = () => {
    if (currentStepIndex < STEPS.length - 1) {
      const nextIdx = currentStepIndex + 1;
      setCurrentStepIndex(nextIdx);
      onNavigateTab(STEPS[nextIdx].tabTarget);
    }
  };

  const handlePrev = () => {
    if (currentStepIndex > 0) {
      const prevIdx = currentStepIndex - 1;
      setCurrentStepIndex(prevIdx);
      onNavigateTab(STEPS[prevIdx].tabTarget);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s < 10 ? "0" : ""}${s}`;
  };

  return (
    <div className="space-y-4">
      {/* 3-Minute Walkthrough Banner */}
      <Card className="bg-gradient-to-r from-slate-900 via-[#0F172A] to-cyan-950/60 border-cyan-500/40 shadow-2xl overflow-hidden">
        <div className="p-4 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-cyan-500/20 border border-cyan-400 text-cyan-300">
              <Sparkles className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-100 font-mono-data uppercase tracking-wider">
                  SIH26175 Demo Mode — 3-Minute Judge-Ready Walkthrough
                </h3>
                <Badge className="bg-cyan-500/20 text-cyan-300 border-cyan-500/40 text-[10px] font-mono-data">
                  Live Judging
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                One-sentence pitch: <strong className="text-slate-200">"DepthWizard converts a single aerial or satellite image into a depth-assisted 3D representation and, when metric reference information is available, estimates object heights without requiring stereo pairs or LiDAR."</strong>
              </p>
            </div>
          </div>

          {/* Presentation Timer */}
          <div className="flex items-center gap-3 bg-black/50 px-3.5 py-1.5 rounded-xl border border-slate-700 font-mono-data text-xs">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-400">Time:</span>
            <span className={`text-base font-extrabold ${secondsRemaining < 30 ? "text-red-400 animate-ping" : "text-cyan-300"}`}>
              {formatTime(secondsRemaining)}
            </span>
            <button
              onClick={() => setIsTimerRunning(!isTimerRunning)}
              className="text-slate-400 hover:text-white ml-1"
            >
              {isTimerRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 text-emerald-400" />}
            </button>
          </div>
        </div>

        {/* Step Progression Ribbon */}
        <CardContent className="p-4 space-y-4">
          <div className="grid grid-cols-5 gap-2">
            {STEPS.map((s, idx) => {
              const isActive = idx === currentStepIndex;
              const isPast = idx < currentStepIndex;
              return (
                <button
                  key={s.step}
                  onClick={() => {
                    setCurrentStepIndex(idx);
                    onNavigateTab(s.tabTarget);
                  }}
                  data-testid={`pitch-step-${s.step}-button`}
                  className={`p-2.5 rounded-lg border text-left transition-all font-mono-data text-xs flex flex-col justify-between ${
                    isActive
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-lg shadow-cyan-500/10"
                      : isPast
                      ? "bg-slate-900/80 border-emerald-500/50 text-emerald-300"
                      : "bg-black/40 border-slate-800 text-slate-500 hover:text-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-bold">STEP {s.step}</span>
                    {isPast && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
                  </div>
                  <span className="text-[11px] font-semibold line-clamp-1">{s.title.split(". ")[1]}</span>
                </button>
              );
            })}
          </div>

          {/* Current Step Active Card */}
          <div className="p-4 bg-black/60 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1 max-w-2xl">
              <div className="flex items-center gap-2">
                <Badge className="bg-cyan-500/10 text-cyan-300 border-cyan-500/30 text-[10px] font-mono-data">
                  {currentStep.tag}
                </Badge>
                <h4 className="text-sm font-bold text-slate-100 font-mono-data">
                  {currentStep.title}: {currentStep.subtitle}
                </h4>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {currentStep.description}
              </p>
            </div>

            {/* Navigation Buttons */}
            <div className="flex items-center gap-2 font-mono-data">
              <Button
                size="sm"
                variant="outline"
                disabled={currentStepIndex === 0}
                onClick={handlePrev}
                data-testid="pitch-prev-button"
                className="h-8 text-xs border-slate-700 text-slate-300"
              >
                <ChevronLeft className="w-3.5 h-3.5 mr-1" />
                Previous Step
              </Button>
              <Button
                size="sm"
                onClick={handleNext}
                disabled={currentStepIndex === STEPS.length - 1}
                data-testid="pitch-next-button"
                className="h-8 text-xs bg-cyan-500 hover:bg-cyan-400 text-black font-semibold"
              >
                Next Step
                <ChevronRight className="w-3.5 h-3.5 ml-1" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
