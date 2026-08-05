import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight, Play } from "lucide-react";

const HERO_BG = "/manus-storage/hero-solar-farm_a473a838.jpg";

export default function HeroSection() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <section className="relative w-full h-screen min-h-[600px] overflow-hidden pt-20">
      {/* Background Image with Overlay */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: `url('${HERO_BG}')`,
          backgroundAttachment: "fixed",
        }}
      >
        {/* Dark Overlay Gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-slate-900/70 via-slate-900/50 to-slate-900/40"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 h-full flex flex-col items-center justify-center px-4">
        <div className="max-w-4xl mx-auto text-center space-y-6 md:space-y-8">
          {/* Main Headline */}
          <h1
            className={`text-4xl md:text-6xl lg:text-7xl font-bold text-white leading-tight transition-all duration-1000 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Enterprise Solar Forecasting
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400 mt-2">
              Powered by AI
            </span>
          </h1>

          {/* Subheadline */}
          <p
            className={`text-lg md:text-xl text-slate-200 max-w-2xl mx-auto transition-all duration-1000 delay-200 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Hybrid SARIMAX-LSTM architecture combined with intelligent BESS dispatch optimization for maximum energy efficiency and grid stability
          </p>

          {/* CTA Buttons */}
          <div
            className={`flex flex-col sm:flex-row gap-4 justify-center pt-4 transition-all duration-1000 delay-300 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            <Button
              className="btn-premium inline-flex items-center gap-2 text-base px-8 py-6 h-auto"
              disabled
            >
              Explore Dashboard
              <ArrowRight className="w-5 h-5" />
            </Button>
            <Button
              variant="outline"
              className="inline-flex items-center gap-2 text-base px-8 py-6 h-auto border-2 border-white text-white hover:bg-white/10"
            >
              <Play className="w-5 h-5" />
              Watch Demo
            </Button>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div
          className={`absolute bottom-8 left-1/2 transform -translate-x-1/2 transition-all duration-1000 delay-500 ${
            isVisible ? "opacity-100" : "opacity-0"
          }`}
        >
          <div className="flex flex-col items-center gap-2">
            <p className="text-sm text-slate-300 font-medium">Scroll to explore</p>
            <div className="w-6 h-10 border-2 border-slate-300 rounded-full flex items-start justify-center p-2">
              <div className="w-1 h-2 bg-slate-300 rounded-full animate-bounce"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
