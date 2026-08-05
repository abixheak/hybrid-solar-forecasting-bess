import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Card } from "@/components/ui/card";
import { Battery, Zap, TrendingUp, Sliders } from "lucide-react";

const bessFeatures = [
  {
    icon: Battery,
    title: "SOC Tracking",
    description:
      "Real-time State-of-Charge monitoring with predictive degradation modeling and optimal charging curves",
  },
  {
    icon: Zap,
    title: "Charge/Discharge Simulation",
    description:
      "Realistic battery behavior simulation with efficiency curves, thermal management, and cycle optimization",
  },
  {
    icon: TrendingUp,
    title: "Grid Import/Export",
    description:
      "Intelligent grid interaction with peak shaving, load shifting, and revenue optimization algorithms",
  },
  {
    icon: Sliders,
    title: "Demand Profile Adjustment",
    description:
      "Customizable load curves with city-specific patterns, seasonal variations, and peak load modifiers",
  },
];

export default function BESSFeaturesSection() {
  const { ref, isVisible } = useScrollReveal();

  return (
    <section id="bess" className="section-padding bg-white">
      <div className="container" ref={ref}>
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2
            className={`text-4xl md:text-5xl font-bold text-slate-900 mb-4 transition-all duration-700 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Battery Energy Storage System
          </h2>
          <p
            className={`text-lg text-slate-600 transition-all duration-700 delay-150 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Advanced microgrid dispatch engine for optimal energy management
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {bessFeatures.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card
                key={feature.title}
                className={`p-8 hover-lift border-0 shadow-md transition-all duration-700 ${
                  isVisible
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-10"
                }`}
                style={{
                  transitionDelay: isVisible ? `${(index + 1) * 100}ms` : "0ms",
                }}
              >
                {/* Icon */}
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 mb-4">
                  <Icon className="w-7 h-7 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-bold text-slate-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            );
          })}
        </div>

        {/* Bottom Section - Technical Specs */}
        <div
          className={`mt-16 p-8 md:p-12 rounded-2xl bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-100 transition-all duration-700 delay-500 ${
            isVisible ? "opacity-100 scale-100" : "opacity-0 scale-95"
          }`}
        >
          <h3 className="text-2xl font-bold text-slate-900 mb-6">
            Dispatch Engine Capabilities
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              "Hourly SOC percentage tracking",
              "Dynamic charge/discharge power optimization",
              "Grid surplus curtailment management",
              "Demand profile with peak load modifiers",
              "Efficiency-aware energy routing",
              "Real-time microgrid balancing",
            ].map((capability, index) => (
              <div
                key={capability}
                className={`flex items-start gap-3 transition-all duration-700 ${
                  isVisible
                    ? "opacity-100 translate-x-0"
                    : "opacity-0 -translate-x-4"
                }`}
                style={{
                  transitionDelay: isVisible ? `${600 + index * 50}ms` : "0ms",
                }}
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center mt-1">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={3}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
                <span className="text-slate-700 font-medium">{capability}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
