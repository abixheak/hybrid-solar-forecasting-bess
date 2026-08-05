import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Card } from "@/components/ui/card";
import { Zap, Cloud, Gauge, BarChart3 } from "lucide-react";

const features = [
  {
    icon: Zap,
    title: "SARIMAX-LSTM Architecture",
    description:
      "Hybrid ML pipeline combining SARIMAX baseline with LSTM residual correction for superior forecast accuracy",
  },
  {
    icon: Cloud,
    title: "Live Weather Synchronization",
    description:
      "Real-time Open-Meteo integration for 8 major Indian solar hubs with hourly telemetry updates",
  },
  {
    icon: Gauge,
    title: "BESS Dispatch Engine",
    description:
      "Intelligent battery management with SOC tracking, charge/discharge simulation, and grid import/export optimization",
  },
  {
    icon: BarChart3,
    title: "Real-Time KPI Metrics",
    description:
      "Live dashboard with generation forecasts, demand profiles, battery status, and energy balance diagnostics",
  },
];

export default function FeaturesSection() {
  const { ref, isVisible } = useScrollReveal();

  return (
    <section id="features" className="section-padding bg-slate-50">
      <div className="container" ref={ref}>
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2
            className={`text-4xl md:text-5xl font-bold text-slate-900 mb-4 transition-all duration-700 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Core Capabilities
          </h2>
          <p
            className={`text-lg text-slate-600 transition-all duration-700 delay-150 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Enterprise-grade features designed for solar energy forecasting and battery management
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card
                key={feature.title}
                className={`p-6 hover-lift border-0 shadow-md transition-all duration-700 ${
                  isVisible
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-10"
                }`}
                style={{
                  transitionDelay: isVisible ? `${(index + 1) * 100}ms` : "0ms",
                }}
              >
                {/* Icon */}
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 mb-4">
                  <Icon className="w-6 h-6 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-lg font-bold text-slate-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
}
