import { useScrollReveal } from "@/hooks/useScrollReveal";
import { useEffect, useState } from "react";

interface MetricProps {
  label: string;
  value: number;
  suffix: string;
  isVisible: boolean;
}

function CountUpMetric({ label, value, suffix, isVisible }: MetricProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (!isVisible) return;

    let current = 0;
    const increment = value / 30;
    const interval = setInterval(() => {
      current += increment;
      if (current >= value) {
        setDisplayValue(value);
        clearInterval(interval);
      } else {
        setDisplayValue(Math.floor(current));
      }
    }, 30);

    return () => clearInterval(interval);
  }, [isVisible, value]);

  return (
    <div className="text-center">
      <div className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500 mb-2">
        {displayValue}
        {suffix}
      </div>
      <p className="text-slate-600 font-medium">{label}</p>
    </div>
  );
}

export default function KPIMetricsSection() {
  const { ref, isVisible } = useScrollReveal();

  const metrics = [
    { label: "Cities Supported", value: 8, suffix: "" },
    { label: "Day Forecast Horizon", value: 7, suffix: "-" },
    { label: "Charge Efficiency", value: 95, suffix: "%" },
  ];

  return (
    <section className="section-padding bg-gradient-to-r from-slate-900 via-blue-900 to-slate-900 text-white relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"></div>
      </div>

      <div className="container relative z-10" ref={ref}>
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2
            className={`text-4xl md:text-5xl font-bold mb-4 transition-all duration-700 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Enterprise-Grade Performance
          </h2>
          <p
            className={`text-lg text-slate-300 transition-all duration-700 delay-150 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Proven metrics for solar forecasting excellence
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid md:grid-cols-3 gap-12 md:gap-8">
          {metrics.map((metric, index) => (
            <div
              key={metric.label}
              className={`transition-all duration-700 ${
                isVisible
                  ? "opacity-100 translate-y-0"
                  : "opacity-0 translate-y-10"
              }`}
              style={{
                transitionDelay: isVisible ? `${(index + 1) * 100}ms` : "0ms",
              }}
            >
              <CountUpMetric
                label={metric.label}
                value={metric.value}
                suffix={metric.suffix}
                isVisible={isVisible}
              />
            </div>
          ))}
        </div>

        {/* Bottom Divider */}
        <div
          className={`mt-16 pt-16 border-t border-white/10 transition-all duration-700 delay-500 ${
            isVisible ? "opacity-100" : "opacity-0"
          }`}
        >
          <p className="text-center text-slate-400 text-sm">
            Trusted by energy companies across India for mission-critical forecasting
          </p>
        </div>
      </div>
    </section>
  );
}
