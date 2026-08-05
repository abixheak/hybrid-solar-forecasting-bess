import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Card } from "@/components/ui/card";

const TECH_BG = "/manus-storage/technology-architecture-bg_b3ef8069.jpg";

export default function TechnologySection() {
  const { ref, isVisible } = useScrollReveal();

  return (
    <section id="technology" className="section-padding relative overflow-hidden">
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-10"
        style={{
          backgroundImage: `url('${TECH_BG}')`,
        }}
      ></div>

      <div className="container relative z-10" ref={ref}>
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2
            className={`text-4xl md:text-5xl font-bold text-slate-900 mb-4 transition-all duration-700 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Hybrid ML Pipeline
          </h2>
          <p
            className={`text-lg text-slate-600 transition-all duration-700 delay-150 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            SARIMAX baseline combined with LSTM residual correction for superior accuracy
          </p>
        </div>

        {/* Pipeline Diagram */}
        <div className="max-w-5xl mx-auto">
          <div
            className={`transition-all duration-700 delay-300 ${
              isVisible ? "opacity-100 scale-100" : "opacity-0 scale-95"
            }`}
          >
            <Card className="p-8 md:p-12 border-0 shadow-lg bg-white">
              {/* Pipeline Flow */}
              <div className="space-y-8">
                {/* Stage 1 */}
                <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6">
                  <div className="flex-1">
                    <div className="bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">Historical Data</h3>
                      <p className="text-sm opacity-90">
                        NASA POWER & Open-Meteo weather telemetry
                      </p>
                    </div>
                  </div>
                  <div className="hidden md:flex items-center justify-center w-12 h-12 rounded-full bg-slate-200">
                    <svg
                      className="w-6 h-6 text-slate-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </div>
                </div>

                {/* Stage 2 */}
                <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6">
                  <div className="flex-1">
                    <div className="bg-gradient-to-br from-purple-500 to-pink-400 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">Feature Engineering</h3>
                      <p className="text-sm opacity-90">
                        Cyclical encodings, cloud index, lag & rolling features
                      </p>
                    </div>
                  </div>
                  <div className="hidden md:flex items-center justify-center w-12 h-12 rounded-full bg-slate-200">
                    <svg
                      className="w-6 h-6 text-slate-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </div>
                </div>

                {/* Stage 3 - Split into SARIMAX and LSTM */}
                <div className="grid md:grid-cols-2 gap-6">
                  {/* SARIMAX */}
                  <div>
                    <div className="bg-gradient-to-br from-orange-500 to-red-400 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">SARIMAX Model</h3>
                      <p className="text-sm opacity-90">
                        Linear baseline capturing meteorological trends and exogenous variables
                      </p>
                    </div>
                  </div>

                  {/* LSTM */}
                  <div>
                    <div className="bg-gradient-to-br from-green-500 to-emerald-400 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">LSTM Residuals</h3>
                      <p className="text-sm opacity-90">
                        Deep learning trained on SARIMAX residual errors for non-linear corrections
                      </p>
                    </div>
                  </div>
                </div>

                {/* Stage 4 - Combination */}
                <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6">
                  <div className="flex-1">
                    <div className="bg-gradient-to-br from-slate-700 to-slate-900 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">Hybrid Forecast</h3>
                      <p className="text-sm opacity-90">
                        Final prediction = SARIMAX baseline + LSTM residual correction
                      </p>
                    </div>
                  </div>
                </div>

                {/* Stage 5 - Output */}
                <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6">
                  <div className="flex-1">
                    <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg p-6 text-white">
                      <h3 className="font-bold text-lg mb-2">BESS Dispatch</h3>
                      <p className="text-sm opacity-90">
                        Microgrid simulation with SOC tracking, charge/discharge, grid import/export
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-3 gap-6 mt-16">
          {[
            { label: "Forecast Horizon", value: "7 Days" },
            { label: "Cities Supported", value: "8 Major Hubs" },
            { label: "Charge Efficiency", value: "95%" },
          ].map((metric, index) => (
            <div
              key={metric.label}
              className={`text-center transition-all duration-700 ${
                isVisible
                  ? "opacity-100 translate-y-0"
                  : "opacity-0 translate-y-10"
              }`}
              style={{
                transitionDelay: isVisible ? `${400 + index * 100}ms` : "0ms",
              }}
            >
              <p className="text-slate-600 text-sm mb-1">{metric.label}</p>
              <p className="text-2xl font-bold text-slate-900">{metric.value}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
