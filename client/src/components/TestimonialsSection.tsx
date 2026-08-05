import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Card } from "@/components/ui/card";
import { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Star } from "lucide-react";

const testimonials = [
  {
    title: "Predictive Accuracy at Scale",
    description:
      "The hybrid SARIMAX-LSTM architecture has improved our forecast accuracy by 23% compared to traditional methods. The real-time BESS dispatch optimization has reduced grid curtailment losses significantly.",
    company: "GreenPower Solar Farms",
    role: "Operations Director",
    rating: 5,
  },
  {
    title: "Enterprise-Grade Reliability",
    description:
      "SolarStream has become mission-critical infrastructure for our microgrid operations. The 7-day forecast horizon and 95% charge efficiency metrics give us confidence in planning.",
    company: "Renewable Energy Solutions Ltd",
    role: "Chief Technology Officer",
    rating: 5,
  },
  {
    title: "Seamless Integration",
    description:
      "Integration with our existing monitoring systems was seamless. The live weather synchronization across 8 cities provides comprehensive coverage for our distributed solar assets.",
    company: "Energy Management Corp",
    role: "System Administrator",
    rating: 5,
  },
  {
    title: "ROI Within Months",
    description:
      "The intelligent demand profile adjustment and grid import/export optimization delivered ROI within the first quarter. The platform pays for itself through efficiency gains.",
    company: "Smart Grid Innovations",
    role: "Finance Manager",
    rating: 5,
  },
];

export default function TestimonialsSection() {
  const { ref, isVisible } = useScrollReveal();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [autoPlay, setAutoPlay] = useState(true);

  useEffect(() => {
    if (!autoPlay) return;

    const timer = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % testimonials.length);
    }, 5000);

    return () => clearInterval(timer);
  }, [autoPlay]);

  const goToPrevious = () => {
    setCurrentIndex((prev) =>
      prev === 0 ? testimonials.length - 1 : prev - 1
    );
    setAutoPlay(false);
  };

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % testimonials.length);
    setAutoPlay(false);
  };

  return (
    <section className="section-padding bg-slate-50">
      <div className="container" ref={ref}>
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h2
            className={`text-4xl md:text-5xl font-bold text-slate-900 mb-4 transition-all duration-700 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Trusted by Industry Leaders
          </h2>
          <p
            className={`text-lg text-slate-600 transition-all duration-700 delay-150 ${
              isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
            }`}
          >
            Real results from energy companies using SolarStream
          </p>
        </div>

        {/* Carousel */}
        <div
          className={`max-w-4xl mx-auto transition-all duration-700 delay-300 ${
            isVisible ? "opacity-100 scale-100" : "opacity-0 scale-95"
          }`}
        >
          <div className="relative">
            {/* Testimonial Card */}
            <Card className="p-8 md:p-12 border-0 shadow-lg bg-white">
              <div className="space-y-6">
                {/* Rating */}
                <div className="flex gap-1">
                  {Array.from({ length: testimonials[currentIndex].rating }).map(
                    (_, i) => (
                      <Star
                        key={i}
                        className="w-5 h-5 fill-yellow-400 text-yellow-400"
                      />
                    )
                  )}
                </div>

                {/* Quote */}
                <blockquote className="text-xl md:text-2xl font-semibold text-slate-900 leading-relaxed">
                  "{testimonials[currentIndex].description}"
                </blockquote>

                {/* Author */}
                <div className="pt-4 border-t border-slate-200">
                  <p className="font-bold text-slate-900">
                    {testimonials[currentIndex].title}
                  </p>
                  <p className="text-slate-600 text-sm mt-1">
                    {testimonials[currentIndex].role} at
                    <span className="font-semibold ml-1">
                      {testimonials[currentIndex].company}
                    </span>
                  </p>
                </div>
              </div>
            </Card>

            {/* Navigation Buttons */}
            <div className="flex items-center justify-between mt-8">
              <button
                onClick={goToPrevious}
                className="p-2 hover:bg-slate-200 rounded-full transition-colors"
                aria-label="Previous testimonial"
              >
                <ChevronLeft className="w-6 h-6 text-slate-900" />
              </button>

              {/* Dots */}
              <div className="flex gap-2">
                {testimonials.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setCurrentIndex(index);
                      setAutoPlay(false);
                    }}
                    className={`w-2 h-2 rounded-full transition-all duration-300 ${
                      index === currentIndex
                        ? "bg-blue-600 w-8"
                        : "bg-slate-300 hover:bg-slate-400"
                    }`}
                    aria-label={`Go to testimonial ${index + 1}`}
                  />
                ))}
              </div>

              <button
                onClick={goToNext}
                className="p-2 hover:bg-slate-200 rounded-full transition-colors"
                aria-label="Next testimonial"
              >
                <ChevronRight className="w-6 h-6 text-slate-900" />
              </button>
            </div>

            {/* Counter */}
            <p className="text-center text-sm text-slate-600 mt-6">
              {currentIndex + 1} / {testimonials.length}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
