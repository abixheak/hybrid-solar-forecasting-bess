import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import FeaturesSection from "@/components/FeaturesSection";
import TechnologySection from "@/components/TechnologySection";
import KPIMetricsSection from "@/components/KPIMetricsSection";
import BESSFeaturesSection from "@/components/BESSFeaturesSection";
import TestimonialsSection from "@/components/TestimonialsSection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <TechnologySection />
      <KPIMetricsSection />
      <BESSFeaturesSection />
      <TestimonialsSection />
      <Footer />
    </div>
  );
}
