import React from 'react';
import { Shield, Camera, CheckCircle } from 'lucide-react';

interface HeroSectionProps {
  onGetStartedClick: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onGetStartedClick }) => {
  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-20 left-20 w-64 h-64 border border-[#c1f21d] rounded-full"></div>
        <div className="absolute bottom-20 right-20 w-48 h-48 border border-[#c1f21d] rounded-lg rotate-45"></div>
        <div className="absolute top-1/2 left-1/4 w-32 h-32 border border-[#c1f21d] rounded-lg"></div>
      </div>

      <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
        <div className="mb-8">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight tracking-tight">
            Check Your Car's
            <span className="text-[#c1f21d] block">Condition</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-12 leading-relaxed max-w-3xl mx-auto">
            Upload a photo of your vehicle and get instant, professional assessment of its condition. 
            Powered by advanced AI technology for accurate results.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-8 justify-center items-center mb-16">
          <div className="flex items-center space-x-3 text-gray-300">
            <Camera className="w-6 h-6 text-[#c1f21d]" />
            <span className="text-lg">Upload Photo</span>
          </div>
          <div className="hidden sm:block w-8 h-px bg-[#c1f21d]/50"></div>
          <div className="flex items-center space-x-3 text-gray-300">
            <Shield className="w-6 h-6 text-[#c1f21d]" />
            <span className="text-lg">AI Analysis</span>
          </div>
          <div className="hidden sm:block w-8 h-px bg-[#c1f21d]/50"></div>
          <div className="flex items-center space-x-3 text-gray-300">
            <CheckCircle className="w-6 h-6 text-[#c1f21d]" />
            <span className="text-lg">Get Results</span>
          </div>
        </div>

        <button
          onClick={onGetStartedClick}
          className="group px-8 py-4 bg-transparent border-2 border-[#c1f21d] text-[#c1f21d] font-semibold rounded-lg hover:bg-[#c1f21d] hover:text-[#141414] transform hover:scale-105 transition-all duration-300 ease-out text-lg"
        >
          Get Started
          <span className="inline-block ml-2 transition-transform group-hover:translate-x-1">→</span>
        </button>
      </div>
    </section>
  );
};