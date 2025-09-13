import React from 'react';

interface BackgroundAnimationProps {
  className?: string;
}

export const BackgroundAnimation: React.FC<BackgroundAnimationProps> = ({ className = '' }) => {
  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {/* Large Floating Circles - Mobile Responsive */}
      <div 
        className="absolute w-48 h-48 md:w-96 md:h-96 border border-[#c1f21d] rounded-full opacity-10"
        style={{
          top: '10%',
          left: '5%',
          animation: 'float 20s ease-in-out infinite, glow 4s ease-in-out infinite alternate'
        }}
      ></div>
      
      <div 
        className="absolute w-32 h-32 md:w-64 md:h-64 border border-[#c1f21d] rounded-full opacity-15"
        style={{
          top: '60%',
          right: '10%',
          animation: 'float 25s ease-in-out infinite reverse, glow 3s ease-in-out infinite alternate'
        }}
      ></div>
      
      <div 
        className="absolute w-24 h-24 md:w-48 md:h-48 border border-[#c1f21d] rounded-full opacity-20"
        style={{
          bottom: '20%',
          left: '15%',
          animation: 'float 30s ease-in-out infinite, glow 5s ease-in-out infinite alternate'
        }}
      ></div>
      
      {/* Geometric Shapes - Mobile Responsive */}
      <div 
        className="absolute w-16 h-16 md:w-32 md:h-32 border border-[#c1f21d] opacity-25 rotate-45"
        style={{
          top: '25%',
          right: '25%',
          animation: 'rotate 15s linear infinite, pulse 2s ease-in-out infinite'
        }}
      ></div>
      
      <div 
        className="absolute w-12 h-12 md:w-24 md:h-24 border border-[#c1f21d] opacity-30"
        style={{
          top: '70%',
          left: '70%',
          animation: 'rotate 20s linear infinite reverse, pulse 3s ease-in-out infinite'
        }}
      ></div>
      

      
      {/* Floating Dots - Mobile Optimized */}
      <div 
        className="hidden md:block absolute w-3 h-3 bg-[#c1f21d] rounded-full opacity-60"
        style={{
          top: '15%',
          left: '80%',
          animation: 'bounce 4s ease-in-out infinite'
        }}
      ></div>
      
      <div 
        className="absolute w-2 h-2 bg-[#c1f21d] rounded-full opacity-50"
        style={{
          top: '80%',
          left: '30%',
          animation: 'bounce 5s ease-in-out infinite reverse'
        }}
      ></div>
      
      {/* Pulsing Rings - Mobile Responsive */}
      <div 
        className="absolute w-12 h-12 md:w-20 md:h-20 border border-[#c1f21d] rounded-full opacity-40"
        style={{
          top: '40%',
          left: '40%',
          animation: 'ping 6s cubic-bezier(0, 0, 0.2, 1) infinite'
        }}
      ></div>
      
      <div 
        className="hidden md:block absolute w-16 h-16 border border-[#c1f21d] rounded-full opacity-35"
        style={{
          top: '65%',
          right: '40%',
          animation: 'ping 8s cubic-bezier(0, 0, 0.2, 1) infinite'
        }}
      ></div>
      

    </div>
  );
};