import React from 'react';
import { cn } from '@/lib/utils';

interface LoaderProps {
  className?: string;
}

export const Loader: React.FC<LoaderProps> = ({ className }) => {
  return (
    <div className={cn("relative", className)}>
      {/* Animated scanning lines */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Horizontal scanning line */}
        <div className="absolute w-full h-0.5 bg-[#c1f21d] opacity-80 animate-scan-horizontal" 
             style={{
               animation: 'scanHorizontal 2s ease-in-out infinite',
               boxShadow: '0 0 10px #c1f21d'
             }} />
        
        {/* Vertical scanning line */}
        <div className="absolute h-full w-0.5 bg-[#c1f21d] opacity-80 animate-scan-vertical"
             style={{
               animation: 'scanVertical 2.5s ease-in-out infinite 0.3s',
               boxShadow: '0 0 10px #c1f21d'
             }} />
        
        {/* Diagonal scanning lines */}
        <div className="absolute w-full h-0.5 bg-[#c1f21d] opacity-60 origin-left rotate-45"
             style={{
               animation: 'scanDiagonal1 3s ease-in-out infinite 0.6s',
               boxShadow: '0 0 8px #c1f21d'
             }} />
        
        <div className="absolute w-full h-0.5 bg-[#c1f21d] opacity-60 origin-right -rotate-45"
             style={{
               animation: 'scanDiagonal2 2.8s ease-in-out infinite 0.9s',
               boxShadow: '0 0 8px #c1f21d'
             }} />
        
        {/* Grid pattern overlay */}
        <div className="absolute inset-0 opacity-30">
          <div className="grid grid-cols-8 grid-rows-6 h-full w-full">
            {Array.from({ length: 48 }).map((_, i) => (
              <div
                key={i}
                className="border border-[#c1f21d]/20"
                style={{
                  animation: `gridPulse 1.5s ease-in-out infinite ${i * 0.05}s`
                }}
              />
            ))}
          </div>
        </div>
        
        {/* Corner brackets */}
        <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-[#c1f21d] opacity-80" />
        <div className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-[#c1f21d] opacity-80" />
        <div className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-[#c1f21d] opacity-80" />
        <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-[#c1f21d] opacity-80" />
      </div>
      
      {/* Central processing indicator */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-16 h-16 border-2 border-[#c1f21d] rounded-full animate-pulse">
          <div className="w-full h-full border-2 border-transparent border-t-[#c1f21d] rounded-full animate-spin" />
        </div>
      </div>
    </div>
  );
};