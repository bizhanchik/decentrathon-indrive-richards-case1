import React from 'react';
import { Car } from 'lucide-react';

interface HeaderProps {
  onTryItClick: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onTryItClick }) => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#141414]/95 backdrop-blur-sm border-b border-[#c1f21d]/20">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Car className="w-8 h-8 text-[#c1f21d]" />
            <h1 className="text-2xl font-bold text-white tracking-tight">
              RichardsDrive
            </h1>
          </div>
          
          <button
            onClick={onTryItClick}
            className="px-6 py-2 bg-[#c1f21d] text-[#141414] font-semibold rounded-lg hover:bg-[#c1f21d]/90 transform hover:scale-105 transition-all duration-200 ease-out shadow-lg hover:shadow-[#c1f21d]/25"
          >
            Try It
          </button>
        </div>
      </div>
    </header>
  );
};