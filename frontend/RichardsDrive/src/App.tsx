import React from 'react';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { UploadSection } from './components/UploadSection';

function App() {
  const scrollToUpload = () => {
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
      uploadSection.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#141414] text-white">
      <Header onTryItClick={scrollToUpload} />
      
      <main>
        <HeroSection onGetStartedClick={scrollToUpload} />
        <UploadSection />
      </main>
    </div>
  );
}

export default App;