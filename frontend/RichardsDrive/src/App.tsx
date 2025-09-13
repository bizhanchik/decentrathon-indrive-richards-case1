import React, { useState } from 'react';
import { Car, Camera } from 'lucide-react';
import { UploadSection } from './components/UploadSection';
import { AnalysisScreen } from './components/AnalysisScreen';
import { RealTimeDetection } from './components/RealTimeDetection';
import { BackgroundAnimation } from './components/BackgroundAnimation';

type AppState = 'upload' | 'analysis' | 'realtime';

function App() {
  const [currentState, setCurrentState] = useState<AppState>('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleFileUpload = (file: File) => {
    setUploadedFile(file);
    setCurrentState('analysis');
  };

  const handleBackToUpload = () => {
    setCurrentState('upload');
    setUploadedFile(null);
  };

  const handleStartRealTime = () => {
    setCurrentState('realtime');
  };

  return (
    <div className="min-h-screen bg-[#141414] text-white">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#141414]/95 backdrop-blur-sm border-b border-[#c1f21d]/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Car className="w-8 h-8 text-[#c1f21d]" />
              <h1 className="text-2xl font-bold text-white tracking-tight">
                RichardsDrive
              </h1>
            </div>
            
            <div className="flex items-center space-x-3">
              {currentState === 'upload' && (
                <button
                  onClick={handleStartRealTime}
                  className="flex items-center space-x-2 px-4 py-2 bg-transparent border-2 border-[#c1f21d] text-[#c1f21d] font-semibold rounded-lg hover:bg-[#c1f21d] hover:text-[#141414] transform hover:scale-105 transition-all duration-200 ease-out"
                >
                  <Camera className="w-4 h-4" />
                  <span>Real-Time</span>
                </button>
              )}
              
              {(currentState === 'analysis' || currentState === 'realtime') && (
                <button
                  onClick={handleBackToUpload}
                  className="px-6 py-2 bg-[#c1f21d] text-[#141414] font-semibold rounded-lg hover:bg-[#c1f21d]/90 transform hover:scale-105 transition-all duration-200 ease-out shadow-lg hover:shadow-[#c1f21d]/25"
                >
                  {currentState === 'analysis' ? 'New Analysis' : 'Back to Upload'}
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-20">
        {currentState === 'upload' && (
          <>
            {/* Hero Section */}
            <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
              {/* Dynamic Background Animation */}
              <BackgroundAnimation />

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

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={() => {
                      const uploadSection = document.getElementById('upload-section');
                      if (uploadSection) {
                        uploadSection.scrollIntoView({ 
                          behavior: 'smooth',
                          block: 'start'
                        });
                      }
                    }}
                    className="group px-8 py-4 bg-transparent border-2 border-[#c1f21d] text-[#c1f21d] font-semibold rounded-lg hover:bg-[#c1f21d] hover:text-[#141414] transform hover:scale-105 transition-all duration-300 ease-out text-lg"
                  >
                    Upload Photo
                    <span className="inline-block ml-2 transition-transform group-hover:translate-x-1">→</span>
                  </button>
                  
                  <button
                    onClick={handleStartRealTime}
                    className="group flex items-center justify-center space-x-2 px-8 py-4 bg-[#c1f21d] text-[#141414] font-semibold rounded-lg hover:bg-[#c1f21d]/90 transform hover:scale-105 transition-all duration-300 ease-out text-lg"
                  >
                    <Camera className="w-5 h-5" />
                    <span>Real-Time Detection</span>
                  </button>
                </div>
              </div>
            </section>

            {/* Upload Section */}
            <div id="upload-section">
              <UploadSection onFileUpload={handleFileUpload} />
            </div>
          </>
        )}

        {currentState === 'analysis' && uploadedFile && (
          <AnalysisScreen file={uploadedFile} onBack={handleBackToUpload} />
        )}

        {currentState === 'realtime' && (
          <RealTimeDetection onBack={handleBackToUpload} />
        )}
      </main>
    </div>
  );
}

export default App;