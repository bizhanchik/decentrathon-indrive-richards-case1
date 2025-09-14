import React, { useState, useRef } from 'react';
import { Upload, Image, CheckCircle, AlertCircle, Zap } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from '@/lib/utils.js';

interface UploadSectionProps {
  onFileUpload: (file: File, modelType: string) => void;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ onFileUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [modelType, setModelType] = useState<'s'>('s');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    
    if (!allowedTypes.includes(file.type)) {
      setUploadStatus('error');
      return;
    }
    
    if (file.size > maxSize) {
      setUploadStatus('error');
      return;
    }
    
    setSelectedFile(file);
    setUploadStatus('success');
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setUploadStatus('idle');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      console.log('UploadSection - modelType being sent:', modelType);
      onFileUpload(selectedFile, modelType);
    }
  };

  return (
    <section className="min-h-screen flex items-center justify-center py-20 px-6">
      <div className="max-w-4xl mx-auto w-full">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Upload Your Car Photo
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Drag and drop your car image or click to browse. 
            Our AI will analyze the condition and provide detailed insights.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={triggerFileInput}
            className={cn(
              "relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300",
              dragActive
                ? 'border-[#c1f21d] bg-[#c1f21d]/5 scale-105'
                : uploadStatus === 'success'
                ? 'border-[#c1f21d] bg-[#c1f21d]/5'
                : uploadStatus === 'error'
                ? 'border-red-500 bg-red-500/5'
                : 'border-gray-600 hover:border-[#c1f21d]/50 hover:bg-[#c1f21d]/5'
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              onChange={handleChange}
              className="hidden"
            />

            {uploadStatus === 'idle' && (
              <>
                <Upload className={cn(
                  "w-16 h-16 mx-auto mb-6 transition-colors",
                  dragActive ? 'text-[#c1f21d]' : 'text-gray-400'
                )} />
                <h3 className="text-2xl font-semibold text-white mb-4">
                  {dragActive ? 'Drop your image here' : 'Upload Car Image'}
                </h3>
                <p className="text-gray-400 mb-6">
                  Drag and drop your file here, or{' '}
                  <span className="text-[#c1f21d] font-semibold">browse</span>
                </p>
              </>
            )}

            {uploadStatus === 'success' && selectedFile && (
              <>
                <CheckCircle className="w-16 h-16 text-[#c1f21d] mx-auto mb-6" />
                <h3 className="text-2xl font-semibold text-white mb-4">
                  File uploaded successfully!
                </h3>
                <p className="text-gray-400 mb-6">
                  <span className="font-semibold">{selectedFile.name}</span>
                  <br />
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </>
            )}

            {uploadStatus === 'error' && (
              <>
                <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-6" />
                <h3 className="text-2xl font-semibold text-white mb-4">
                  Upload Error
                </h3>
                <p className="text-red-400 mb-6">
                  Please ensure your file is a valid image (JPG, PNG, WebP) under 10MB
                </p>
              </>
            )}
          </div>

          <div className="mt-8 text-center text-sm text-gray-500">
            <div className="flex items-center justify-center space-x-6">
              <span className="flex items-center">
                <Image className="w-4 h-4 mr-2" />
                JPG, PNG, WebP
              </span>
              <span>Max 10MB</span>
            </div>
          </div>

          {uploadStatus === 'success' && (
            <>
              <div className="mt-8 flex justify-center">
                <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                  <h4 className="text-white font-medium mb-3">Using Trained Model:</h4>
                  <div className="flex justify-center">
                    <div className="flex items-center px-4 py-3 rounded-lg bg-[#c1f21d]/20 border border-[#c1f21d] text-white">
                      <CheckCircle className="w-5 h-5 mr-2 text-[#c1f21d]" />
                      <div className="text-left">
                        <div className="font-medium">YOLOv8s</div>
                        <div className="text-xs opacity-70">Trained for car defect detection</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-8 flex justify-center space-x-4">
                <Button
                  variant="outline"
                  onClick={resetUpload}
                  className="border-gray-600 text-gray-300 hover:border-gray-500 hover:text-white hover:bg-transparent"
                >
                  Upload Another
                </Button>
                <Button
                  onClick={handleAnalyze}
                  className="bg-[#c1f21d] text-[#141414] hover:bg-[#c1f21d]/90 font-semibold transform hover:scale-105 transition-all duration-200"
                >
                  Analyze Now
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
};