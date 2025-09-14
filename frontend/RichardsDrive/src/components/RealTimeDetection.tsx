import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, AlertTriangle, CheckCircle, Loader2, Eye, EyeOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DetectionResult {
  damage_detected: boolean;
  total_detections?: number;
  confidence: 'low' | 'medium' | 'high';
  confidence_score?: number;
  timestamp: number;
  mode?: string;
  error?: string;
  heatmap?: {
    areas: Array<{
      x: number;
      y: number;
      severity: 'low' | 'medium' | 'high';
      description: string;
      bbox?: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
      };
      defect_type?: string;
      confidence?: number;
      source_model?: string;
    }>;
  };
  integrity?: {
    damaged: boolean;
    damageLevel: 'none' | 'minor' | 'moderate' | 'severe';
  };
  ensemble_details?: {
    prediction: string;
    reasoning: string;
    override_applied: boolean;
    ensemble_score: number;
  };
  // Legacy fields for backward compatibility
  yolo_detections?: number;
  gemini_confirmed?: boolean;
}

interface RealTimeDetectionProps {
  onBack: () => void;
}

export const RealTimeDetection: React.FC<RealTimeDetectionProps> = ({ onBack }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentResult, setCurrentResult] = useState<DetectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sessionResults, setSessionResults] = useState<DetectionResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Use back camera on mobile
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera access error:', err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8000/ws/realtime-detection`;
    
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setError(null);
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const result: DetectionResult = JSON.parse(event.data);
        setCurrentResult(result);
        setSessionResults(prev => [...prev, result]);
        setIsAnalyzing(false);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection to analysis server failed');
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
  }, []);

  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    // Performance optimization: Downsample frame for faster processing
    const maxWidth = 640; // Reduce from original resolution
    const maxHeight = 480;
    
    let { videoWidth, videoHeight } = video;
    
    // Calculate scaling to maintain aspect ratio
    const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight);
    const scaledWidth = Math.floor(videoWidth * scale);
    const scaledHeight = Math.floor(videoHeight * scale);
    
    // Set canvas size to downsampled dimensions
    canvas.width = scaledWidth;
    canvas.height = scaledHeight;
    
    // Draw and downsample current video frame to canvas
    ctx.drawImage(video, 0, 0, scaledWidth, scaledHeight);
    
    // Convert to blob with optimized quality for faster processing
    canvas.toBlob((blob) => {
      if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        setIsAnalyzing(true);
        wsRef.current.send(blob);
      }
    }, 'image/jpeg', 0.7); // Reduced quality for faster processing
   }, []);

  const startDetection = useCallback(async () => {
    await startCamera();
    connectWebSocket();
    setIsDetecting(true);
    setSessionResults([]);
    
    // Start capturing frames every 2 seconds
    intervalRef.current = setInterval(captureAndSendFrame, 2000);
  }, [startCamera, connectWebSocket, captureAndSendFrame]);

  const stopDetection = useCallback(() => {
    setIsDetecting(false);
    setIsAnalyzing(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    stopCamera();
    disconnectWebSocket();
  }, [stopCamera, disconnectWebSocket]);

  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, [stopDetection]);

  const getStatusColor = (result: DetectionResult | null) => {
    if (!result) return 'text-gray-400';
    if (result.error) return 'text-red-400';
    if (result.damage_detected) {
      const confidence = result.confidence_score || 0;
      if (confidence >= 0.8) return 'text-red-500'; // High confidence damage
      if (confidence >= 0.6) return 'text-orange-400'; // Medium confidence damage
      return 'text-yellow-400'; // Low confidence damage
    }
    return 'text-green-400';
  };

  const getStatusIcon = (result: DetectionResult | null) => {
    if (!result) return null;
    if (result.error) return <AlertTriangle className="w-6 h-6" />;
    if (result.damage_detected) return <AlertTriangle className="w-6 h-6" />;
    return <CheckCircle className="w-6 h-6" />;
  };

  const getStatusText = (result: DetectionResult | null) => {
    if (!result) return 'Ready to analyze';
    if (result.error) return `Error: ${result.error}`;
    if (result.damage_detected) {
      const agreement = result.models_agreement || 0;
      const totalModels = result.total_models || 4;
      const detections = result.total_detections || 0;
      return `Damage Detected (${agreement}/${totalModels} models agree, ${detections} detections, ${result.confidence} confidence)`;
    }
    const agreement = result.models_agreement || 0;
    const totalModels = result.total_models || 4;
    return `No Damage Detected (${agreement}/${totalModels} models agree)`;
  };

  const damageCount = sessionResults.filter(r => r.damage_detected && !r.error).length;
  const totalAnalyzed = sessionResults.filter(r => !r.error).length;

  return (
    <div className="min-h-screen bg-[#141414] text-white">
      {/* Header */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-[#141414]/95 backdrop-blur-sm border-b border-[#c1f21d]/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Real-Time Detection
            </h1>
            <button
              onClick={onBack}
              className="px-6 py-2 bg-[#c1f21d] text-[#141414] font-semibold rounded-lg hover:bg-[#c1f21d]/90 transform hover:scale-105 transition-all duration-200 ease-out shadow-lg hover:shadow-[#c1f21d]/25"
            >
              Back to Upload
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="pt-20 p-4 sm:p-6">
        <div className="max-w-6xl mx-auto">
          {/* Camera View */}
          <div className="relative mb-6">
            {/* Bounding Box Toggle */}
            {isDetecting && currentResult && currentResult.heatmap && currentResult.heatmap.areas.length > 0 && (
              <div className="mb-4 flex justify-end">
                <button
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={cn(
                    "text-sm flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all",
                    showBoundingBoxes 
                      ? "bg-[#c1f21d] text-black hover:bg-[#a8d119]" 
                      : "border border-gray-600 text-gray-300 hover:bg-gray-800"
                  )}
                >
                  {showBoundingBoxes ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  {showBoundingBoxes ? 'Hide' : 'Show'} Detections
                </button>
              </div>
            )}
            
            <div className="aspect-video bg-black rounded-xl overflow-hidden relative shadow-2xl">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {/* Bounding Box Overlay */}
              {isDetecting && currentResult && showBoundingBoxes && currentResult.heatmap && (
                <div className="absolute inset-0">
                  {currentResult.heatmap.areas.map((area, index) => (
                    <div key={index}>
                      {/* Bounding Box */}
                      {area.bbox && (
                        <div
                          className={cn(
                            "absolute border-2 rounded-lg transition-all duration-300",
                            area.severity === 'high' ? 'border-red-400 bg-red-400/10' :
                            area.severity === 'medium' ? 'border-orange-400 bg-orange-400/10' :
                            'border-yellow-400 bg-yellow-400/10'
                          )}
                          style={{
                            left: `${area.bbox.x1}%`,
                            top: `${area.bbox.y1}%`,
                            width: `${area.bbox.x2 - area.bbox.x1}%`,
                            height: `${area.bbox.y2 - area.bbox.y1}%`,
                            boxShadow: `0 0 15px ${area.severity === 'high' ? '#ef444480' : area.severity === 'medium' ? '#f9731680' : '#eab30880'}`
                          }}
                        >
                          {/* Label with Confidence and Model Source */}
                          <div className={cn(
                            "absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white shadow-lg",
                            area.severity === 'high' ? 'bg-red-500' :
                            area.severity === 'medium' ? 'bg-orange-500' :
                            'bg-yellow-500'
                          )}>
                            {area.source_model?.toUpperCase() || 'MODEL'}: {area.defect_type?.toUpperCase() || 'DEFECT'}
                            {area.confidence && ` - ${Math.round(area.confidence * 100)}%`}
                          </div>
                        </div>
                      )}
                      
                      {/* Center Point */}
                      <div
                        className={cn(
                          "absolute w-3 h-3 rounded-full animate-pulse cursor-pointer z-10",
                          area.severity === 'high' ? 'bg-red-400' :
                          area.severity === 'medium' ? 'bg-orange-400' :
                          'bg-yellow-400'
                        )}
                        style={{
                          left: `${area.x}%`,
                          top: `${area.y}%`,
                          transform: 'translate(-50%, -50%)',
                          boxShadow: `0 0 10px ${area.severity === 'high' ? '#ef4444' : area.severity === 'medium' ? '#f97316' : '#eab308'}`
                        }}
                        title={area.description}
                      />
                    </div>
                  ))}
                </div>
              )}
              
              {/* Overlay */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Analysis indicator */}
                {isAnalyzing && (
                  <div className="absolute top-4 left-4 flex items-center space-x-2 bg-black/50 px-3 py-2 rounded-lg">
                    <Loader2 className="w-4 h-4 animate-spin text-[#c1f21d]" />
                    <span className="text-sm text-white">Analyzing...</span>
                  </div>
                )}
                
                {/* Status display */}
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="bg-black/70 backdrop-blur-sm rounded-lg p-4">
                    <div className={`flex items-center space-x-3 ${getStatusColor(currentResult)}`}>
                      {getStatusIcon(currentResult)}
                      <span className="font-medium">{getStatusText(currentResult)}</span>
                    </div>
                    


                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex justify-center mb-6">
            {!isDetecting ? (
              <button
                onClick={startDetection}
                className="flex items-center space-x-2 px-6 sm:px-8 py-3 sm:py-4 bg-[#c1f21d] text-[#141414] font-semibold rounded-lg hover:bg-[#c1f21d]/90 transform hover:scale-105 transition-all duration-200 ease-out shadow-lg hover:shadow-[#c1f21d]/25 text-sm sm:text-base"
              >
                <Camera className="w-4 h-4 sm:w-5 sm:h-5" />
                <span className="hidden sm:inline">Start Real-Time Detection</span>
                <span className="sm:hidden">Start Detection</span>
              </button>
            ) : (
              <button
                onClick={stopDetection}
                className="flex items-center space-x-2 px-6 sm:px-8 py-3 sm:py-4 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transform hover:scale-105 transition-all duration-200 ease-out shadow-lg text-sm sm:text-base"
              >
                <CameraOff className="w-4 h-4 sm:w-5 sm:h-5" />
                <span className="hidden sm:inline">Stop Detection</span>
                <span className="sm:hidden">Stop</span>
              </button>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mb-6">
              <div className="flex items-center space-x-2 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <span className="font-medium">Error</span>
              </div>
              <p className="text-red-300 mt-1">{error}</p>
            </div>
          )}


        </div>
      </div>
    </div>
  );
};