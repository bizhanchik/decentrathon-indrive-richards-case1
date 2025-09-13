import React, { useState, useEffect } from 'react';
import { ArrowLeft, CheckCircle, AlertTriangle, Shield, Eye, EyeOff, Sparkles } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Loader } from './ui/loader';
import { cn } from '@/lib/utils';

interface AnalysisScreenProps {
  file: File;
  onBack: () => void;
}

interface AnalysisResult {
  cleanliness: {
    score: number;
    status: 'excellent' | 'good' | 'fair' | 'poor';
    description: string;
  };
  integrity: {
    damaged: boolean;
    damageLevel: 'none' | 'minor' | 'moderate' | 'severe';
    description: string;
  };
  heatmap: {
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
    }>;
  };
  ai_analysis?: {
    damage_detected: boolean;
    ai_repaired_image?: string;
  };
}

export const AnalysisScreen: React.FC<AnalysisScreenProps> = ({ file, onBack }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);

  const analyzeImage = async (imageFile: File) => {
    try {
      setIsAnalyzing(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }
      
      const analysisResult = await response.json();
      setResults(analysisResult);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
      // Fallback to mock data on error
      setResults({
        cleanliness: {
          score: 0,
          status: 'poor',
          description: 'Analysis failed - please try again'
        },
        integrity: {
          damaged: false,
          damageLevel: 'none',
          description: 'Unable to analyze image'
        },
        heatmap: {
          areas: []
        }
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    // Create image URL
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    // Start real analysis
    analyzeImage(file);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
      case 'good':
        return 'text-[#c1f21d]';
      case 'fair':
        return 'text-yellow-400';
      case 'poor':
      case 'damaged':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'bg-yellow-400';
      case 'medium':
        return 'bg-orange-400';
      case 'high':
        return 'bg-red-400';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-[#141414] py-8 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Button
            variant="ghost"
            onClick={onBack}
            className="text-gray-300 hover:text-white hover:bg-gray-800 mr-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Upload
          </Button>
          <h1 className="text-3xl font-bold text-white">Vehicle Analysis</h1>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Image Section */}
          <div className="relative">
            {/* Bounding Box Toggle */}
            {!isAnalyzing && results && results.heatmap.areas.length > 0 && (
              <div className="mb-4 flex justify-end">
                <Button
                   variant={showBoundingBoxes ? "default" : "outline"}
                   onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                   className={cn(
                     "text-sm flex items-center gap-2",
                     showBoundingBoxes 
                       ? "bg-[#c1f21d] text-black hover:bg-[#a8d119]" 
                       : "border-gray-600 text-gray-300 hover:bg-gray-800"
                   )}
                 >
                   {showBoundingBoxes ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                   {showBoundingBoxes ? 'Hide' : 'Show'} Detections
                 </Button>
              </div>
            )}
            
            <div className="relative rounded-2xl overflow-hidden bg-[#2c2c2c] border border-black">
              <img
                src={imageUrl}
                alt="Uploaded vehicle"
                className={cn(
                  "w-full h-auto transition-transform duration-500",
                  isAnalyzing ? "scale-105" : "scale-100"
                )}
              />
              
              {/* Analysis Loader Overlay */}
              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/20">
                  <Loader className="w-full h-full" />
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="bg-black/80 rounded-lg p-4 backdrop-blur-sm">
                      <p className="text-[#c1f21d] font-semibold mb-2">Analyzing Vehicle...</p>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-[#c1f21d] h-2 rounded-full transition-all duration-300"
                          style={{ width: '100%', animation: 'progress 3s ease-in-out' }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Bounding Box Overlay */}
              {!isAnalyzing && results && showBoundingBoxes && (
                <div className="absolute inset-0">
                  {results.heatmap.areas.map((area, index) => (
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
                          {/* Label with Confidence */}
                          <div className={cn(
                            "absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white shadow-lg",
                            area.severity === 'high' ? 'bg-red-500' :
                            area.severity === 'medium' ? 'bg-orange-500' :
                            'bg-yellow-500'
                          )}>
                            {area.defect_type?.toUpperCase() || 'DEFECT'}
                            {area.confidence && ` - ${Math.round(area.confidence * 100)}% conf`}
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
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {isAnalyzing ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <div className="w-16 h-16 border-2 border-[#c1f21d] rounded-full animate-spin border-t-transparent mx-auto mb-4" />
                  <p className="text-xl text-gray-300">Processing your image...</p>
                  <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
                </div>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-xl text-red-400 mb-2">Analysis Failed</p>
                  <p className="text-gray-400 mb-4">{error}</p>
                  <Button
                    onClick={() => analyzeImage(file)}
                    className="bg-[#c1f21d] text-[#141414] hover:bg-[#c1f21d]/90 font-semibold"
                  >
                    Try Again
                  </Button>
                </div>
              </div>
            ) : (
              results && (
                <div className="space-y-6 animate-fade-in">
                  {/* Cleanliness Card */}
                  <Card className="bg-[#2c2c2c] border-black">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center text-white">
                        <Shield className="w-5 h-5 mr-2 text-[#c1f21d]" />
                        Cleanliness Assessment
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-gray-300">Score:</span>
                        <span className={cn("font-bold text-lg", getStatusColor(results.cleanliness.status))}>
                          {results.cleanliness.score}/100
                        </span>
                      </div>
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-gray-300">Status:</span>
                        <span className={cn("font-semibold capitalize", getStatusColor(results.cleanliness.status))}>
                          {results.cleanliness.status}
                        </span>
                      </div>
                      <p className="text-gray-400 text-sm">{results.cleanliness.description}</p>
                    </CardContent>
                  </Card>

                  {/* Integrity Card */}
                  <Card className="bg-[#2c2c2c] border-black">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center text-white">
                        {results.integrity.damaged ? (
                          <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
                        ) : (
                          <CheckCircle className="w-5 h-5 mr-2 text-[#c1f21d]" />
                        )}
                        Structural Integrity
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-gray-300">Condition:</span>
                        <span className={cn(
                          "font-semibold",
                          results.integrity.damaged ? "text-red-400" : "text-[#c1f21d]"
                        )}>
                          {results.integrity.damaged ? 'Damaged' : 'Intact'}
                        </span>
                      </div>
                      {results.integrity.damaged && (
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-gray-300">Damage Level:</span>
                          <span className={cn("font-semibold capitalize", getStatusColor(results.integrity.damageLevel))}>
                            {results.integrity.damageLevel}
                          </span>
                        </div>
                      )}
                      <p className="text-gray-400 text-sm">{results.integrity.description}</p>
                    </CardContent>
                  </Card>

                  {/* Damage Heatmap Info */}
                  {results.integrity.damaged && (
                    <Card className="bg-[#2c2c2c] border-black">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-white">Damage Locations</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-gray-400 text-sm mb-4">
                          Colored dots on the image indicate damage locations:
                        </p>
                        <div className="space-y-2">
                          <div className="flex items-center space-x-3">
                            <div className="w-3 h-3 bg-yellow-400 rounded-full" />
                            <span className="text-gray-300 text-sm">Minor damage</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-3 h-3 bg-orange-400 rounded-full" />
                            <span className="text-gray-300 text-sm">Moderate damage</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-3 h-3 bg-red-400 rounded-full" />
                            <span className="text-gray-300 text-sm">Severe damage</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* AI-Repaired Preview */}
                  {results.ai_analysis?.damage_detected && results.ai_analysis?.ai_repaired_image && (
                    <Card className="bg-[#2c2c2c] border-black animate-fade-in">
                      <CardHeader className="pb-3">
                        <CardTitle className="flex items-center text-white">
                          <Sparkles className="w-5 h-5 mr-2 text-[#c1f21d]" />
                          AI-Repaired Preview
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-gray-400 text-sm mb-4">
                          Our AI has generated a preview of how your car would look after repairs:
                        </p>
                        <div className="relative rounded-xl overflow-hidden bg-[#1a1a1a] border border-[#c1f21d]/20">
                          <img
                            src={`data:image/jpeg;base64,${results.ai_analysis.ai_repaired_image}`}
                            alt="AI-repaired vehicle preview"
                            className="w-full h-auto transition-all duration-500 hover:scale-105"
                          />
                          <div className="absolute top-3 right-3">
                            <div className="bg-[#c1f21d]/90 text-[#141414] px-3 py-1 rounded-full text-xs font-semibold">
                              AI Generated
                            </div>
                          </div>
                        </div>
                        <div className="mt-4 p-3 bg-[#1a1a1a] rounded-lg border border-[#c1f21d]/20">
                          <p className="text-[#c1f21d] text-sm font-medium mb-1">
                            ✨ Preview Only
                          </p>
                          <p className="text-gray-400 text-xs">
                            This is an AI-generated visualization of potential repairs. Actual results may vary.
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Action Buttons */}
                  <div className="flex space-x-4 pt-4">
                    <Button
                      onClick={onBack}
                      className="flex-1 bg-[#c1f21d] text-[#141414] hover:bg-[#c1f21d]/90 font-semibold"
                    >
                      Analyze Another Image
                    </Button>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};