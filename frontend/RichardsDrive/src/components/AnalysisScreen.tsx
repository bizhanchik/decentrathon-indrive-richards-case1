import React, { useState, useEffect } from 'react';
import { ArrowLeft, CheckCircle, AlertTriangle, AlertCircle, Shield, Eye, EyeOff, Sparkles } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Loader } from './ui/loader';
import { cn } from '@/lib/utils';

interface AnalysisScreenProps {
  file: File;
  modelType: string;
  onBack: () => void;
}

interface SingleModelResult {
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
  model_info?: {
    type: string;
    name: string;
    requested_type?: string;
    fallback_used?: boolean;
  };
}

interface AllModelsAnalysisResult {
  gemini_analysis?: {
    damage_detected: boolean;
    repaired_image: string | null;
    model_info: {
      type: string;
      name: string;
      description: string;
    };
  };
  main_analysis: SingleModelResult | null;
  model2_analysis: SingleModelResult | null;
  model3_analysis: SingleModelResult | null;
  model4_analysis: SingleModelResult | null;
  individual_boxes?: {
    model1: Array<{
      box: number[];
      confidence: number;
      class: string;
    }>;
    model2: Array<{
      box: number[];
      confidence: number;
      class: string;
    }>;
    model3: Array<{
      box: number[];
      confidence: number;
      class: string;
    }>;
    model4: Array<{
      box: number[];
      confidence: number;
      class: string;
    }>;
  };
  unified_boxes?: Array<{
    box: number[];
    confidence: number;
    class: string;
  }>;
  combined_summary: {
    overall_score: number;
    overall_status: 'excellent' | 'good' | 'fair' | 'poor';
    description: string;
    recommendations: string[];
    models_used: {
      main_model: boolean;
      model2: boolean;
      model3: boolean;
      model4: boolean;
    };
    analysis_coverage: string;
    ensemble_details?: {
      prediction: string;
      confidence: number;
      reasoning: string;
      override_applied: boolean;
      ensemble_score: number;
      damage_detected: boolean;
      damage_types: Array<{
        type: string;
        confidence: number;
        supporting_models: string[];
      }>;
      severity_score: number;
      models_agreement: number;
      unified_detections?: Array<{
        bbox: number[];
        class_name: string;
        confidence: number;
        source_models: string[];
        aggregated_confidence: number;
        detection_count: number;
        severity?: string;
      }>;
      numeric_assessment?: {
        score: number;
        severity_level: string;
        damage_impact: number;
        confidence_factor: number;
        area_coverage: number;
        damage_breakdown: Record<string, {
          count: number;
          total_impact: number;
          avg_confidence: number;
          severity_level: string;
          confidences: number[];
        }>;
        total_damages: number;
      };
    };
  };
  model_info: {
    main_model: any;
    model2: any;
    model3: any;
    model4: any;
  };
}

type AnalysisResult = SingleModelResult | AllModelsAnalysisResult;

export const AnalysisScreen: React.FC<AnalysisScreenProps> = ({ file, modelType, onBack }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showMainModelBoxes, setShowMainModelBoxes] = useState(false);
  const [showModel2Boxes, setShowModel2Boxes] = useState(false);
  const [showModel3Boxes, setShowModel3Boxes] = useState(false);
  const [showModel4Boxes, setShowModel4Boxes] = useState(false);
  const [boxDisplayMode, setBoxDisplayMode] = useState<'individual' | 'unified'>('unified');


  const analyzeImage = async (imageFile: File) => {
    try {
      setIsAnalyzing(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('model_type', modelType);
      
      console.log(`Analyzing with model type: ${modelType}`);
      
      // Use analyze-all endpoint for comprehensive results from all 4 models
      const response = await fetch('http://localhost:8000/analyze-all', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }
      
      const analysisResult = await response.json();
      
      // Transform backend response to match frontend expectations
      const transformedResult = {
        gemini_analysis: analysisResult.gemini_analysis || null,
        main_analysis: analysisResult.main_model?.analysis || null,
        model2_analysis: analysisResult.model2?.analysis || null,
        model3_analysis: analysisResult.model3?.analysis || null,
        model4_analysis: analysisResult.model4?.analysis || null,
        individual_boxes: analysisResult.individual_boxes || null,
        unified_boxes: analysisResult.unified_boxes || null,
        combined_summary: analysisResult.combined_summary || null,
        model_info: {
          main_model: analysisResult.main_model?.model_info || null,
          model2: analysisResult.model2?.model_info || null,
          model3: analysisResult.model3?.model_info || null,
          model4: analysisResult.model4?.model_info || null
        }
      };
      
      console.log('Backend response:', analysisResult);
      console.log('Transformed result:', transformedResult);
      
      setResults(transformedResult);
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
        },
        model_info: {
          type: modelType,
          name: `YOLOv8${modelType}`
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
            {/* Detection Toggles */}
            {!isAnalyzing && results && (
              ('heatmap' in results && results.heatmap.areas.length > 0) ||
              ('main_analysis' in results && (
                (results.main_analysis?.heatmap.areas.length || 0) > 0 ||
                (results.model2_analysis?.heatmap.areas.length || 0) > 0
              ))
            ) && (
              <div className="mb-4 space-y-3">
                {/* Main Toggle */}
                <div className="flex justify-end">
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
                     {showBoundingBoxes ? 'Hide' : 'Show'} All Detections
                   </Button>
                </div>
                
                {/* WBF Display Mode Toggle */}
                {'combined_summary' in results && showBoundingBoxes && (
                  <div className="flex justify-end mb-2">
                    <div className="flex gap-2">
                      <Button
                        variant={boxDisplayMode === 'individual' ? "default" : "outline"}
                        size="sm"
                        onClick={() => setBoxDisplayMode('individual')}
                        className={cn(
                          "text-xs",
                          boxDisplayMode === 'individual'
                            ? "bg-orange-500 text-white hover:bg-orange-600"
                            : "border-orange-500/40 text-orange-400 hover:bg-orange-500/10"
                        )}
                      >
                        Individual Models
                      </Button>
                      <Button
                        variant={boxDisplayMode === 'unified' ? "default" : "outline"}
                        size="sm"
                        onClick={() => setBoxDisplayMode('unified')}
                        className={cn(
                          "text-xs",
                          boxDisplayMode === 'unified'
                            ? "bg-green-500 text-white hover:bg-green-600"
                            : "border-green-500/40 text-green-400 hover:bg-green-500/10"
                        )}
                      >
                        Unified (WBF)
                      </Button>
                    </div>
                  </div>
                )}
                
                {/* Individual Model Toggles */}
                {'combined_summary' in results && showBoundingBoxes && boxDisplayMode === 'individual' && (
                  <div className="flex justify-end">
                    <div className="flex flex-wrap gap-2">
                      {results.combined_summary.models_used.main_model && (
                        <Button
                          variant={showMainModelBoxes ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowMainModelBoxes(!showMainModelBoxes)}
                          className={cn(
                            "text-xs flex items-center gap-1",
                            showMainModelBoxes 
                              ? "bg-[#c1f21d]/80 text-black hover:bg-[#c1f21d]" 
                              : "border-[#c1f21d]/40 text-[#c1f21d] hover:bg-[#c1f21d]/10"
                          )}
                        >
                          {showMainModelBoxes ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                          Main
                        </Button>
                      )}
                      {results.combined_summary.models_used.model2 && (
                        <Button
                          variant={showModel2Boxes ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowModel2Boxes(!showModel2Boxes)}
                          className={cn(
                            "text-xs flex items-center gap-1",
                            showModel2Boxes 
                              ? "bg-blue-500/80 text-white hover:bg-blue-500" 
                              : "border-blue-500/40 text-blue-400 hover:bg-blue-500/10"
                          )}
                        >
                          {showModel2Boxes ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                          Model 2
                        </Button>
                      )}
                      {results.combined_summary.models_used.model3 && (
                        <Button
                          variant={showModel3Boxes ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowModel3Boxes(!showModel3Boxes)}
                          className={cn(
                            "text-xs flex items-center gap-1",
                            showModel3Boxes 
                              ? "bg-purple-500/80 text-white hover:bg-purple-500" 
                              : "border-purple-500/40 text-purple-400 hover:bg-purple-500/10"
                          )}
                        >
                          {showModel3Boxes ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                          Model 3
                        </Button>
                      )}
                      {results.combined_summary.models_used.model4 && (
                        <Button
                          variant={showModel4Boxes ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowModel4Boxes(!showModel4Boxes)}
                          className={cn(
                            "text-xs flex items-center gap-1",
                            showModel4Boxes 
                              ? "bg-orange-500/80 text-white hover:bg-orange-500" 
                              : "border-orange-500/40 text-orange-400 hover:bg-orange-500/10"
                          )}
                        >
                          {showModel4Boxes ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                          Model 4
                        </Button>
                      )}

                    </div>
                  </div>
                )}
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
                  {/* Single Analysis Bounding Boxes */}
                  {'heatmap' in results && results.heatmap.areas.map((area, index) => (
                    <div key={`single-${index}`}>
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
                  
                  {/* All Models Analysis Bounding Boxes */}
                  {'main_analysis' in results && (
                    <>
                      {/* Main Model Bounding Boxes */}
                      {showMainModelBoxes && results.main_analysis?.heatmap.areas.map((area, index) => (
                        <div key={`main-${index}`}>
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
                                MAIN: {area.defect_type?.toUpperCase() || 'DEFECT'}
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
                            title={`Main Model: ${area.description}`}
                          />
                        </div>
                      ))}


                      {/* Model 2 Bounding Boxes */}
                      {showModel2Boxes && results.model2_analysis?.heatmap.areas.map((area, index) => (
                        <div key={`model2-${index}`}>
                          {/* Bounding Box */}
                          {area.bbox && (
                            <div
                              className={cn(
                                "absolute border-2 rounded-lg transition-all duration-300",
                                area.severity === 'high' ? 'border-blue-400 bg-blue-400/10' :
                                area.severity === 'medium' ? 'border-blue-300 bg-blue-300/10' :
                                'border-blue-200 bg-blue-200/10'
                              )}
                              style={{
                                left: `${area.bbox.x1}%`,
                                top: `${area.bbox.y1}%`,
                                width: `${area.bbox.x2 - area.bbox.x1}%`,
                                height: `${area.bbox.y2 - area.bbox.y1}%`,
                                boxShadow: `0 0 15px ${area.severity === 'high' ? '#3b82f680' : area.severity === 'medium' ? '#93c5fd80' : '#dbeafe80'}`
                              }}
                            >
                              {/* Label with Confidence */}
                              <div className={cn(
                                "absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white shadow-lg",
                                area.severity === 'high' ? 'bg-blue-500' :
                                area.severity === 'medium' ? 'bg-blue-400' :
                                'bg-blue-300'
                              )}>
                                M2: {area.defect_type?.toUpperCase() || 'DEFECT'}
                                {area.confidence && ` - ${Math.round(area.confidence * 100)}% conf`}
                              </div>
                            </div>
                          )}
                          
                          {/* Center Point */}
                          <div
                            className={cn(
                              "absolute w-3 h-3 rounded-full animate-pulse cursor-pointer z-10",
                              area.severity === 'high' ? 'bg-blue-400' :
                              area.severity === 'medium' ? 'bg-blue-300' :
                              'bg-blue-200'
                            )}
                            style={{
                              left: `${area.x}%`,
                              top: `${area.y}%`,
                              transform: 'translate(-50%, -50%)',
                              boxShadow: `0 0 10px ${area.severity === 'high' ? '#3b82f6' : area.severity === 'medium' ? '#93c5fd' : '#dbeafe'}`
                            }}
                            title={`Model 2: ${area.description}`}
                          />
                        </div>
                      ))}

                      {/* Model 3 Bounding Boxes */}
                      {showModel3Boxes && results.model3_analysis?.heatmap.areas.map((area, index) => (
                        <div key={`model3-${index}`}>
                          {/* Bounding Box */}
                          {area.bbox && (
                            <div
                              className={cn(
                                "absolute border-2 rounded-lg transition-all duration-300",
                                area.severity === 'high' ? 'border-purple-400 bg-purple-400/10' :
                                area.severity === 'medium' ? 'border-purple-300 bg-purple-300/10' :
                                'border-purple-200 bg-purple-200/10'
                              )}
                              style={{
                                left: `${area.bbox.x1}%`,
                                top: `${area.bbox.y1}%`,
                                width: `${area.bbox.x2 - area.bbox.x1}%`,
                                height: `${area.bbox.y2 - area.bbox.y1}%`,
                                boxShadow: `0 0 15px ${area.severity === 'high' ? '#a855f780' : area.severity === 'medium' ? '#c4b5fd80' : '#e9d5ff80'}`
                              }}
                            >
                              {/* Label with Confidence */}
                              <div className={cn(
                                "absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white shadow-lg",
                                area.severity === 'high' ? 'bg-purple-500' :
                                area.severity === 'medium' ? 'bg-purple-400' :
                                'bg-purple-300'
                              )}>
                                M3: {area.defect_type?.toUpperCase() || 'DEFECT'}
                                {area.confidence && ` - ${Math.round(area.confidence * 100)}% conf`}
                              </div>
                            </div>
                          )}
                          
                          {/* Center Point */}
                          <div
                            className={cn(
                              "absolute w-3 h-3 rounded-full animate-pulse cursor-pointer z-10",
                              area.severity === 'high' ? 'bg-purple-400' :
                              area.severity === 'medium' ? 'bg-purple-300' :
                              'bg-purple-200'
                            )}
                            style={{
                              left: `${area.x}%`,
                              top: `${area.y}%`,
                              transform: 'translate(-50%, -50%)',
                              boxShadow: `0 0 10px ${area.severity === 'high' ? '#a855f7' : area.severity === 'medium' ? '#c4b5fd' : '#e9d5ff'}`
                            }}
                            title={`Model 3: ${area.description}`}
                          />
                        </div>
                      ))}

                      {/* Model 4 Bounding Boxes */}
                      {showModel4Boxes && results.model4_analysis?.heatmap.areas.map((area, index) => (
                        <div key={`model4-${index}`}>
                          {/* Bounding Box */}
                          {area.bbox && (
                            <div
                              className={cn(
                                "absolute border-2 rounded-lg transition-all duration-300",
                                area.severity === 'high' ? 'border-orange-400 bg-orange-400/10' :
                                area.severity === 'medium' ? 'border-orange-300 bg-orange-300/10' :
                                'border-orange-200 bg-orange-200/10'
                              )}
                              style={{
                                left: `${area.bbox.x1}%`,
                                top: `${area.bbox.y1}%`,
                                width: `${area.bbox.x2 - area.bbox.x1}%`,
                                height: `${area.bbox.y2 - area.bbox.y1}%`,
                                boxShadow: `0 0 15px ${area.severity === 'high' ? '#f97316b0' : area.severity === 'medium' ? '#fdba74b0' : '#fed7aab0'}`
                              }}
                            >
                              {/* Label with Confidence */}
                              <div className={cn(
                                "absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white shadow-lg",
                                area.severity === 'high' ? 'bg-orange-500' :
                                area.severity === 'medium' ? 'bg-orange-400' :
                                'bg-orange-300'
                              )}>
                                M4: {area.defect_type?.toUpperCase() || 'DEFECT'}
                                {area.confidence && ` - ${Math.round(area.confidence * 100)}% conf`}
                              </div>
                            </div>
                          )}
                          
                          {/* Center Point */}
                          <div
                            className={cn(
                              "absolute w-3 h-3 rounded-full animate-pulse cursor-pointer z-10",
                              area.severity === 'high' ? 'bg-orange-400' :
                              area.severity === 'medium' ? 'bg-orange-300' :
                              'bg-orange-200'
                            )}
                            style={{
                              left: `${area.x}%`,
                              top: `${area.y}%`,
                              transform: 'translate(-50%, -50%)',
                              boxShadow: `0 0 10px ${area.severity === 'high' ? '#f97316' : area.severity === 'medium' ? '#fdba74' : '#fed7aa'}`
                            }}
                            title={`Model 4: ${area.description}`}
                          />
                        </div>
                      ))}                    </>                  )}                  
                  {/* WBF Unified Boxes */}
                  {'combined_summary' in results && boxDisplayMode === 'unified' && results.unified_boxes && (
                    <>
                      {results.unified_boxes.map((detection, index) => {
                        // Convert [x_min, y_min, x_max, y_max] to percentage coordinates
                        const [x_min, y_min, x_max, y_max] = detection.box;
                        const imageElement = document.querySelector('img');
                        if (!imageElement) return null;
                        
                        const imgWidth = imageElement.naturalWidth;
                        const imgHeight = imageElement.naturalHeight;
                        
                        const x1_percent = (x_min / imgWidth) * 100;
                        const y1_percent = (y_min / imgHeight) * 100;
                        const x2_percent = (x_max / imgWidth) * 100;
                        const y2_percent = (y_max / imgHeight) * 100;
                        
                        return (
                          <div key={`unified-${index}`}>
                            {/* Bounding Box */}
                            <div
                              className="absolute border-2 border-green-400 bg-green-400/10 rounded-lg transition-all duration-300"
                              style={{
                                left: `${x1_percent}%`,
                                top: `${y1_percent}%`,
                                width: `${x2_percent - x1_percent}%`,
                                height: `${y2_percent - y1_percent}%`,
                                boxShadow: '0 0 15px #22c55e80'
                              }}
                            >
                              {/* Label with Confidence */}
                              <div className="absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white bg-green-500 shadow-lg">
                                WBF: {detection.class.toUpperCase()} - {Math.round(detection.confidence * 100)}% conf
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </>
                  )}
                  
                  {/* Individual Model Boxes */}
                  {'combined_summary' in results && boxDisplayMode === 'individual' && results.individual_boxes && (
                    <>
                      {/* Model 1 Boxes */}
                      {showMainModelBoxes && results.individual_boxes.model1?.map((detection, index) => {
                        const [x_min, y_min, x_max, y_max] = detection.box;
                        const imageElement = document.querySelector('img');
                        if (!imageElement) return null;
                        
                        const imgWidth = imageElement.naturalWidth;
                        const imgHeight = imageElement.naturalHeight;
                        
                        const x1_percent = (x_min / imgWidth) * 100;
                        const y1_percent = (y_min / imgHeight) * 100;
                        const x2_percent = (x_max / imgWidth) * 100;
                        const y2_percent = (y_max / imgHeight) * 100;
                        
                        return (
                          <div key={`model1-${index}`}>
                            <div
                              className="absolute border-2 border-yellow-400 bg-yellow-400/10 rounded-lg transition-all duration-300"
                              style={{
                                left: `${x1_percent}%`,
                                top: `${y1_percent}%`,
                                width: `${x2_percent - x1_percent}%`,
                                height: `${y2_percent - y1_percent}%`,
                                boxShadow: '0 0 15px #eab30880'
                              }}
                            >
                              <div className="absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-black bg-yellow-400 shadow-lg">
                                M1: {detection.class.toUpperCase()} - {Math.round(detection.confidence * 100)}%
                              </div>
                            </div>
                          </div>
                        );
                      })}
                      
                      {/* Model 2 Boxes */}
                      {showModel2Boxes && results.individual_boxes.model2?.map((detection, index) => {
                        const [x_min, y_min, x_max, y_max] = detection.box;
                        const imageElement = document.querySelector('img');
                        if (!imageElement) return null;
                        
                        const imgWidth = imageElement.naturalWidth;
                        const imgHeight = imageElement.naturalHeight;
                        
                        const x1_percent = (x_min / imgWidth) * 100;
                        const y1_percent = (y_min / imgHeight) * 100;
                        const x2_percent = (x_max / imgWidth) * 100;
                        const y2_percent = (y_max / imgHeight) * 100;
                        
                        return (
                          <div key={`model2-${index}`}>
                            <div
                              className="absolute border-2 border-blue-400 bg-blue-400/10 rounded-lg transition-all duration-300"
                              style={{
                                left: `${x1_percent}%`,
                                top: `${y1_percent}%`,
                                width: `${x2_percent - x1_percent}%`,
                                height: `${y2_percent - y1_percent}%`,
                                boxShadow: '0 0 15px #3b82f680'
                              }}
                            >
                              <div className="absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white bg-blue-500 shadow-lg">
                                M2: {detection.class.toUpperCase()} - {Math.round(detection.confidence * 100)}%
                              </div>
                            </div>
                          </div>
                        );
                      })}
                      
                      {/* Model 3 Boxes */}
                      {showModel3Boxes && results.individual_boxes.model3?.map((detection, index) => {
                        const [x_min, y_min, x_max, y_max] = detection.box;
                        const imageElement = document.querySelector('img');
                        if (!imageElement) return null;
                        
                        const imgWidth = imageElement.naturalWidth;
                        const imgHeight = imageElement.naturalHeight;
                        
                        const x1_percent = (x_min / imgWidth) * 100;
                        const y1_percent = (y_min / imgHeight) * 100;
                        const x2_percent = (x_max / imgWidth) * 100;
                        const y2_percent = (y_max / imgHeight) * 100;
                        
                        return (
                          <div key={`model3-${index}`}>
                            <div
                              className="absolute border-2 border-purple-400 bg-purple-400/10 rounded-lg transition-all duration-300"
                              style={{
                                left: `${x1_percent}%`,
                                top: `${y1_percent}%`,
                                width: `${x2_percent - x1_percent}%`,
                                height: `${y2_percent - y1_percent}%`,
                                boxShadow: '0 0 15px #a855f780'
                              }}
                            >
                              <div className="absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white bg-purple-500 shadow-lg">
                                M3: {detection.class.toUpperCase()} - {Math.round(detection.confidence * 100)}%
                              </div>
                            </div>
                          </div>
                        );
                      })}
                      
                      {/* Model 4 Boxes */}
                      {showModel4Boxes && results.individual_boxes.model4?.map((detection, index) => {
                        const [x_min, y_min, x_max, y_max] = detection.box;
                        const imageElement = document.querySelector('img');
                        if (!imageElement) return null;
                        
                        const imgWidth = imageElement.naturalWidth;
                        const imgHeight = imageElement.naturalHeight;
                        
                        const x1_percent = (x_min / imgWidth) * 100;
                        const y1_percent = (y_min / imgHeight) * 100;
                        const x2_percent = (x_max / imgWidth) * 100;
                        const y2_percent = (y_max / imgHeight) * 100;
                        
                        return (
                          <div key={`model4-${index}`}>
                            <div
                              className="absolute border-2 border-orange-400 bg-orange-400/10 rounded-lg transition-all duration-300"
                              style={{
                                left: `${x1_percent}%`,
                                top: `${y1_percent}%`,
                                width: `${x2_percent - x1_percent}%`,
                                height: `${y2_percent - y1_percent}%`,
                                boxShadow: '0 0 15px #f97316880'
                              }}
                            >
                              <div className="absolute -top-6 left-0 px-2 py-1 rounded text-xs font-semibold text-white bg-orange-500 shadow-lg">
                                M4: {detection.class.toUpperCase()} - {Math.round(detection.confidence * 100)}%
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </>
                  )}
                </div>              )}            </div>          </div>

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
                  {/* Check if it's dual analysis result */}
                  {'combined_summary' in results ? (
                    // Dual Analysis Results
                    <>
                      {/* Combined Summary Card */}
                      <Card className="bg-[#2c2c2c] border-black">
                        <CardHeader className="pb-3">
                          <CardTitle className="flex items-center text-white">
                            <Shield className="w-5 h-5 mr-2 text-[#c1f21d]" />
                            Overall Assessment
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-gray-300">Status:</span>
                            <span className={cn("font-semibold capitalize", getStatusColor(results.combined_summary.overall_status))}>
                              {results.combined_summary.overall_status}
                            </span>
                          </div>
                          <p className="text-gray-400 text-sm mb-4">{results.combined_summary.description}</p>
                          


                          {/* Analysis Summary */}
                          {results.combined_summary.ensemble_details && (
                            <div className="mt-4 border-t border-gray-600 pt-4">
                              <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
                                <Shield className="w-4 h-4 mr-2 text-[#c1f21d]" />
                                Analysis Summary
                              </h4>
                              <div className="bg-[#1a1a1a] rounded-lg p-4 space-y-4">
                                {/* Main Result with Large Visual Indicator */}
                                <div className="text-center py-4">
                                  <div className={cn(
                                    "inline-flex items-center justify-center w-16 h-16 rounded-full mb-3",
                                    (results.combined_summary.ensemble_details.prediction === 'damage_detected' || results.combined_summary.ensemble_details.prediction === 'damage') ? "bg-red-900/30 border-2 border-red-500" : 
                                    results.combined_summary.ensemble_details.prediction === 'uncertain_condition' ? "bg-yellow-900/30 border-2 border-yellow-500" : 
                                    "bg-green-900/30 border-2 border-green-500"
                                  )}>
                                    {(results.combined_summary.ensemble_details.prediction === 'damage_detected' || results.combined_summary.ensemble_details.prediction === 'damage') ? (
                                      <AlertTriangle className="w-8 h-8 text-red-400" />
                                    ) : results.combined_summary.ensemble_details.prediction === 'uncertain_condition' ? (
                                      <AlertCircle className="w-8 h-8 text-yellow-400" />
                                    ) : (
                                      <CheckCircle className="w-8 h-8 text-green-400" />
                                    )}
                                  </div>
                                  <h3 className={cn(
                                    "text-xl font-bold mb-2",
                                    (results.combined_summary.ensemble_details.prediction === 'damage_detected' || results.combined_summary.ensemble_details.prediction === 'damage') ? "text-red-400" : 
                                    results.combined_summary.ensemble_details.prediction === 'uncertain_condition' ? "text-yellow-400" : 
                                    "text-green-400"
                                  )}>
                                    {(results.combined_summary.ensemble_details.prediction === 'damage_detected' || results.combined_summary.ensemble_details.prediction === 'damage') ? 'Damage Detected' : 
                                     results.combined_summary.ensemble_details.prediction === 'uncertain_condition' ? 'Uncertain Condition' : 
                                     'No Damage Found'}
                                  </h3>
                                  <p className="text-gray-400 text-sm">
                                    Confidence: {Math.round(results.combined_summary.ensemble_details.confidence * 100)}%
                                  </p>
                                </div>

                                {/* Damage Types - Show if damage detected or uncertain condition */}
                                {((results.combined_summary.ensemble_details.prediction === 'damage_detected' || results.combined_summary.ensemble_details.prediction === 'damage') && results.combined_summary.ensemble_details.damage_types && results.combined_summary.ensemble_details.damage_types.length > 0) && (
                                  <div className="space-y-2">
                                    <span className="text-gray-400 text-sm font-medium">Detected Issues:</span>
                                    <div className="flex flex-wrap gap-2">
                                      {results.combined_summary.ensemble_details.damage_types.map((damageType, index) => (
                                        <span key={index} className="bg-red-900/30 text-red-300 px-3 py-1 rounded-full text-sm font-medium border border-red-700/50">
                                          {typeof damageType === 'string' ? damageType.charAt(0).toUpperCase() + damageType.slice(1) : damageType.type?.charAt(0).toUpperCase() + damageType.type?.slice(1)}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {/* Key Insights - Simplified reasoning */}
                                {results.combined_summary.ensemble_details.reasoning && (
                                  <div className="space-y-2">
                                    <span className="text-gray-400 text-sm font-medium">Key Insights:</span>
                                    <div className="bg-gray-800/50 rounded-lg p-3">
                                      <p className="text-gray-300 text-sm leading-relaxed">
                                        {typeof results.combined_summary.ensemble_details.reasoning === 'object' 
                                          ? JSON.stringify(results.combined_summary.ensemble_details.reasoning, null, 2)
                                          : results.combined_summary.ensemble_details.reasoning}
                                      </p>
                                    </div>
                                  </div>
                                )}

                                {/* Override Notice */}
                                {results.combined_summary.ensemble_details.override_applied && (
                                  <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                                    <div className="flex items-center">
                                      <div className="w-2 h-2 bg-blue-400 rounded-full mr-3"></div>
                                      <span className="text-blue-300 text-sm font-medium">Advanced model override applied for enhanced accuracy</span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Numeric Assessment */}
                          {results.combined_summary.ensemble_details?.numeric_assessment && (
                            <div className="mt-4 border-t border-gray-600 pt-4">
                              <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
                                <Sparkles className="w-4 h-4 mr-2 text-[#c1f21d]" />
                                Numeric Assessment (0-100)
                              </h4>
                              <div className="bg-[#1a1a1a] rounded-lg p-4 space-y-4">
                                {/* Overall Score with Progress Bar */}
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-gray-400 text-sm">Overall Score:</span>
                                    <span className={cn(
                                      "text-lg font-bold",
                                      results.combined_summary.ensemble_details.numeric_assessment.score >= 90 ? "text-green-400" :
                                      results.combined_summary.ensemble_details.numeric_assessment.score >= 70 ? "text-[#c1f21d]" :
                                      results.combined_summary.ensemble_details.numeric_assessment.score >= 50 ? "text-yellow-400" :
                                      results.combined_summary.ensemble_details.numeric_assessment.score >= 30 ? "text-orange-400" :
                                      "text-red-400"
                                    )}>
                                      {results.combined_summary.ensemble_details.numeric_assessment.score.toFixed(1)}/100
                                    </span>
                                  </div>
                                  <div className="w-full bg-gray-700 rounded-full h-3">
                                    <div 
                                      className={cn(
                                        "h-3 rounded-full transition-all duration-500",
                                        results.combined_summary.ensemble_details.numeric_assessment.score >= 90 ? "bg-green-400" :
                                        results.combined_summary.ensemble_details.numeric_assessment.score >= 70 ? "bg-[#c1f21d]" :
                                        results.combined_summary.ensemble_details.numeric_assessment.score >= 50 ? "bg-yellow-400" :
                                        results.combined_summary.ensemble_details.numeric_assessment.score >= 30 ? "bg-orange-400" :
                                        "bg-red-400"
                                      )}
                                      style={{ width: `${results.combined_summary.ensemble_details.numeric_assessment.score}%` }}
                                    />
                                  </div>
                                </div>

                                {/* Severity Level Badge */}
                                <div className="flex items-center justify-between">
                                  <span className="text-gray-400 text-sm">Severity Level:</span>
                                  <span className={cn(
                                    "px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide",
                                    results.combined_summary.ensemble_details.numeric_assessment.severity_level === 'excellent' ? "bg-green-900/30 text-green-300 border border-green-700/50" :
                                    results.combined_summary.ensemble_details.numeric_assessment.severity_level === 'good' ? "bg-[#c1f21d]/20 text-[#c1f21d] border border-[#c1f21d]/50" :
                                    results.combined_summary.ensemble_details.numeric_assessment.severity_level === 'fair' ? "bg-yellow-900/30 text-yellow-300 border border-yellow-700/50" :
                                    results.combined_summary.ensemble_details.numeric_assessment.severity_level === 'poor' ? "bg-orange-900/30 text-orange-300 border border-orange-700/50" :
                                    "bg-red-900/30 text-red-300 border border-red-700/50"
                                  )}>
                                    {results.combined_summary.ensemble_details.numeric_assessment.severity_level}
                                  </span>
                                </div>

                                {/* Key Metrics Grid */}
                                <div className="grid grid-cols-2 gap-3">
                                  <div className="bg-gray-800/50 rounded-lg p-3">
                                    <div className="text-xs text-gray-400 mb-1">Damage Impact</div>
                                    <div className="text-sm font-semibold text-white">
                                      {results.combined_summary.ensemble_details.numeric_assessment.damage_impact.toFixed(1)}
                                    </div>
                                  </div>
                                  <div className="bg-gray-800/50 rounded-lg p-3">
                                    <div className="text-xs text-gray-400 mb-1">Area Coverage</div>
                                    <div className="text-sm font-semibold text-white">
                                      {results.combined_summary.ensemble_details.numeric_assessment.area_coverage.toFixed(2)}%
                                    </div>
                                  </div>
                                  <div className="bg-gray-800/50 rounded-lg p-3">
                                    <div className="text-xs text-gray-400 mb-1">Confidence Factor</div>
                                    <div className="text-sm font-semibold text-white">
                                      {results.combined_summary.ensemble_details.numeric_assessment.confidence_factor.toFixed(2)}
                                    </div>
                                  </div>
                                  <div className="bg-gray-800/50 rounded-lg p-3">
                                    <div className="text-xs text-gray-400 mb-1">Total Damages</div>
                                    <div className="text-sm font-semibold text-white">
                                      {results.combined_summary.ensemble_details.numeric_assessment.total_damages}
                                    </div>
                                  </div>
                                </div>

                                {/* Damage Breakdown */}
                                {Object.keys(results.combined_summary.ensemble_details.numeric_assessment.damage_breakdown).length > 0 && (
                                  <div className="space-y-2">
                                    <span className="text-gray-400 text-sm">Damage Breakdown:</span>
                                    <div className="space-y-2">
                                      {Object.entries(results.combined_summary.ensemble_details.numeric_assessment.damage_breakdown).map(([damageType, details]) => (
                                        <div key={damageType} className="bg-gray-800/30 rounded-lg p-3">
                                          <div className="flex items-center justify-between mb-2">
                                            <span className="text-sm font-medium text-white capitalize">{damageType}</span>
                                            <span className={cn(
                                              "text-xs px-2 py-1 rounded",
                                              details.severity_level === 'minor' ? "bg-yellow-900/30 text-yellow-300" :
                                              details.severity_level === 'moderate' ? "bg-orange-900/30 text-orange-300" :
                                              "bg-red-900/30 text-red-300"
                                            )}>
                                              {details.severity_level}
                                            </span>
                                          </div>
                                          <div className="grid grid-cols-3 gap-2 text-xs">
                                            <div>
                                              <span className="text-gray-400">Count: </span>
                                              <span className="text-white">{details.count}</span>
                                            </div>
                                            <div>
                                              <span className="text-gray-400">Impact: </span>
                                              <span className="text-white">{details.total_impact.toFixed(1)}</span>
                                            </div>
                                            <div>
                                              <span className="text-gray-400">Avg Conf: </span>
                                              <span className="text-white">{(details.avg_confidence * 100).toFixed(0)}%</span>
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      

                      {/* Individual Model Results */}
                      <Card className="bg-[#2c2c2c] border-black">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-white">Detailed Analysis</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Main Model Results */}
                            {results.main_analysis && (
                              <div>
                                <h4 className="text-md font-semibold text-gray-300 mb-3">
                                  Main Model Analysis
                                </h4>
                                <div className="bg-[#1a1a1a] rounded-lg p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-gray-400 text-sm">Verdict:</span>
                                    <span className={cn(
                                      "font-semibold text-sm",
                                      results.main_analysis.integrity.damaged ? "text-red-400" : "text-[#c1f21d]"
                                    )}>
                                      {results.main_analysis.integrity.damaged ? 'Damaged' : 'Intact'}
                                    </span>
                                  </div>
                                  {results.main_analysis.heatmap.areas.length > 0 && (
                                    <>
                                      <div className="flex items-center justify-between">
                                        <span className="text-gray-400 text-sm">Detections:</span>
                                        <span className="text-white text-sm">{results.main_analysis.heatmap.areas.length}</span>
                                      </div>
                                      <div className="mt-2">
                                        <span className="text-gray-400 text-sm">Damage Types:</span>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                          {results.main_analysis.heatmap.areas.map((area, index) => (
                                            <span key={index} className="bg-[#c1f21d]/20 text-[#c1f21d] text-xs px-2 py-1 rounded">
                                              {area.description} {area.confidence ? `(${Math.round(area.confidence * 100)}%)` : ''}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Model 2 Results */}
                            {results.model2_analysis && (
                              <div>
                                <h4 className="text-md font-semibold text-blue-400 mb-3">
                                  Model 2 Analysis
                                </h4>
                                <div className="bg-[#1a1a1a] rounded-lg p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-gray-400 text-sm">Verdict:</span>
                                    <span className={cn(
                                      "font-semibold text-sm",
                                      results.model2_analysis.integrity.damaged ? "text-red-400" : "text-[#c1f21d]"
                                    )}>
                                      {results.model2_analysis.integrity.damaged ? 'Damaged' : 'Intact'}
                                    </span>
                                  </div>
                                  {results.model2_analysis.heatmap.areas.length > 0 && (
                                    <>
                                      <div className="flex items-center justify-between">
                                        <span className="text-gray-400 text-sm">Detections:</span>
                                        <span className="text-white text-sm">{results.model2_analysis.heatmap.areas.length}</span>
                                      </div>
                                      <div className="mt-2">
                                        <span className="text-gray-400 text-sm">Damage Types:</span>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                          {results.model2_analysis.heatmap.areas.map((area, index) => (
                                            <span key={index} className="bg-blue-900/30 text-blue-300 text-xs px-2 py-1 rounded">
                                              {area.description} {area.confidence ? `(${Math.round(area.confidence * 100)}%)` : ''}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Model 3 Results */}
                            {results.model3_analysis && (
                              <div>
                                <h4 className="text-md font-semibold text-purple-400 mb-3">
                                  Model 3 Analysis
                                </h4>
                                <div className="bg-[#1a1a1a] rounded-lg p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-gray-400 text-sm">Verdict:</span>
                                    <span className={cn(
                                      "font-semibold text-sm",
                                      results.model3_analysis.integrity.damaged ? "text-red-400" : "text-[#c1f21d]"
                                    )}>
                                      {results.model3_analysis.integrity.damaged ? 'Damaged' : 'Intact'}
                                    </span>
                                  </div>
                                  {results.model3_analysis.heatmap.areas.length > 0 && (
                                    <>
                                      <div className="flex items-center justify-between">
                                        <span className="text-gray-400 text-sm">Detections:</span>
                                        <span className="text-white text-sm">{results.model3_analysis.heatmap.areas.length}</span>
                                      </div>
                                      <div className="mt-2">
                                        <span className="text-gray-400 text-sm">Damage Types:</span>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                          {results.model3_analysis.heatmap.areas.map((area, index) => (
                                            <span key={index} className="bg-purple-900/30 text-purple-300 text-xs px-2 py-1 rounded">
                                              {area.description} {area.confidence ? `(${Math.round(area.confidence * 100)}%)` : ''}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Model 4 Results */}
                            {results.model4_analysis && (
                              <div>
                                <h4 className="text-md font-semibold text-orange-400 mb-3">
                                  Model 4 Analysis
                                </h4>
                                <div className="bg-[#1a1a1a] rounded-lg p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-gray-400 text-sm">Verdict:</span>
                                    <span className={cn(
                                      "font-semibold text-sm",
                                      results.model4_analysis.integrity.damaged ? "text-red-400" : "text-[#c1f21d]"
                                    )}>
                                      {results.model4_analysis.integrity.damaged ? 'Damaged' : 'Intact'}
                                    </span>
                                  </div>
                                  {results.model4_analysis.heatmap.areas.length > 0 && (
                                    <>
                                      <div className="flex items-center justify-between">
                                        <span className="text-gray-400 text-sm">Detections:</span>
                                        <span className="text-white text-sm">{results.model4_analysis.heatmap.areas.length}</span>
                                      </div>
                                      <div className="mt-2">
                                        <span className="text-gray-400 text-sm">Damage Types:</span>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                          {results.model4_analysis.heatmap.areas.map((area, index) => (
                                            <span key={index} className="bg-orange-900/30 text-orange-300 text-xs px-2 py-1 rounded">
                                              {area.description} {area.confidence ? `(${Math.round(area.confidence * 100)}%)` : ''}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>


                    </>
                  ) : (
                    // Single Model Results (fallback)
                    <>
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

                      {/* Model Information */}
                      {results.model_info && (
                        <Card className="bg-[#2c2c2c] border-black">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-white">Model Information</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="text-sm text-gray-400 space-y-1">
                              <p>Model: {results.model_info.name}</p>
                              <p>Type: {results.model_info.type}</p>
                              {results.model_info.fallback_used && (
                                <p className="text-yellow-400">⚠️ Fallback model used</p>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </>
                  )}

                  {/* AI-Repaired Preview */}
                      {results.gemini_analysis?.damage_detected && results.gemini_analysis?.repaired_image && (
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
                                src={`data:image/jpeg;base64,${results.gemini_analysis.repaired_image}`}
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