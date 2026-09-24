import { useState } from 'react';
import UploadPanel from '../components/UploadPanel';
import AnalysisProgress from '../components/AnalysisProgress';
import ScreeningResults from '../components/ScreeningResults';
import { analyzeDocument } from '../services/api';
import type { ScreeningResponse } from '../types/screening';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

type Step = 'upload' | 'analyzing' | 'results';

export default function ScreeningFlow() {
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  const [results, setResults] = useState<ScreeningResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleAnalyze = async (documentFile: File, faceImageFile: File | null, documentType: string) => {
    setCurrentStep('analyzing');
    setError(null);
    try {
      const data = await analyzeDocument(documentFile, faceImageFile, documentType);
      setResults(data);
      setCurrentStep('results');
    } catch (err) {
      setError('Analysis failed. Please try again.');
      setCurrentStep('upload');
    }
  };

  const handleReset = () => {
    setResults(null);
    setCurrentStep('upload');
  };

  return (
    <div className="max-w-4xl mx-auto">
      {currentStep === 'upload' && (
        <div className="mb-8">
          <button 
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 text-sm text-[var(--sentinel-text-muted)] hover:text-[var(--sentinel-text)] transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </button>
          <h1 className="sentinel-display text-4xl mb-2">New Screening</h1>
          <p className="text-[var(--sentinel-text-muted)]">Upload documents to verify identity and check for tampering.</p>
        </div>
      )}

      {currentStep === 'upload' && <UploadPanel onAnalyze={handleAnalyze} />}
      {currentStep === 'analyzing' && <AnalysisProgress />}
      {currentStep === 'results' && results && (
        <ScreeningResults results={results} onReset={handleReset} />
      )}
      
      {error && (
        <div className="mt-4 p-4 border border-[var(--sentinel-critical)] bg-[var(--sentinel-critical-tint)] text-[var(--sentinel-critical)] rounded-md">
          {error}
        </div>
      )}
    </div>
  );
}
