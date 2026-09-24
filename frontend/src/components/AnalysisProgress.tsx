import { useState, useEffect } from 'react';
import { Loader2, CheckCircle2 } from 'lucide-react';

const stages = [
  "Extracting document information...",
  "Validating document structure...",
  "Checking tampering indicators...",
  "Verifying identity...",
  "Calculating risk score..."
];

export default function AnalysisProgress() {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStage(prev => {
        if (prev < stages.length - 1) return prev + 1;
        clearInterval(timer);
        return prev;
      });
    }, 500); // Progress every 500ms for a total of ~2.5s matching mockApi

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="relative w-32 h-32 mb-12">
        <div className="absolute inset-0 rounded-full border-2 border-[var(--sentinel-rule)]"></div>
        <div 
          className="absolute inset-0 rounded-full border-2 border-[var(--sentinel-accent)] border-t-transparent animate-spin"
          style={{ animationDuration: '1.5s' }}
        ></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-[var(--sentinel-accent)] animate-pulse" />
        </div>
      </div>

      <div className="w-full max-w-md space-y-4">
        {stages.map((stage, index) => (
          <div 
            key={stage} 
            className={`flex items-center gap-4 p-4 border border-[var(--sentinel-rule)] rounded-md transition-all duration-300
              ${index === currentStage ? 'bg-[var(--sentinel-surface-raised)] shadow-[var(--sentinel-document-shadow)] scale-105' : 'bg-[var(--sentinel-surface)] opacity-50'}
              ${index < currentStage ? 'opacity-40' : ''}
              ${index > currentStage ? 'opacity-10 scale-95' : ''}
            `}
          >
            {index < currentStage ? (
              <CheckCircle2 className="w-5 h-5 text-[var(--sentinel-positive)]" />
            ) : index === currentStage ? (
              <Loader2 className="w-5 h-5 text-[var(--sentinel-accent)] animate-spin" />
            ) : (
              <div className="w-5 h-5 rounded-full border border-[var(--sentinel-rule)]"></div>
            )}
            <span className={`text-sm font-medium ${index === currentStage ? 'text-[var(--sentinel-text)]' : 'text-[var(--sentinel-text-muted)]'}`}>
              {stage}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
