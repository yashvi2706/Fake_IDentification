import type { ScreeningResponse } from '../types/screening';
import { ShieldCheck, AlertTriangle, XCircle, CheckCircle2 } from 'lucide-react';

interface Props {
  results: ScreeningResponse;
  onReset: () => void;
}

export default function ScreeningResults({ results, onReset }: Props) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'text-[var(--sentinel-positive)]';
      case 'REVIEW': return 'text-[var(--sentinel-caution)]';
      case 'HIGH': return 'text-[var(--sentinel-critical)]';
      default: return 'text-[var(--sentinel-text)]';
    }
  };

  const getRiskBg = (level: string) => {
    switch (level) {
      case 'LOW': return 'bg-[var(--sentinel-positive-tint)]';
      case 'REVIEW': return 'bg-[var(--sentinel-caution-tint)]';
      case 'HIGH': return 'bg-[var(--sentinel-critical-tint)]';
      default: return 'bg-transparent';
    }
  };

  return (
    <div className="space-y-16 pb-20">
      {/* Header / Main Risk Score */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-8 pb-12 border-b border-[var(--sentinel-rule)]">
        <div>
          <div className="text-[var(--sentinel-text-muted)] font-mono text-sm mb-4">{results.screening_id}</div>
          <h1 className="sentinel-display text-5xl mb-2">Screening Results</h1>
          <p className="text-[var(--sentinel-text-muted)] capitalize">{results.document_type} • {results.risk.decision.replace('_', ' ')}</p>
        </div>
        
        <div className={`p-8 rounded-lg border border-[var(--sentinel-rule)] flex flex-col items-end ${getRiskBg(results.risk.level)}`}>
          <div className="sentinel-display text-7xl leading-none mb-2 font-serif">
            {results.risk.score} <span className="text-3xl text-[var(--sentinel-text-muted)]">/ 100</span>
          </div>
          <div className={`text-sm font-bold tracking-widest uppercase ${getRiskColor(results.risk.level)}`}>
            {results.risk.level} RISK
          </div>
        </div>
      </header>

      {/* Asymmetric composition */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-16">
        
        {/* Left Column (Primary Evidence) */}
        <div className="lg:col-span-8 space-y-16">
          
          {/* Risk Factors */}
          <section>
            <h2 className="sentinel-display text-3xl mb-6">Risk Factors</h2>
            {(results.risk.reasons ?? []).length > 0 ? (
              <ul className="space-y-4">
                {(results.risk.reasons ?? []).map((reason, idx) => (
                  <li key={idx} className="flex items-start gap-3 p-4 bg-[var(--sentinel-critical-tint)] border border-[var(--sentinel-critical)] border-opacity-20 rounded-md">
                    <AlertTriangle className="w-5 h-5 text-[var(--sentinel-critical)] shrink-0 mt-0.5" />
                    <span className="text-[var(--sentinel-text)]">{reason.message}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center gap-3 p-4 border border-[var(--sentinel-rule)] rounded-md text-[var(--sentinel-text-muted)]">
                <ShieldCheck className="w-5 h-5 text-[var(--sentinel-positive)]" />
                No significant risk indicators detected.
              </div>
            )}
          </section>

          {/* OCR Results */}
          <section>
            <h2 className="sentinel-display text-3xl mb-6">Extracted Data</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
              {Object.entries(results.ocr ?? {}).map(([key, value]) => {
                if (key === 'confidence' || key === 'raw_text' || key === 'error') return null;
                const display = (value === null || value === undefined || value === '') ? 'Not extracted' : String(value);
                return (
                  <div key={key} className="border-b border-[var(--sentinel-rule)] pb-2">
                    <div className="text-xs text-[var(--sentinel-text-muted)] uppercase tracking-wider mb-1">{key.replace(/_/g, ' ')}</div>
                    <div className={`text-lg font-medium ${display === 'Not extracted' ? 'text-[var(--sentinel-text-muted)] italic text-base' : 'text-[var(--sentinel-text)]'}`}>{display}</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-6 flex items-center gap-2 text-sm text-[var(--sentinel-text-muted)]">
              <span className="w-2 h-2 rounded-full bg-[var(--sentinel-positive)]"></span>
              Extraction Confidence: {((results.ocr?.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </section>

          {/* Document Validation */}
          <section>
            <h2 className="sentinel-display text-3xl mb-6">Document Validation</h2>
            <div className="space-y-4">
              {(results.validation?.checks ?? []).map((check, idx) => (
                <div key={idx} className="flex items-start gap-4 py-3 border-b border-[var(--sentinel-rule)] last:border-0">
                  {check.status === 'pass' && <CheckCircle2 className="w-5 h-5 text-[var(--sentinel-positive)] shrink-0" />}
                  {check.status === 'warning' && <AlertTriangle className="w-5 h-5 text-[var(--sentinel-caution)] shrink-0" />}
                  {check.status === 'fail' && <XCircle className="w-5 h-5 text-[var(--sentinel-critical)] shrink-0" />}
                  
                  <div>
                    <div className="font-medium text-[var(--sentinel-text)]">{check.name}</div>
                    <div className="text-sm text-[var(--sentinel-text-muted)] mt-1">{check.message}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right Column (Secondary / Meta) */}
        <div className="lg:col-span-4 space-y-12 lg:border-l lg:border-[var(--sentinel-rule)] lg:pl-12">
          
          {/* Image Quality */}
          {results.quality && (
            <section>
              <h2 className="sentinel-display text-2xl mb-6 border-b border-[var(--sentinel-rule)] pb-4">Image Quality</h2>
              <div className="space-y-4">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-[var(--sentinel-text-muted)]">Acceptable</span>
                  <span className={results.quality.acceptable ? 'text-[var(--sentinel-positive)]' : 'text-[var(--sentinel-critical)]'}>
                    {results.quality.acceptable ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-[var(--sentinel-text-muted)]">Blur/Sharpness Score</span>
                  <span className="text-[var(--sentinel-text)]">{results.quality.blur_score.toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-[var(--sentinel-text-muted)]">Glare Detected</span>
                  <span className={results.quality.glare_score > 5 ? 'text-[var(--sentinel-caution)]' : 'text-[var(--sentinel-text)]'}>
                    {results.quality.glare_score.toFixed(1)}%
                  </span>
                </div>
                {(results.quality.issues ?? []).length > 0 && (
                  <div className="pt-4 mt-4 border-t border-[var(--sentinel-rule)]">
                    <div className="text-sm text-[var(--sentinel-text-muted)] mb-2">Issues:</div>
                    <ul className="list-disc pl-4 text-sm text-[var(--sentinel-critical)] space-y-1">
                      {(results.quality.issues ?? []).map((issue, i) => (
                        <li key={i}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Template Analysis */}
          {results.template && results.template.available && (
            <section>
              <h2 className="sentinel-display text-2xl mb-6 border-b border-[var(--sentinel-rule)] pb-4">Template Validation</h2>
              <div className="space-y-4">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-[var(--sentinel-text-muted)]">Layout Consistency</span>
                  <span className={results.template.layout_consistent ? 'text-[var(--sentinel-positive)]' : 'text-[var(--sentinel-critical)]'}>
                    {results.template.layout_consistent ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                {(results.template.checks ?? []).length > 0 && (
                  <div className="pt-4 mt-4 border-t border-[var(--sentinel-rule)] space-y-3">
                    {(results.template.checks ?? []).map((check, idx) => (
                       <div key={idx} className="text-sm">
                          <div className={`font-medium ${check.status === 'pass' ? 'text-[var(--sentinel-positive)]' : check.status === 'warn' ? 'text-[var(--sentinel-caution)]' : 'text-[var(--sentinel-critical)]'}`}>
                            {check.name}
                          </div>
                          <div className="text-[var(--sentinel-text-muted)] mt-0.5 text-xs">{check.message}</div>
                       </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Tampering Detection */}
          <section>
            <h2 className="sentinel-display text-2xl mb-6 border-b border-[var(--sentinel-rule)] pb-4">Tampering Analysis</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-[var(--sentinel-text-muted)]">Suspicious</span>
                <span className={results.tampering.suspicious ? 'text-[var(--sentinel-critical)]' : 'text-[var(--sentinel-positive)]'}>
                  {results.tampering.suspicious ? 'TRUE' : 'FALSE'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[var(--sentinel-text-muted)]">Photo Replacement</span>
                <span className={results.tampering.photo_replacement ? 'text-[var(--sentinel-critical)]' : 'text-[var(--sentinel-text)]'}>
                  {results.tampering.photo_replacement ? 'TRUE' : 'FALSE'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-[var(--sentinel-text-muted)]">Text Manipulation</span>
                <span className={results.tampering.text_manipulation ? 'text-[var(--sentinel-critical)]' : 'text-[var(--sentinel-text)]'}>
                  {results.tampering.text_manipulation ? 'TRUE' : 'FALSE'}
                </span>
              </div>
              
              {(results.tampering?.indicators ?? []).length > 0 && (
                <div className="pt-4 mt-4 border-t border-[var(--sentinel-rule)]">
                  <div className="text-sm text-[var(--sentinel-text-muted)] mb-2">Indicators:</div>
                  <ul className="list-disc pl-4 text-sm text-[var(--sentinel-critical)] space-y-1">
                    {(results.tampering?.indicators ?? []).map((ind, i) => (
                      <li key={i}>{ind}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </section>

          {/* Face Verification */}
          <section>
            <h2 className="sentinel-display text-2xl mb-6 border-b border-[var(--sentinel-rule)] pb-4">Biometric Match</h2>
            {results.face_verification.available ? (
              <div className="space-y-4">
                <div className="text-4xl sentinel-display mb-4 text-[var(--sentinel-text)]">
                  {results.face_verification.similarity != null
                    ? `${(results.face_verification.similarity * 100).toFixed(0)}%`
                    : 'Unavailable'}
                </div>
                <div className="text-sm">
                  {results.face_verification.match ? (
                    <span className="text-[var(--sentinel-positive)] flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> {results.face_verification.message}</span>
                  ) : (
                    <span className="text-[var(--sentinel-critical)] flex items-center gap-2"><XCircle className="w-4 h-4" /> {results.face_verification.message}</span>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-sm text-[var(--sentinel-text-muted)] italic">
                Face verification not available or not requested.
              </div>
            )}
          </section>
          
          <div className="pt-8">
            <button onClick={onReset} className="sentinel-secondary-action w-full">
              Process New Document
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
