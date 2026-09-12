import React from 'react'
import { AlertTriangle, CheckCircle, ShieldAlert, UserCheck } from 'lucide-react'

export default function Dashboard() {
  // Mock Data (matches the expected Risk Fusion output shape)
  const results = {
    risk_score: 82,
    risk_band: 'High',
    modules: {
      ocr: { status: 'success', confidence: 0.94 },
      validation: { status: 'failed', issues: ['MRZ checksum mismatch'] },
      tamper: { 
        status: 'warning', 
        score: 0.76, 
        flags: ['Possible copy-move forgery on photo', 'Metadata anomaly'] 
      },
      face: { status: 'success', match_score: 0.88, match: true }
    },
    extracted_data: {
      document_type: 'Passport',
      document_number: 'A1234567',
      name: 'JOHN DOE',
      nationality: 'GBR',
      dob: '1985-04-12'
    }
  }

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Screening Results</h2>
          <p className="text-muted-foreground mt-1">Report ID: #9824-AX (Simulated Data)</p>
        </div>
        <div className={`px-6 py-3 rounded-full flex items-center gap-3 font-bold shadow-sm border ${
          results.risk_band === 'High' ? 'bg-destructive/10 text-destructive border-destructive/20' : 
          results.risk_band === 'Medium' ? 'bg-orange-500/10 text-orange-600 border-orange-500/20' :
          'bg-green-500/10 text-green-600 border-green-500/20'
        }`}>
          {results.risk_band === 'High' && <AlertTriangle className="w-6 h-6" />}
          <span className="text-xl">RISK: {results.risk_band} ({results.risk_score}/100)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Extracted Data Card */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm col-span-1">
          <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
            <UserCheck className="w-5 h-5 text-primary" />
            Extracted Identity
          </h3>
          <div className="space-y-4">
            {Object.entries(results.extracted_data).map(([k, v]) => (
              <div key={k}>
                <div className="text-xs text-muted-foreground uppercase tracking-wider">{k.replace('_', ' ')}</div>
                <div className="font-medium text-foreground">{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Module Breakdown */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm col-span-1 lg:col-span-2 space-y-4">
          <h3 className="font-semibold text-lg mb-4">Module Analysis</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg border border-border bg-secondary flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium">OCR Extraction</h4>
                <p className="text-sm text-muted-foreground">Confidence: {results.modules.ocr.confidence * 100}%</p>
              </div>
            </div>

            <div className="p-4 rounded-lg border border-destructive/30 bg-destructive/5 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium text-destructive">Document Validation</h4>
                <ul className="text-sm text-destructive/80 list-disc list-inside mt-1">
                  {results.modules.validation.issues.map((i, idx) => <li key={idx}>{i}</li>)}
                </ul>
              </div>
            </div>

            <div className="p-4 rounded-lg border border-orange-500/30 bg-orange-500/5 flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium text-orange-600">Tampering Detection</h4>
                <p className="text-sm text-orange-600/80">Anomaly Score: {results.modules.tamper.score * 100}%</p>
                <ul className="text-sm text-orange-600/80 list-disc list-inside mt-1">
                  {results.modules.tamper.flags.map((i, idx) => <li key={idx}>{i}</li>)}
                </ul>
              </div>
            </div>

            <div className="p-4 rounded-lg border border-border bg-secondary flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium">Face Match</h4>
                <p className="text-sm text-muted-foreground">Similarity: {results.modules.face.match_score * 100}%</p>
              </div>
            </div>
          </div>

        </div>

      </div>

      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <h3 className="font-semibold text-lg mb-4">Explainability & Evidence (Mock)</h3>
        <div className="aspect-[21/9] bg-secondary border border-dashed border-border rounded-lg flex items-center justify-center text-muted-foreground">
          [ Document Image with Heatmap Overlay Placeholder ]
        </div>
      </div>
    </div>
  )
}
