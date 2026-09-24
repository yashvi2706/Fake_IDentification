import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

interface UploadPanelProps {
  onAnalyze: (document: File, face: File | null, type: string) => void;
}

const documentTypes = [
  { id: 'passport', label: 'Passport' },
  { id: 'visa', label: 'Visa' },
  { id: 'national_id', label: 'National ID' },
  { id: 'driving_license', label: 'Driving License' },
  { id: 'permit', label: 'Permit' },
];

export default function UploadPanel({ onAnalyze }: UploadPanelProps) {
  const [documentType, setDocumentType] = useState('passport');
  const [docFile, setDocFile] = useState<File | null>(null);
  const [docPreview, setDocPreview] = useState<string | null>(null);
  
  const [faceFile, setFaceFile] = useState<File | null>(null);
  const [facePreview, setFacePreview] = useState<string | null>(null);

  const onDropDoc = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.[0]) {
      setDocFile(acceptedFiles[0]);
      setDocPreview(URL.createObjectURL(acceptedFiles[0]));
    }
  }, []);

  const onDropFace = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.[0]) {
      setFaceFile(acceptedFiles[0]);
      setFacePreview(URL.createObjectURL(acceptedFiles[0]));
    }
  }, []);

  const { getRootProps: getDocRootProps, getInputProps: getDocInputProps, isDragActive: isDocDrag } = useDropzone({
    onDrop: onDropDoc,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1
  });

  const { getRootProps: getFaceRootProps, getInputProps: getFaceInputProps, isDragActive: isFaceDrag } = useDropzone({
    onDrop: onDropFace,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1
  });

  const handleAnalyze = () => {
    if (docFile) {
      onAnalyze(docFile, faceFile, documentType);
    }
  };

  return (
    <div className="space-y-12">
      <section className="space-y-6 border-b border-[var(--sentinel-rule)] pb-12">
        <h2 className="sentinel-display text-2xl text-[var(--sentinel-text)]">Select Document Type</h2>
        <div className="flex flex-wrap gap-4">
          {documentTypes.map((type) => (
            <label key={type.id} className="cursor-pointer">
              <input
                type="radio"
                name="documentType"
                value={type.id}
                checked={documentType === type.id}
                onChange={(e) => setDocumentType(e.target.value)}
                className="sr-only peer"
              />
              <div className={`px-6 py-3 border rounded-md text-sm font-medium transition-colors
                ${documentType === type.id 
                  ? 'border-[var(--sentinel-accent)] text-[var(--sentinel-accent)] bg-[var(--sentinel-surface-raised)]' 
                  : 'border-[var(--sentinel-rule)] text-[var(--sentinel-text-muted)] hover:border-[var(--sentinel-text-muted)]'
                }`}
              >
                {type.label}
              </div>
            </label>
          ))}
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        <section className="space-y-4">
          <div>
            <h2 className="sentinel-display text-2xl">Identity Document <span className="text-[var(--sentinel-critical)]">*</span></h2>
            <p className="text-sm text-[var(--sentinel-text-muted)] mt-1">Upload the front of the document</p>
          </div>
          
          {!docPreview ? (
            <div 
              {...getDocRootProps()} 
              className={`border border-dashed border-[var(--sentinel-rule)] rounded-[var(--sentinel-radius-image)] p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-colors min-h-[320px] bg-[var(--sentinel-surface)] hover:bg-[var(--sentinel-surface-raised)] ${isDocDrag ? 'border-[var(--sentinel-accent)]' : ''}`}
            >
              <input {...getDocInputProps()} />
              <Upload className="w-8 h-8 text-[var(--sentinel-text-muted)] mb-4" />
              <p className="text-sm font-medium mb-1">Drag and drop document image here</p>
              <p className="text-xs text-[var(--sentinel-text-muted)]">JPEG or PNG format</p>
            </div>
          ) : (
            <div className="relative rounded-[var(--sentinel-radius-image)] overflow-hidden bg-[var(--sentinel-image-surface)] border border-[var(--sentinel-rule)] min-h-[320px] flex items-center justify-center">
              <img src={docPreview} alt="Document Preview" className="max-h-[320px] object-contain p-4" />
              <button 
                onClick={() => { setDocFile(null); setDocPreview(null); }}
                className="absolute top-4 right-4 p-2 bg-[var(--sentinel-canvas-deep)] rounded-full hover:bg-[var(--sentinel-surface-raised)] transition-colors border border-[var(--sentinel-rule)]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </section>

        <section className="space-y-4">
          <div>
            <h2 className="sentinel-display text-2xl">Live Portrait</h2>
            <p className="text-sm text-[var(--sentinel-text-muted)] mt-1">Optional for face verification</p>
          </div>
          
          {!facePreview ? (
            <div 
              {...getFaceRootProps()} 
              className={`border border-dashed border-[var(--sentinel-rule)] rounded-[var(--sentinel-radius-image)] p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-colors min-h-[320px] bg-[var(--sentinel-portrait-surface)] hover:bg-[var(--sentinel-surface-raised)] ${isFaceDrag ? 'border-[var(--sentinel-accent)]' : ''}`}
            >
              <input {...getFaceInputProps()} />
              <ImageIcon className="w-8 h-8 text-[var(--sentinel-text-muted)] mb-4" />
              <p className="text-sm font-medium mb-1">Drag and drop face photo here</p>
              <p className="text-xs text-[var(--sentinel-text-muted)]">JPEG or PNG format</p>
            </div>
          ) : (
            <div className="relative rounded-[var(--sentinel-radius-image)] overflow-hidden bg-[var(--sentinel-portrait-surface)] border border-[var(--sentinel-rule)] min-h-[320px] flex items-center justify-center">
              <img src={facePreview} alt="Face Preview" className="max-h-[320px] object-cover h-full w-full" />
              <button 
                onClick={() => { setFaceFile(null); setFacePreview(null); }}
                className="absolute top-4 right-4 p-2 bg-[var(--sentinel-canvas-deep)] rounded-full hover:bg-[var(--sentinel-surface-raised)] transition-colors border border-[var(--sentinel-rule)]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </section>
      </div>

      <div className="flex justify-end pt-8 border-t border-[var(--sentinel-rule)]">
        <button 
          onClick={handleAnalyze} 
          disabled={!docFile}
          className="sentinel-primary-action text-sm px-8 py-3"
        >
          Analyze Document
        </button>
      </div>
    </div>
  );
}
