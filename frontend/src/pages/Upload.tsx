import React, { useCallback, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { Upload as UploadIcon, Camera, X } from 'lucide-react'

export default function Upload() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const selected = acceptedFiles[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxFiles: 1
  })

  const startCamera = async () => {
    try {
      setIsCameraActive(true)
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error("Error accessing camera:", err)
      setIsCameraActive(false)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setIsCameraActive(false)
  }

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0)
        canvas.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], "camera-capture.jpg", { type: "image/jpeg" })
            setFile(capturedFile)
            setPreview(URL.createObjectURL(capturedFile))
            stopCamera()
          }
        }, 'image/jpeg')
      }
    }
  }

  const clearSelection = () => {
    setFile(null)
    setPreview(null)
  }

  const handleUpload = () => {
    setIsProcessing(true)
    // Simulate API call to Gateway
    setTimeout(() => {
      navigate('/dashboard')
    }, 2000)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-foreground">Document Capture</h2>
        <p className="text-muted-foreground mt-2">Upload or capture an identity document for screening.</p>
      </div>

      {!preview && !isCameraActive && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary bg-primary/5' : 'border-border bg-card hover:border-primary/50 hover:bg-secondary'
            }`}
          >
            <input {...getInputProps()} />
            <UploadIcon className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="font-semibold text-lg text-foreground">Upload File</h3>
            <p className="text-sm text-muted-foreground text-center mt-2">
              Drag & drop a passport, visa, or ID card image here, or click to browse.
            </p>
          </div>

          <div 
            onClick={startCamera}
            className="border-2 border-border rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer bg-card hover:border-primary/50 hover:bg-secondary transition-colors"
          >
            <Camera className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="font-semibold text-lg text-foreground">Live Capture</h3>
            <p className="text-sm text-muted-foreground text-center mt-2">
              Use your device camera to scan a document directly.
            </p>
          </div>
        </div>
      )}

      {isCameraActive && (
        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm">
          <div className="p-4 border-b border-border flex justify-between items-center bg-muted/30">
            <h3 className="font-semibold">Live Camera</h3>
            <button onClick={stopCamera} className="p-1 hover:bg-destructive/10 hover:text-destructive rounded-md transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="relative bg-black aspect-video flex items-center justify-center">
            <video ref={videoRef} autoPlay playsInline className="max-h-full" />
          </div>
          <div className="p-4 text-center">
            <button 
              onClick={captureImage}
              className="bg-primary text-primary-foreground px-6 py-2 rounded-full font-medium shadow-md hover:bg-primary/90 transition-transform active:scale-95"
            >
              Capture Frame
            </button>
          </div>
        </div>
      )}

      {preview && (
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="font-semibold text-lg text-foreground">Selected Document</h3>
              <p className="text-sm text-muted-foreground">{file?.name}</p>
            </div>
            <button onClick={clearSelection} disabled={isProcessing} className="p-2 text-muted-foreground hover:bg-secondary rounded-md disabled:opacity-50">
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <div className="flex flex-col md:flex-row gap-6 items-start">
            <div className="w-full md:w-1/2 aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center border border-border">
              <img src={preview} alt="Document Preview" className="max-w-full max-h-full object-contain" />
            </div>
            <div className="w-full md:w-1/2 space-y-4">
              <div className="bg-secondary p-4 rounded-lg border border-border">
                <h4 className="font-medium mb-2 text-sm text-muted-foreground">Document Type</h4>
                <select className="w-full bg-background border border-border rounded-md px-3 py-2 outline-none focus:border-primary">
                  <option>Auto-Detect</option>
                  <option>Passport</option>
                  <option>Visa</option>
                  <option>National ID</option>
                </select>
              </div>
              <button 
                onClick={handleUpload}
                disabled={isProcessing}
                className="w-full bg-primary text-primary-foreground py-3 rounded-lg font-medium shadow-md hover:bg-primary/90 transition-all active:scale-[0.98] disabled:opacity-70 disabled:active:scale-100 flex justify-center items-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Processing Pipeline...
                  </>
                ) : 'Run AI Screening'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
