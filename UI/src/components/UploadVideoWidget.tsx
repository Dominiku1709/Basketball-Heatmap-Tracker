import React, { useRef, useState } from 'react';
import { Upload, Film, Play, CheckCircle2, X, RefreshCw, AlertTriangle, Cpu } from 'lucide-react';
import { PlayerModelOption } from '../api';

interface UploadVideoWidgetProps {
  selectedFile: File | null;
  videoUrl: string | null;
  onFileSelect: (file: File) => void;
  onClearFile: () => void;
  onProcessVideo: () => void;
  isProcessing: boolean;
  processProgress: number;
  processError?: string | null;
  models: PlayerModelOption[];
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

export const UploadVideoWidget: React.FC<UploadVideoWidgetProps> = ({
  selectedFile,
  videoUrl,
  onFileSelect,
  onClearFile,
  onProcessVideo,
  isProcessing,
  processProgress,
  processError,
  models,
  selectedModel,
  onSelectModel
}) => {
  const [dragActive, setDragActive] = useState(false);
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
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        onFileSelect(file);
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div id="upload-video-widget" className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800 mb-4">
        <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
          <Upload className="h-4 w-4 text-emerald-400" />
          <span>Upload Video Widget</span>
        </h2>
        {selectedFile && (
          <span className="text-xs text-emerald-400 flex items-center gap-1 font-mono">
            <CheckCircle2 className="h-3.5 w-3.5" />
            File Loaded
          </span>
        )}
      </div>

      <div className="space-y-4">
        {/* Dropzone */}
        {!selectedFile ? (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center p-8 rounded-lg border-2 border-dashed cursor-pointer transition-colors text-center ${
              dragActive
                ? 'border-emerald-500 bg-emerald-500/10'
                : 'border-neutral-700 bg-neutral-950/50 hover:border-neutral-600 hover:bg-neutral-950'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/mkv"
              onChange={handleInputChange}
              className="hidden"
              id="upload-file-input"
            />
            <div className="h-10 w-10 rounded-full bg-neutral-800 flex items-center justify-center text-neutral-300 mb-3">
              <Film className="h-5 w-5" />
            </div>
            <p className="text-sm font-medium text-neutral-200">
              Drag & drop a match or training video here, or <span className="text-emerald-400 underline">browse</span>
            </p>
            <p className="text-xs text-neutral-400 mt-1">
              Supports MP4, WebM, MOV video files
            </p>
          </div>
        ) : (
          <div className="flex items-center justify-between p-3.5 rounded-lg border border-neutral-800 bg-neutral-950">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="h-9 w-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0">
                <Film className="h-4 w-4" />
              </div>
              <div className="truncate">
                <p className="text-sm font-medium text-white truncate">{selectedFile.name}</p>
                <p className="text-xs text-neutral-400 font-mono">
                  {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Video'}
                </p>
              </div>
            </div>
            <button
              onClick={onClearFile}
              disabled={isProcessing}
              id="clear-video-btn"
              title="Remove file"
              className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors ml-2"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Player Detector Model Picker */}
        {models.length > 0 && (
          <div>
            <label className="flex items-center gap-1.5 text-xs font-medium text-neutral-400 mb-2">
              <Cpu className="h-3.5 w-3.5" />
              Player Detector Model
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {models.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => onSelectModel(m.id)}
                  disabled={isProcessing}
                  title={m.description}
                  className={`text-left rounded-lg border px-3 py-2 transition-colors ${
                    selectedModel === m.id
                      ? 'border-emerald-500 bg-emerald-500/10'
                      : 'border-neutral-800 bg-neutral-950 hover:border-neutral-700'
                  } ${isProcessing ? 'cursor-not-allowed opacity-60' : ''}`}
                >
                  <div className={`text-xs font-semibold ${selectedModel === m.id ? 'text-emerald-400' : 'text-neutral-200'}`}>
                    {m.label}
                  </div>
                  <div className="text-[10px] text-neutral-500 mt-0.5 leading-snug">{m.description}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Process Video Button */}
        <div className="pt-2">
          <button
            onClick={onProcessVideo}
            disabled={!videoUrl || isProcessing}
            id="process-video-button"
            className={`w-full flex items-center justify-center gap-2 rounded-lg py-2 px-4 text-xs font-semibold transition-all ${
              !videoUrl || isProcessing
                ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed border border-neutral-800'
                : 'bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold shadow-sm'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                <span>Processing Tracking ({processProgress}%)...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Process Video</span>
              </>
            )}
          </button>
        </div>

        {/* Processing Progress Bar */}
        {isProcessing && (
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-xs text-neutral-400">
              <span>Running player detection, tracking &amp; heatmap generation on the server...</span>
              <span className="font-mono text-emerald-400">{processProgress}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-200"
                style={{ width: `${processProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error banner (upload failed, backend unreachable, analysis error) */}
        {processError && (
          <div className="flex items-start gap-2 p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <span>{processError}</span>
          </div>
        )}
      </div>
    </div>
  );
};
