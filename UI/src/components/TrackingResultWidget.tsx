import React, { useRef, useState } from 'react';
import {
  Play, Pause, RotateCcw, Download, Camera, Layers,
  Flame, Film
} from 'lucide-react';

interface TrackingResultWidgetProps {
  hasResult: boolean;
  /** Backend-annotated output video — already has player boxes, IDs and the
   * tactical-view minimap burned into every frame by the Python pipeline,
   * so this widget just plays it directly (no client-side overlay drawing). */
  resultVideoUrl?: string;
  team1HeatmapUrl?: string;
  team2HeatmapUrl?: string;
  fileName?: string;
  stats?: {
    trackedEntitiesCount: number;
  };
  durationSec?: number;
}

export const TrackingResultWidget: React.FC<TrackingResultWidgetProps> = ({
  hasResult,
  resultVideoUrl,
  team1HeatmapUrl,
  team2HeatmapUrl,
  fileName,
  stats,
  durationSec = 15
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationSec);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch((e) => console.warn('Play error:', e));
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || durationSec);
    }
  };

  const downloadUrl = (url: string, filename: string) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.click();
  };

  const baseName = fileName ? fileName.replace(/\.[^/.]+$/, '') : 'basketball';

  // Snapshot the current frame of the real result video (already has
  // boxes/IDs/minimap baked in — no separate overlay to composite).
  const handleCaptureFrame = () => {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/png');
      downloadUrl(dataUrl, `${baseName}_snapshot.png`);
    } catch (e) {
      console.warn('Frame capture failed', e);
    }
  };

  return (
    <div id="tracking-result-widget" className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-sm space-y-5">
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-400" />
            <span>Tracking Result</span>
          </h2>
          {hasResult && (
            <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-mono text-emerald-400 border border-emerald-500/20">
              Completed
            </span>
          )}
        </div>
        {hasResult && (
          <button
            onClick={handleCaptureFrame}
            id="capture-frame-btn"
            className="flex items-center gap-1.5 rounded-lg border border-neutral-700 bg-neutral-950 px-2.5 py-1 text-xs text-neutral-300 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            <Camera className="h-3.5 w-3.5 text-emerald-400" />
            <span>Snapshot Frame</span>
          </button>
        )}
      </div>

      {!hasResult || !resultVideoUrl ? (
        <div className="flex flex-col items-center justify-center p-12 text-center rounded-lg border border-neutral-800 bg-neutral-950/40">
          <Film className="h-10 w-10 text-neutral-600 mb-3" />
          <h3 className="text-sm font-semibold text-white">No tracking results yet</h3>
          <p className="text-xs text-neutral-400 max-w-sm mt-1">
            Upload a video above and click &quot;Process Video&quot;. The annotated output video and per-team heatmaps from the analysis pipeline will appear here.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Summary Stats Row — real numbers from run_info.json, not simulated */}
          {stats && (
            <div className="grid grid-cols-1 gap-3 p-3 rounded-lg border border-neutral-800 bg-neutral-950 text-xs w-fit">
              <div>
                <span className="text-neutral-400 block text-[11px]">Players Detected</span>
                <span className="font-mono text-base font-bold text-white">{stats.trackedEntitiesCount}</span>
              </div>
            </div>
          )}

          {/* Result Video — annotated by the backend, played as-is */}
          <div className="space-y-3">
            <span className="text-xs font-semibold text-neutral-300 flex items-center gap-1.5 uppercase tracking-wide">
              <Film className="h-3.5 w-3.5 text-emerald-400" />
              Annotated Output Video
            </span>

            <div className="relative w-full aspect-video bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800 flex items-center justify-center select-none">
              <video
                ref={videoRef}
                src={resultVideoUrl}
                playsInline
                muted
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
                className="w-full h-full object-contain"
              />

              {!isPlaying && (
                <button
                  onClick={togglePlay}
                  id="result-play-overlay-btn"
                  className="absolute z-20 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-500/90 hover:bg-emerald-400 text-neutral-950 shadow-lg transition-transform active:scale-95"
                >
                  <Play className="h-6 w-6 fill-current ml-0.5" />
                </button>
              )}
            </div>

            <div className="space-y-1">
              <input
                type="range"
                min="0"
                max={duration || 1}
                step="0.05"
                value={currentTime}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setCurrentTime(val);
                  if (videoRef.current) videoRef.current.currentTime = val;
                }}
                className="w-full h-1.5 rounded-lg bg-neutral-800 accent-emerald-500 cursor-pointer"
              />
              <div className="flex items-center justify-between text-[11px] font-mono text-neutral-400">
                <span>{currentTime.toFixed(1)}s</span>
                <span>{duration.toFixed(1)}s</span>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={togglePlay}
                id="result-play-btn"
                className="h-8 w-8 flex items-center justify-center rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold transition-colors"
              >
                {isPlaying ? <Pause className="h-3.5 w-3.5 fill-current" /> : <Play className="h-3.5 w-3.5 fill-current ml-0.5" />}
              </button>
              <button
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.currentTime = 0;
                    setCurrentTime(0);
                  }
                }}
                title="Restart"
                className="h-8 w-8 flex items-center justify-center rounded-lg border border-neutral-800 bg-neutral-950 hover:bg-neutral-800 text-neutral-300 transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
              <a
                href={resultVideoUrl}
                download={`${baseName}_output.mp4`}
                className="flex items-center gap-1.5 rounded-lg border border-neutral-700 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-300 hover:text-white hover:bg-neutral-800 transition-colors ml-auto"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Download Video</span>
              </a>
            </div>
          </div>

          {/* Per-team heatmaps — real PNGs generated by heatmap/heatmap_generator.py */}
          <div className="space-y-3">
            <span className="text-xs font-semibold text-neutral-300 flex items-center gap-1.5 uppercase tracking-wide">
              <Flame className="h-3.5 w-3.5 text-amber-400" />
              Per-Team Workrate Heatmaps
            </span>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[
                { label: 'Team 1', url: team1HeatmapUrl },
                { label: 'Team 2', url: team2HeatmapUrl },
              ].map(({ label, url }) => (
                <div key={label} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-neutral-400">{label}</span>
                    {url && (
                      <button
                        onClick={() => downloadUrl(url, `${baseName}_${label.replace(' ', '_').toLowerCase()}_heatmap.png`)}
                        className="text-[11px] text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                      >
                        <Download className="h-3 w-3" />
                        <span>Download</span>
                      </button>
                    )}
                  </div>
                  <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-neutral-800 bg-neutral-950 flex items-center justify-center">
                    {url ? (
                      <img src={url} alt={`${label} workrate heatmap`} className="w-full h-full object-contain" />
                    ) : (
                      <span className="text-xs text-neutral-500">No heatmap</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
