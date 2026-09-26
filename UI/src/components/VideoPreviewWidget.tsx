import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Eye, VideoOff } from 'lucide-react';

interface VideoPreviewWidgetProps {
  videoUrl: string | null;
  fileName?: string;
}

export const VideoPreviewWidget: React.FC<VideoPreviewWidgetProps> = ({
  videoUrl,
  fileName
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(true);

  // Reset playback when video changes
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  }, [videoUrl]);

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
      setDuration(videoRef.current.duration || 0);
    }
  };

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div id="video-preview-widget" className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-sm flex flex-col justify-between">
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800 mb-4">
        <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
          <Eye className="h-4 w-4 text-emerald-400" />
          <span>Video Preview Widget</span>
        </h2>
        {fileName && (
          <span className="text-xs text-neutral-400 truncate max-w-xs font-mono">
            {fileName}
          </span>
        )}
      </div>

      {videoUrl ? (
        <div className="flex flex-col space-y-3">
          {/* Video Container */}
          <div className="relative w-full aspect-video bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800 flex items-center justify-center">
            <video
              ref={videoRef}
              src={videoUrl}
              playsInline
              muted={isMuted}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              className="w-full h-full object-contain"
            />

            {!isPlaying && (
              <button
                onClick={togglePlay}
                id="preview-play-overlay-btn"
                className="absolute z-10 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-500/90 hover:bg-emerald-400 text-neutral-950 shadow-lg transition-transform active:scale-95"
              >
                <Play className="h-6 w-6 fill-current ml-0.5" />
              </button>
            )}
          </div>

          {/* Timeline Scrubber */}
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
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Controls Bar */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2">
              <button
                onClick={togglePlay}
                id="preview-play-btn"
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
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setIsMuted(!isMuted);
                  if (videoRef.current) videoRef.current.muted = !isMuted;
                }}
                className="h-8 w-8 flex items-center justify-center rounded-lg border border-neutral-800 bg-neutral-950 hover:bg-neutral-800 text-neutral-300 transition-colors"
              >
                {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center aspect-video w-full rounded-lg border border-neutral-800 bg-neutral-950/50 p-6 text-center">
          <div className="h-10 w-10 rounded-full bg-neutral-800/80 flex items-center justify-center text-neutral-400 mb-2">
            <VideoOff className="h-5 w-5" />
          </div>
          <p className="text-xs font-medium text-neutral-300">No video selected</p>
          <p className="text-[11px] text-neutral-400 mt-0.5">
            Select or drop a video file in the upload widget to preview it here
          </p>
        </div>
      )}
    </div>
  );
};
