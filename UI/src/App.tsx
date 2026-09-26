import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { UploadVideoWidget } from './components/UploadVideoWidget';
import { VideoPreviewWidget } from './components/VideoPreviewWidget';
import { TrackingResultWidget } from './components/TrackingResultWidget';
import { PlayerHeatmapsWidget } from './components/PlayerHeatmapsWidget';
import { TrackingHistoryPage } from './components/TrackingHistoryPage';
import { TrackingSession, SportType, TrackingStats } from './types';
import {
  uploadVideoForAnalysis,
  pollJobUntilDone,
  resolveMediaUrl,
  listPlayerModels,
  AnalysisResult,
  PlayerModelOption,
} from './api';

export default function App() {
  const [activePage, setActivePage] = useState<'analysis' | 'history'>('analysis');

  // Video state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [sport, setSport] = useState<SportType>('basketball');

  // Player detector model state — fetched from GET /models so the UI never
  // hardcodes what's actually available on the backend.
  const [models, setModels] = useState<PlayerModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [processError, setProcessError] = useState<string | null>(null);

  // Result state — real backend output, not simulated
  const [hasResult, setHasResult] = useState(false);
  const [resultVideoUrl, setResultVideoUrl] = useState<string | undefined>(undefined);
  const [team1HeatmapUrl, setTeam1HeatmapUrl] = useState<string | undefined>(undefined);
  const [team2HeatmapUrl, setTeam2HeatmapUrl] = useState<string | undefined>(undefined);
  const [playerHeatmaps, setPlayerHeatmaps] = useState<{ player_id: number; url: string }[]>([]);
  const [currentResultStats, setCurrentResultStats] = useState<TrackingStats | undefined>(undefined);
  const [resultDuration, setResultDuration] = useState(15);

  // History state persisted in localStorage
  const [sessions, setSessions] = useState<TrackingSession[]>(() => {
    try {
      const saved = localStorage.getItem('sports_tracking_sessions');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Could not read saved sessions', e);
    }
    return [];
  });

  useEffect(() => {
    try {
      localStorage.setItem('sports_tracking_sessions', JSON.stringify(sessions));
    } catch (e) {
      console.warn('Could not write saved sessions', e);
    }
  }, [sessions]);

  // Fetch the real list of player-detector models from the backend once on
  // load, so the picker never hardcodes what's actually available.
  useEffect(() => {
    listPlayerModels()
      .then(({ default: def, models: list }) => {
        setModels(list);
        setSelectedModel(def);
      })
      .catch((e) => console.warn('Could not fetch player model list', e));
  }, []);

  const handleFileSelect = (file: File) => {
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl);
    }
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setVideoUrl(url);
    setHasResult(false);
    setProcessError(null);
  };

  const handleClearFile = () => {
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl);
    }
    setSelectedFile(null);
    setVideoUrl(null);
    setHasResult(false);
    setProcessError(null);
  };

  // Runs the real pipeline via the FastAPI backend (api/app.py):
  // upload -> job_id -> poll /jobs/{id} until done -> real video/heatmap URLs + stats.
  const handleProcessVideo = async () => {
    if (!videoUrl || !selectedFile) return;

    if (!(selectedFile instanceof File)) {
      // The old UI had a "load sample clip" shortcut that faked a File
      // object just for local preview — that object has no real bytes to
      // upload. Real analysis needs a real uploaded file.
      setProcessError('This sample clip is preview-only. Upload a real video file to run analysis.');
      return;
    }

    setIsProcessing(true);
    setProcessProgress(0);
    setProcessError(null);

    try {
      const { job_id } = await uploadVideoForAnalysis(selectedFile, selectedModel);
      setProcessProgress(10);

      const finalJob = await pollJobUntilDone(job_id, () => {
        // No real percentage from the backend (it's one long-running job,
        // not per-frame progress) — nudge the bar forward on each poll so
        // it doesn't look frozen, without claiming false precision.
        setProcessProgress((prev) => Math.min(90, prev + 5));
      });

      if (finalJob.status === 'error' || !finalJob.result) {
        throw new Error(finalJob.error || 'Analysis failed on the server.');
      }

      setProcessProgress(100);
      finishProcessing(finalJob.result);
    } catch (err) {
      setProcessError(err instanceof Error ? err.message : String(err));
      setIsProcessing(false);
    }
  };

  const finishProcessing = (result: AnalysisResult) => {
    setIsProcessing(false);
    setHasResult(true);

    const stats: TrackingStats = {
      trackedEntitiesCount: result.num_players_detected,
    };

    const videoUrlAbs = resolveMediaUrl(result.output_video_url);
    const team1Url = resolveMediaUrl(`${result.heatmap_dir_url}/teams/team_1_heatmap.png`);
    const team2Url = resolveMediaUrl(`${result.heatmap_dir_url}/teams/team_2_heatmap.png`);
    const playerHeatmapUrls = result.player_heatmaps.map((h) => ({
      player_id: h.player_id,
      url: resolveMediaUrl(h.url),
    }));
    const durationSec = result.num_frames > 0 ? result.num_frames / 30 : 15; // fps not returned by the API yet — 30 is an approximation, not the real source fps

    setResultVideoUrl(videoUrlAbs);
    setTeam1HeatmapUrl(team1Url);
    setTeam2HeatmapUrl(team2Url);
    setPlayerHeatmaps(playerHeatmapUrls);
    setCurrentResultStats(stats);
    setResultDuration(durationSec);

    const fileName = selectedFile?.name || 'uploaded_match_clip.mp4';
    const newSession: TrackingSession = {
      id: `session-${Date.now()}`,
      fileName,
      sport,
      createdAt: new Date().toISOString(),
      durationSec,
      videoUrl: videoUrl || '',
      resultVideoUrl: videoUrlAbs,
      team1HeatmapUrl: team1Url,
      team2HeatmapUrl: team2Url,
      playerHeatmaps: playerHeatmapUrls,
      stats,
    };

    setSessions((prev) => [newSession, ...prev]);
  };

  // Re-open session from history
  const handleSelectSession = (session: TrackingSession) => {
    setVideoUrl(session.videoUrl);
    setSelectedFile({
      name: session.fileName,
      size: 0,
      type: 'video/mp4'
    } as File);
    setSport(session.sport);
    setResultVideoUrl(session.resultVideoUrl);
    setTeam1HeatmapUrl(session.team1HeatmapUrl);
    setTeam2HeatmapUrl(session.team2HeatmapUrl);
    setPlayerHeatmaps(session.playerHeatmaps || []);
    setCurrentResultStats(session.stats);
    setResultDuration(session.durationSec);
    setHasResult(true);
    setActivePage('analysis');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
  };

  const handleClearAllHistory = () => {
    setSessions([]);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-neutral-950">
      <Header
        activePage={activePage}
        onNavigate={setActivePage}
        historyCount={sessions.length}
      />

      <main className="flex-1 mx-auto w-full max-w-6xl px-4 sm:px-6 py-6">
        {activePage === 'analysis' ? (
          <div className="space-y-6">
            {/* Top Row: Upload Widget & Video Preview Widget */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 1. Upload Video Widget */}
              <UploadVideoWidget
                selectedFile={selectedFile}
                videoUrl={videoUrl}
                onFileSelect={handleFileSelect}
                onClearFile={handleClearFile}
                onProcessVideo={handleProcessVideo}
                isProcessing={isProcessing}
                processProgress={processProgress}
                processError={processError}
                models={models}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
              />

              {/* 2. Video Preview Widget */}
              <VideoPreviewWidget
                videoUrl={videoUrl}
                fileName={selectedFile?.name}
              />
            </div>

            {/* Bottom Row: 3. Tracking Result Widget */}
            <TrackingResultWidget
              hasResult={hasResult}
              resultVideoUrl={resultVideoUrl}
              team1HeatmapUrl={team1HeatmapUrl}
              team2HeatmapUrl={team2HeatmapUrl}
              fileName={selectedFile?.name}
              stats={currentResultStats}
              durationSec={resultDuration}
            />

            {/* 4. Per-Player Heatmaps Widget */}
            <PlayerHeatmapsWidget
              hasResult={hasResult}
              playerHeatmaps={playerHeatmaps}
            />
          </div>
        ) : (
          /* Tracking History Page */
          <TrackingHistoryPage
            sessions={sessions}
            onSelectSession={handleSelectSession}
            onDeleteSession={handleDeleteSession}
            onClearAll={handleClearAllHistory}
            onGoToAnalysis={() => setActivePage('analysis')}
          />
        )}
      </main>

      <footer className="border-t border-neutral-900 py-4 text-center text-xs text-neutral-600">
        Basketball Player Tracking &amp; Heatmap Analysis
      </footer>
    </div>
  );
}
