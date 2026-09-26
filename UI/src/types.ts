export type SportType = 'soccer' | 'basketball' | 'tennis' | 'general';

export interface TrackingStats {
  trackedEntitiesCount: number;
}

export interface TrackingSession {
  id: string;
  fileName: string;
  sport: SportType;
  createdAt: string;
  durationSec: number;
  /** Original uploaded video (blob: URL — only valid for this browser session). */
  videoUrl: string;
  /** Backend-hosted annotated output video (persists across reloads, unlike blob: URLs). */
  resultVideoUrl?: string;
  team1HeatmapUrl?: string;
  team2HeatmapUrl?: string;
  playerHeatmaps?: { player_id: number; url: string }[];
  stats: TrackingStats;
}

