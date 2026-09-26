import React, { useState } from 'react';
import { Users, X } from 'lucide-react';

interface PlayerHeatmap {
  player_id: number;
  url: string;
}

interface PlayerHeatmapsWidgetProps {
  hasResult: boolean;
  playerHeatmaps: PlayerHeatmap[];
}

// Each image is already a 2-panel composite (heatmap + player snapshot,
// see heatmap/heatmap_generator.py) — this widget is just a gallery/lightbox
// over the real per-player images the backend generated, no client-side
// rendering of its own.
export const PlayerHeatmapsWidget: React.FC<PlayerHeatmapsWidgetProps> = ({
  hasResult,
  playerHeatmaps,
}) => {
  const [expandedUrl, setExpandedUrl] = useState<string | null>(null);

  if (!hasResult || playerHeatmaps.length === 0) {
    return null;
  }

  return (
    <div id="player-heatmaps-widget" className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
        <h2 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
          <Users className="h-4 w-4 text-emerald-400" />
          <span>Per-Player Heatmaps</span>
        </h2>
        <span className="text-xs text-neutral-400 font-mono">{playerHeatmaps.length} players</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {playerHeatmaps.map(({ player_id, url }) => (
          <button
            key={player_id}
            onClick={() => setExpandedUrl(url)}
            className="group rounded-lg overflow-hidden border border-neutral-800 bg-neutral-950 hover:border-emerald-500/60 transition-colors text-left"
          >
            <div className="aspect-video overflow-hidden">
              <img
                src={url}
                alt={`Player #${player_id} heatmap`}
                loading="lazy"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
              />
            </div>
            <div className="px-2 py-1.5 text-[11px] font-mono text-neutral-300">
              Player #{player_id}
            </div>
          </button>
        ))}
      </div>

      {/* Lightbox */}
      {expandedUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
          onClick={() => setExpandedUrl(null)}
        >
          <button
            onClick={() => setExpandedUrl(null)}
            className="absolute top-4 right-4 h-9 w-9 flex items-center justify-center rounded-full bg-neutral-900 border border-neutral-700 text-neutral-300 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
          <img
            src={expandedUrl}
            alt="Player heatmap enlarged"
            className="max-w-full max-h-full rounded-lg border border-neutral-700"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};
