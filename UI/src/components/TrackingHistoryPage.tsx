import React from 'react';
import { History, PlaySquare, Trash2, Calendar, Clock, Film, Upload } from 'lucide-react';
import { TrackingSession } from '../types';

interface TrackingHistoryPageProps {
  sessions: TrackingSession[];
  onSelectSession: (session: TrackingSession) => void;
  onDeleteSession: (sessionId: string) => void;
  onClearAll: () => void;
  onGoToAnalysis: () => void;
}

export const TrackingHistoryPage: React.FC<TrackingHistoryPageProps> = ({
  sessions,
  onSelectSession,
  onDeleteSession,
  onClearAll,
  onGoToAnalysis
}) => {
  return (
    <div id="tracking-history-page" className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <History className="h-5 w-5 text-emerald-400" />
            <span>Tracking History</span>
          </h1>
          <p className="text-xs text-neutral-400 mt-0.5">
            Past analyzed video sessions and generated court heatmaps
          </p>
        </div>

        {sessions.length > 0 && (
          <div className="flex items-center gap-2">
            <button
              onClick={onClearAll}
              id="clear-all-history-btn"
              className="px-3 py-1.5 rounded-lg border border-neutral-800 bg-neutral-900 hover:bg-neutral-800 text-xs text-neutral-400 hover:text-rose-400 transition-colors"
            >
              Clear History
            </button>
            <button
              onClick={onGoToAnalysis}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-semibold text-xs transition-colors"
            >
              <Upload className="h-3.5 w-3.5" />
              <span>Upload New Video</span>
            </button>
          </div>
        )}
      </div>

      {sessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-16 text-center rounded-xl border border-neutral-800 bg-neutral-900/40">
          <Film className="h-10 w-10 text-neutral-600 mb-3" />
          <h3 className="text-sm font-semibold text-white">No tracking history yet</h3>
          <p className="text-xs text-neutral-400 max-w-sm mt-1 mb-5">
            Processed videos and generated court heatmaps will be recorded here for quick access.
          </p>
          <button
            onClick={onGoToAnalysis}
            className="flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-semibold px-4 py-2 text-xs transition-colors"
          >
            <Upload className="h-4 w-4" />
            <span>Upload and Analyze Video</span>
          </button>
        </div>
      ) : (
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/70 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-neutral-800 bg-neutral-950/80 text-neutral-400 font-medium">
                  <th className="p-4 font-semibold">Video File</th>
                  <th className="p-4 font-semibold">Sport</th>
                  <th className="p-4 font-semibold">Processed Date</th>
                  <th className="p-4 font-semibold">Duration</th>
                  <th className="p-4 font-semibold">Tracked Objects</th>
                  <th className="p-4 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60">
                {sessions.map((session) => (
                  <tr
                    key={session.id}
                    className="hover:bg-neutral-800/40 transition-colors"
                  >
                    <td className="p-4">
                      <div
                        onClick={() => onSelectSession(session)}
                        className="font-medium text-white hover:text-emerald-400 cursor-pointer flex items-center gap-2"
                      >
                        <PlaySquare className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                        <span className="truncate max-w-xs">{session.fileName}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-neutral-800 text-neutral-300">
                        {session.sport}
                      </span>
                    </td>
                    <td className="p-4 text-neutral-400">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="h-3 w-3 text-neutral-500" />
                        <span>{new Date(session.createdAt).toLocaleDateString()}</span>
                      </div>
                    </td>
                    <td className="p-4 text-neutral-300 font-mono">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-neutral-500" />
                        <span>{session.durationSec.toFixed(1)}s</span>
                      </div>
                    </td>
                    <td className="p-4 text-neutral-300 font-mono">
                      {session.stats.trackedEntitiesCount}
                    </td>
                    <td className="p-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => onSelectSession(session)}
                          className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 font-medium transition-colors"
                        >
                          View Result
                        </button>
                        <button
                          onClick={() => onDeleteSession(session.id)}
                          title="Delete Session"
                          className="p-1 rounded text-neutral-500 hover:text-rose-400 hover:bg-neutral-800 transition-colors"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
