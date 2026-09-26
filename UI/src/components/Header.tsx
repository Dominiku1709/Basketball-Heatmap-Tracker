import React from 'react';
import { Activity, Film, History } from 'lucide-react';

interface HeaderProps {
  activePage: 'analysis' | 'history';
  onNavigate: (page: 'analysis' | 'history') => void;
  historyCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  activePage,
  onNavigate,
  historyCount
}) => {
  return (
    <header className="border-b border-neutral-800 bg-neutral-950 sticky top-0 z-30">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Brand */}
        <div 
          onClick={() => onNavigate('analysis')} 
          className="flex items-center gap-2.5 cursor-pointer"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500 text-neutral-950 font-bold">
            <Activity className="h-4 w-4 stroke-[2.5]" />
          </div>
          <div>
            <span className="font-semibold text-white text-sm">Basketball Heatmap Tracking</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-1 bg-neutral-900 p-1 rounded-lg border border-neutral-800 text-xs">
          <button
            onClick={() => onNavigate('analysis')}
            id="nav-analysis-btn"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-colors ${
              activePage === 'analysis'
                ? 'bg-neutral-800 text-white shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <Film className="h-3.5 w-3.5 text-emerald-400" />
            <span>Analysis</span>
          </button>

          <button
            onClick={() => onNavigate('history')}
            id="nav-history-btn"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-colors ${
              activePage === 'history'
                ? 'bg-neutral-800 text-white shadow-sm'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <History className="h-3.5 w-3.5 text-cyan-400" />
            <span>Tracking History</span>
            {historyCount > 0 && (
              <span className="ml-1 rounded-full bg-neutral-700 px-1.5 py-0.2 text-[10px] text-neutral-300 font-mono">
                {historyCount}
              </span>
            )}
          </button>
        </nav>
      </div>
    </header>
  );
};
