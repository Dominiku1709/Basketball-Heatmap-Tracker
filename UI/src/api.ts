// api.ts
//
// Talks to the real FastAPI backend (api/app.py) — replaces the fake
// setTimeout + Math.random() flow that used to live in App.tsx.

const API_BASE_URL = 'http://localhost:8000';

export interface AnalysisResult {
  run_id: string;
  timestamp: string;
  input_video: string;
  num_frames: number;
  num_players_detected: number;
  models: {
    format: string;
    player_model: string;
    player_model_label: string;
    player_detector: string;
    court_keypoint_detector: string;
  };
  output_video: string;
  heatmap_dir: string;
  output_video_url: string;
  heatmap_dir_url: string;
  /** One composite (heatmap + player snapshot) image per player — see heatmap/heatmap_generator.py. */
  player_heatmaps: { player_id: number; url: string }[];
}

export type JobStatusValue = 'pending' | 'processing' | 'done' | 'error';

export interface JobStatus {
  job_id: string;
  status: JobStatusValue;
  source_filename?: string;
  created_at?: string;
  result?: AnalysisResult;
  error?: string;
}

export interface PlayerModelOption {
  id: string;
  label: string;
  description: string;
}

export async function listPlayerModels(): Promise<{ default: string; models: PlayerModelOption[] }> {
  const res = await fetch(`${API_BASE_URL}/models`);
  if (!res.ok) {
    throw new Error(`Could not fetch model list (HTTP ${res.status}).`);
  }
  return res.json();
}

export async function uploadVideoForAnalysis(file: File, playerModel: string): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('player_model', playerModel);

  const res = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body: formData });
  if (!res.ok) {
    throw new Error(`Upload failed (HTTP ${res.status}). Is the backend running on ${API_BASE_URL}?`);
  }
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Could not fetch job status (HTTP ${res.status}).`);
  }
  return res.json();
}

/** Resolves a path the backend returned (e.g. "/output/run_3/x.mp4") into an absolute URL. */
export function resolveMediaUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

/** Polls GET /jobs/{jobId} every `intervalMs` until status is 'done' or 'error'. */
export async function pollJobUntilDone(
  jobId: string,
  onTick?: (job: JobStatus) => void,
  intervalMs = 2000,
): Promise<JobStatus> {
  while (true) {
    const job = await getJobStatus(jobId);
    onTick?.(job);
    if (job.status === 'done' || job.status === 'error') {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
