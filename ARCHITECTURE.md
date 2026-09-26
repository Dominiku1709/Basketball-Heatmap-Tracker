# Kiến trúc hệ thống

Sơ đồ kiến trúc module + luồng xử lý end-to-end cho
`AI_BasketBall_Analysis_v1`. Render trực tiếp trên GitHub (Mermaid), không
cần file ảnh riêng.

---

## 1. Kiến trúc module

```mermaid
flowchart TB
    subgraph Input
        V[Video input<br/>input_videos/*.mp4]
    end

    subgraph Detection & Tracking
        PT["PlayerTracker<br/>(YOLOv8m/YOLOv8n/RT-DETR-L + BoT-SORT<br/>— chọn qua player_model)<br/>trackers/player_tracker.py"]
        CK["CourtKeypointDetector<br/>(YOLOv8m-pose)<br/>Court_keypoint_detection/"]
        REG["PLAYER_MODEL_REGISTRY<br/>configs/configs.py"]
    end

    subgraph "Team & Geometry"
        TA["TeamAssigner<br/>(K-means màu áo)<br/>team_assigner/"]
        TV["TacticalViewConverter<br/>(Homography)<br/>tactical_view/"]
    end

    subgraph Analytics
        HM["HeatmapGenerator<br/>(per-player + per-team)<br/>heatmap/"]
    end

    subgraph Rendering
        DR["4 Drawers<br/>(player/keypoint/frame#/tactical)<br/>drawers/"]
        SW["StreamingVideoWriter<br/>(ffmpeg H.264)<br/>utils/video_utils.py"]
    end

    subgraph "Serving (mục 7)"
        API["FastAPI<br/>api/app.py<br/>GET /models, POST /analyze"]
        UI["React UI<br/>UI/<br/>widget chọn model"]
    end

    REG -.resolve path theo player_model.-> PT
    V --> PT
    V --> CK
    V --> TA
    PT --> TA
    PT --> TV
    CK --> TV
    TV --> HM
    TA --> HM
    V --> HM
    PT --> DR
    CK --> DR
    TV --> DR
    TA --> DR
    DR --> SW
    SW --> OUT["output/run_N/*.mp4 + heatmaps/*.png + run_info.json"]

    PIPE["pipeline.run_analysis()<br/>(dùng chung CLI + API)"] -.gọi tất cả.-> PT
    PIPE -.-> CK
    PIPE -.-> TA
    PIPE -.-> TV
    PIPE -.-> HM
    PIPE -.-> DR

    CLI["main.py (CLI)<br/>--player_model"] --> PIPE
    API --> PIPE
    UI -- "GET /models (danh sách), POST /analyze (kèm player_model), poll GET /jobs/{id}" --> API
```

**Ghi chú kiến trúc quan trọng** (đã đổi so với repo gốc — xem
`DISCOVERY_REPORT.md`/`PRD_CLAUDE_CODE.md` để biết lý do):
- Ball detection/tracking, pass/interception, ball possession — **đã bỏ**
  (out-of-scope theo PRD, thu hẹp còn 2 chức năng: track player + heatmap).
- Tracker: BoT-SORT (Ultralytics native, có Camera Motion Compensation +
  Re-ID) thay ByteTrack thuần của repo gốc.
- Team assigner: K-means color clustering thay Fashion CLIP (nhẹ, không
  cần internet/model ngoài).
- Player detector có thể chọn giữa 3 kiến trúc đã train thật (YOLOv8m mặc
  định, YOLOv8n, RT-DETR-L — số liệu mAP xem `REPORT.md` mục 3.1/3.3), qua
  `PLAYER_MODEL_REGISTRY` (`configs/configs.py`) → `pipeline.run_analysis(player_model=...)`
  → `--player_model` (CLI) hoặc form field `player_model` (`POST /analyze`).
  Stub cache tách riêng theo model (`stubs/<video>/<model>/...`); stub
  court-keypoint vẫn dùng chung theo video vì không phụ thuộc detector
  cầu thủ. Chỉ 3 model này có file `.pt` thật trong repo.

## 2. Luồng xử lý end-to-end (1 lần chạy `pipeline.run_analysis()`)

```mermaid
sequenceDiagram
    participant U as User (CLI/UI)
    participant P as pipeline.run_analysis
    participant D as PlayerTracker
    participant K as CourtKeypointDetector
    participant T as TeamAssigner
    participant G as TacticalViewConverter
    participant H as HeatmapGenerator
    participant W as StreamingVideoWriter

    U->>P: input_video_path, player_model
    P->>P: read_video() — toàn bộ frame vào RAM
    P->>P: get_player_detector_path(player_model)
    P->>D: get_object_tracks(frames)<br/>(batch 20, cache theo stub — riêng theo player_model)
    D-->>P: player_tracks (bbox theo track_id/frame)
    P->>K: get_court_keypoints(frames)
    K-->>P: court_keypoints_per_frame
    P->>T: get_player_teams_across_frames(frames, player_tracks)
    T-->>P: player_assignment (team_id/track_id)
    P->>G: transform_players_to_tactical_view(keypoints, tracks)
    G-->>P: tactical_player_positions + homography_valid
    P->>P: lọc court-boundary<br/>(bỏ track ngoài sân khi homography hợp lệ)
    P->>H: generate(positions, assignment, tracks, frames)
    H-->>P: heatmap PNG (player composite + team)
    loop mỗi frame (streaming, giải phóng RAM ngay sau khi ghi)
        P->>P: vẽ 4 lớp (player/keypoint/frame#/tactical)
        P->>W: write(frame)
    end
    W-->>P: video H.264 hoàn chỉnh
    P-->>U: run_info.json (stats + đường dẫn output)
```

**Vì sao streaming ở bước vẽ cuối**: giữ toàn bộ video trong RAM 1 lần
(cần thiết vì 3 bước detect/track/keypoint/team-assign đều duyệt hết
frame) là chấp nhận được, nhưng **vẽ + ghi video phải streaming từng
frame** — nếu không sẽ tạo thêm nhiều bản copy toàn video (1 bản/drawer),
gây tràn RAM với video dài/độ phân giải cao (đã gặp thật với video 2312
frame/1080p, xem `DISCOVERY_REPORT.md`).
