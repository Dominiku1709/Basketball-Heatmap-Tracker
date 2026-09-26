# 🏀 Basketball Player Tracking & Heatmap Analysis

Theo dõi từng cầu thủ bóng rổ xuyên suốt video broadcast (chịu được camera
pan + occlusion) và sinh heatmap workrate — theo từng cầu thủ và từng đội
— ánh xạ lên toạ độ sân thực (không phải toạ độ pixel thô).

> Dự án này là bài mini-project cuối module Computer Vision, được adapt từ
> repo gốc [HanaFEKI/AI_BasketBall_Analysis_v1](https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1)
> theo hướng thu hẹp phạm vi (chỉ giữ 2 chức năng: tracking + heatmap, bỏ
> ball detection/pass/possession) — xem `PRD_CLAUDE_CODE.md` và
> `DISCOVERY_REPORT.md` để biết đầy đủ lý do/quá trình adapt.

**Tài liệu liên quan**: [`REPORT.md`](REPORT.md) (báo cáo đầy đủ 7 mục —
problem statement, data, training, evaluation, feedback loop, ý tưởng
riêng, deployment) · [`ARCHITECTURE.md`](ARCHITECTURE.md) (sơ đồ kiến
trúc + luồng xử lý).

---

## Chức năng chính

1. **Track từng cầu thủ** — YOLOv8m + BoT-SORT (Camera Motion Compensation
   + Re-ID), ID nhất quán xuyên suốt clip.
2. **Heatmap workrate** — per-player (composite: heatmap + ảnh crop cầu
   thủ) và per-team, ánh xạ toạ độ sân thực qua homography (từ court
   keypoint model YOLOv8m-pose).
3. **Web demo** — FastAPI backend + React UI, upload video → xem video
   output + heatmap thật ngay trên trình duyệt.

---

## 1. Cài đặt môi trường

Dùng conda (khuyến nghị, tránh xung đột với các project Python khác trên
máy — xem lý do cụ thể ở `DISCOVERY_REPORT.md`):

```cmd
conda create -n basketball-ai python=3.11 -y
conda activate basketball-ai
cd /d <đường-dẫn-tới-repo>
pip install -r requirements.txt
```

## 2. Cấu hình model weight

Model weight (`.pt`) **không nằm trong repo** (file lớn, xem `.gitignore`).
Copy `.env.example` thành `.env` rồi trỏ đúng đường dẫn weight cục bộ của
bạn:

```
PLAYER_WEIGHT_PATH="đường/dẫn/tới/player_detector/best.pt"
COURT_WEIGHT_PATH="đường/dẫn/tới/court_keypoint/best.pt"
MODEL_FORMAT="pt"   # hoặc "onnx" sau khi export (mục 5)
```

Nếu chưa có weight: xem mục 3 (huấn luyện lại) hoặc liên hệ để lấy weight
đã train sẵn (yolov8m player detector, yolov8m-pose court keypoint — số
liệu train thật ở `REPORT.md` mục 3).

## 3. Huấn luyện lại (tuỳ chọn — pipeline đã có weight sẵn thì bỏ qua)

Notebook mẫu (khung quy trình, cần điền API key Roboflow thật):
- [`training/player_detection_train.ipynb`](training/player_detection_train.ipynb)
- [`training/keypoint_court_train.ipynb`](training/keypoint_court_train.ipynb)

Huấn luyện thật của model đang dùng trong pipeline được chạy trên Kaggle
(2× Tesla T4) — hyperparameter, log train theo epoch, kết quả mAP đầy đủ ở
`REPORT.md` mục 3. Kết quả train (`results.csv`, `args.yaml`, confusion
matrix, PR curve...) nằm ở `training/result/weight/player/` và
`training/result/weight/pose/`.

## 4. Chạy pipeline (CLI)

```cmd
conda activate basketball-ai
python main.py input_videos/video_1.mp4
```

Kết quả: mỗi lần chạy tạo 1 folder mới `output/run_N/` gồm video output đã
annotate (H.264, xem được trên browser), heatmap PNG (per-player composite
+ per-team) và `run_info.json` (thống kê + đường dẫn).

Lần chạy đầu cho mỗi video sẽ chậm (phải detect+track toàn bộ); các lần
sau tái dùng cache trong `stubs/<tên_video>/` nên nhanh hơn nhiều. Đổi
video khác bằng cách đổi đường dẫn tham số.

## 5. Export ONNX + đổi backend suy luận

```cmd
python scripts/export_onnx.py
python scripts/benchmark_export.py --video input_videos/video_1.mp4
```

`export_onnx.py` xuất cả player detector + court keypoint model sang ONNX
(lưu cạnh file `.pt` gốc). Đặt `MODEL_FORMAT="onnx"` trong `.env` để
pipeline tự chuyển sang dùng bản ONNX (tự fallback về `.pt` nếu chưa
export) — không cần sửa code. `benchmark_export.py` đo latency/FPS thật
PyTorch vs ONNXRuntime, kết quả lưu `output/benchmark_export.json` (số
liệu thật đã có sẵn trong `REPORT.md` mục 7).

## 6. Chạy web demo (backend + frontend)

Mở 2 cửa sổ terminal riêng:

```cmd
:: Terminal 1 — backend
conda activate basketball-ai
uvicorn api.app:app --reload --port 8000
```

```cmd
:: Terminal 2 — frontend
cd UI
npm install
npm run dev
```

Mở trình duyệt vào `http://localhost:3000`, upload video, bấm "Process
Video" — chạy pipeline thật qua API (`POST /analyze` → poll
`GET /jobs/{id}`), hiển thị video/heatmap output thật (không phải số liệu
giả). Swagger UI của backend: `http://127.0.0.1:8000/docs`.

---

## Cấu trúc thư mục (tóm tắt — chi tiết xem `ARCHITECTURE.md`)

```
main.py                CLI entry point (gọi pipeline.py)
pipeline.py             Logic xử lý chính, dùng chung cho CLI + API
api/                    FastAPI backend (web demo)
UI/                     React frontend (web demo)
trackers/               PlayerTracker (YOLOv8m + BoT-SORT)
Court_keypoint_detection/  CourtKeypointDetector (YOLOv8m-pose)
team_assigner/           TeamAssigner (K-means màu áo)
tactical_view/           Homography — pixel video → toạ độ sân thực
heatmap/                 HeatmapGenerator (per-player composite + per-team)
drawers/                 Vẽ overlay lên video output
utils/                   Tiện ích dùng chung (video I/O, run folder, bbox)
configs/                 Cấu hình tập trung (đường dẫn weight, màu team...)
scripts/                 export_onnx.py, benchmark_export.py
training/                Notebook + script train, kết quả train thật
```

## Hạn chế đã biết (trung thực, không giấu)

Tracker (BoT-SORT) vẫn còn fragment ID trong các pha chuyển động nhanh +
camera pan đồng thời — đã thử 3 hướng cải thiện (buffer/threshold/ReID
model thật), cải thiện ~10%, chưa giải quyết tận gốc. Chi tiết phân tích
lỗi + bằng chứng trực quan (ảnh thật từng frame): `REPORT.md` mục 4–5.
