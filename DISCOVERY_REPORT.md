# DISCOVERY_REPORT.md — Phase 0

Khảo sát repo `AI_BasketBall_Analysis_v1` theo yêu cầu mục 4 của `PRD_CLAUDE_CODE.md`.
**Không có code nào bị sửa trong quá trình này.**

---

## 1. Cấu trúc thư mục thật

```
Court_keypoint_detection/{Readme.md, __init__.py, court_keypoint_detection.py}
ball_acquisition/{README.md, __init__.py, ball_acquisition.py}
configs/{__init__.py, configs.py}
drawers/{README.md, __init__.py, ball_tracks_drawer.py, court_keypoints_drawer.py,
         pass_interception_drawer.py, player_tracks_drawer.py, tactical_view_drawer.py,
         team_ball_control_drawer.py, utils.py}
input_videos/{README.md, video_1.mp4, video_2.mp4, video_3.mp4}
output_videos/{Video_1_output.gif, Video_1_output.mp4}
pass_interception_detector/{README.md, __init__.py, pass_interception_detector.py}
speed_and_distance_calculation/{Readme.md, __init__.py, speed_and_distance.py}
tactical_view/{Readme.md, __init__.py, homography.py, tactical_view.py,
               court_images/{README.md, basketball-court-dimensions.png, basketball_court.png}}
team_assigner/{README.md, __init__.py, team_assigner.py}
trackers/{README.md, __init__.py, ball_tracker.py, player_tracker.py}
training/{README.md, ball_detection.py, dataset_config.py, get_weights.py,
          keypoint_court_train.py, player_detection.py,
          keypoint_court_train.ipynb, player_detection_train.ipynb}  ← 2 file .ipynb do tôi vừa tạo trước đó
utils/{README.md, __init__.py, bbox_utils.py, stubs_utils.py, video_utils.py}
main.py, requirements.txt, README.md, PRD_CLAUDE_CODE.md
```

**Không có** thư mục `models/` hay `stubs/` nào trong repo, và **không có** file `.gitignore`.
→ PRD (mục 3) giả định weight `.pt` (yolov8m, yolov8n, rtdetr-l) đã có sẵn trong repo — **thực tế repo không chứa weight nào**. `training/get_weights.py` và `README.md` gốc chỉ trỏ tới 1 link Google Drive (`best.pt`) để tải thủ công. Cần Phuc xác nhận: 3 weight mới sẽ được đặt ở đâu (dự kiến `models/`), và có đúng class `player` (chữ thường) không (xem mục 6).

---

## 2. Khảo sát từng module

### `trackers/`
- **Tracker đang dùng**: `supervision.ByteTrack()` thuần (`trackers/player_tracker.py:18`, `trackers/ball_tracker.py`). **Không có** Camera Motion Compensation, **không có** Re-ID/appearance matching — đúng như lo ngại nêu ở PRD mục 5 (kinh nghiệm dự án cũ: ByteTrack không xử lý được camera pan liên tục → ID fragmentation).
- `PlayerTracker` lọc detection theo tên class `"player"` (so khớp string qua `detection.names`), `BallTracker` lọc theo `"Ball"` (viết hoa chữ B). Đây là 2 model YOLO **hoàn toàn tách biệt** (không phải 1 model multi-class) — player detector và ball detector chạy độc lập, `main.py` gọi 2 tracker riêng.
- `BallTracker` còn có `remove_wrong_detections` (lọc theo khoảng cách di chuyển tối đa/frame) và `interpolate_ball_positions` (nội suy pandas) — 2 hàm này chỉ phục vụ ball, sẽ bị loại bỏ hoàn toàn nếu drop ball detection.

### `team_assigner/`
- Dùng **Fashion CLIP** (`patrickjohncyh/fashion-clip`) qua `transformers.CLIPModel`/`CLIPProcessor` — đúng README (zero-shot).
- **Cần internet** ở lần chạy đầu để tải weight CLIP từ Hugging Face Hub (sau đó cache local, có thể chạy offline nếu cache đã có — nhưng máy môi trường build/CI không internet sẽ fail lần đầu).
- **Tốc độ**: forward pass CLIP riêng lẻ **cho từng player, từng frame** (`get_player_color` gọi 1 lần/player, không batch) — sẽ chậm với video dài nhiều cầu thủ. Có cache theo `player_id` trong 50 frame (`player_team_dict` reset mỗi 50 frame) để giảm số lần gọi model.
- Package `transformers` **không có trong `requirements.txt`** và **không được cài** trong môi trường Python hiện tại (xem mục 4).

### `Court_keypoint_detection/`
- Chỉ có **code**, **không có weight** train sẵn trong repo, và **không có dataset keypoint** kèm theo. `training/keypoint_court_train.py` (+ notebook mới tạo) tải dataset Roboflow (`fyp-3bwmg/reloc2-den7l`) on-demand khi train, không lưu trong repo.
- `CourtKeypointDetector` (`court_keypoint_detection.py:7`) nhận `model_path` trỏ tới 1 YOLOv8-pose `.pt` — path này lấy từ `configs.COURT_KEYPOINT_DETECTOR_PATH = 'models/court_keypoint_detector.pt'` → **không tồn tại**.
- **Bug**: `Court_keypoint_detection/__init__.py` viết `from .court_keypoint_detection import CourtKeypointDetection` nhưng class thật trong file là `CourtKeypointDetector` (không phải `CourtKeypointDetection`) → `ImportError` ngay khi import package này.

### `tactical_view/`
- `TacticalViewConverter` (`tactical_view.py`): nhận court keypoints/frame + player tracks, dùng `Homography` (RANSAC + flip-correction + reprojection-error check + temporal smoothing bằng blend với homography frame trước) để map foot-position của player sang toạ độ **top-down 300×161 px** tương ứng sân thực 28×15 m.
- Input: `court_keypoints_per_frame` (từ `CourtKeypointDetector`) + `player_tracks` (dict `{track_id: {"bbox":...}}`/frame). Output: `tactical_player_positions` — list[dict] `{player_id: [x, y]}` theo toạ độ tactical-view mỗi frame. **Đây chính xác là output PRD mục 5 đề xuất tái sử dụng làm input cho module heatmap** (đã có toạ độ court-space, không phải pixel thô) — khả thi, không cần làm lại homography.
- **Bug tương tự Court_keypoint_detection**: `tactical_view/__init__.py` viết `from .tactical_view import TacticalView` nhưng class thật là `TacticalViewConverter` → `ImportError`.

### `drawers/`
Những gì đang được vẽ lên video output (`main.py`):
- `PlayerTracksDrawer` — ellipse dưới chân + ID player, tam giác đỏ cho player đang giữ bóng.
- `BallTracksDrawer` — tam giác chỉ vị trí bóng (màu khác nhau tuỳ có ai giữ bóng hay không).
- `CourtKeypointsDrawer`(**có "s"**) — chấm đỏ + label tại các keypoint sân.
- `TeamBallControlDrawer` — box % kiểm soát bóng theo đội, góc dưới trái.
- `PassInterceptionDrawer` — box đếm pass/interception theo đội.
- `TacticalViewDrawer` — dán ảnh sân top-down lên góc frame, vẽ chấm tròn màu đội cho vị trí player + khoanh đỏ cho ai giữ bóng. **Lưu ý**: file này có hard-code đặc biệt cho `pid == 6`, `pid == 13`, `pid == 26` (gán màu cố định / skip vẽ) — rõ ràng là code debug/test riêng cho 1 video cụ thể, không tổng quát, nên loại bỏ khi viết lại.
- `main.py` còn gọi `FrameNumberDrawer` và `SpeedAndDistanceDrawer` — **2 class này không tồn tại ở bất kỳ đâu trong repo** (`grep` toàn repo không ra kết quả nào ngoài chính `main.py`). Đây là tính năng chưa từng được implement, không phải lỗi import path.

### Model detection gốc — class & tương thích với weight mới
- 2 model tách biệt, mỗi model 1 class: player detector → class `"player"` (chữ thường), ball detector → class `"Ball"` (chữ B hoa). Việc match class hoàn toàn bằng so sánh tên string lấy từ `model.names` của chính file `.pt` đó.
- Weight mới (yolov8m/yolov8n/rtdetr-l, chỉ có class player) **tương thích trực tiếp** làm player detector **nếu** tên class trong weight mới đúng là `"player"` (cần Phuc xác nhận khi có file thật — nếu tên khác, ví dụ `"Player"` hoa, phải sửa `class_id_map.get("player")` ở `trackers/player_tracker.py:61`).
- Vì weight mới **không có** class ball → không cần sửa gì thêm cho việc "xử lý output" ngoài việc **không gọi `BallTracker` nữa** — điều này tự động kéo theo phải bỏ luôn `ball_acquisition/`, `pass_interception_detector/`, `drawers/ball_tracks_drawer.py`, `TeamBallControlDrawer`, `PassInterceptionDrawer` (tất cả đều nhận `ball_tracks`/`ball_aquisition` làm input bắt buộc) — khớp với phạm vi "ngoài scope" ở PRD mục 2.

---

## 3. Kết quả chạy thử pipeline gốc

Lệnh: `python main.py input_videos/video_1.mp4`

**Không chạy được.** Lỗi xảy ra ngay ở import đầu tiên (`from utils import read_video, save_video` → `import cv2`), **trước khi** kịp chạm tới các lỗi import sai path/class nêu ở mục 2:

```
AttributeError: _ARRAY_API not found
...
ImportError: numpy.core.multiarray failed to import
```

Nguyên nhân: môi trường Python hiện tại (Python 3.14, global site-packages) có `opencv-python-headless 4.9.0.80` (biên dịch với NumPy 1.x) xung đột với `numpy 2.5.0` đã cài. Đây là lỗi **môi trường**, không phải lỗi logic code.

Ngoài ra, môi trường hiện tại **chưa cài** `ultralytics`, `supervision`, `roboflow`, `transformers` — tức là dù fix được xung đột numpy/opencv, `main.py` sẽ tiếp tục fail ngay sau đó vì thiếu package, rồi tiếp tục fail vì các lỗi import path/class sai liệt kê dưới đây. Tôi dừng lại ở đây, chưa cài thêm gì / sửa gì, đúng theo yêu cầu "chỉ báo cáo lỗi, không code" của Phase 0.

**Danh sách lỗi sẽ gặp tiếp theo nếu chạy tiếp (đọc code, chưa chạy thực tế vì đã chặn ở bước trên):**

| Vị trí | Lỗi |
|---|---|
| `main.py:6` | `from court_keypoint_detector import CourtKeypointDetector` — sai path (đúng: `Court_keypoint_detection`), và package đích cũng tự broken (mục 2) |
| `main.py:7` | `from ball_aquisition import BallAquisitionDetector` — sai path (đúng: `ball_acquisition`) |
| `main.py:8` | `from pass_and_interception_detector import ...` — sai path (đúng: `pass_interception_detector`) |
| `main.py:9` | `from tactical_view_converter import TacticalViewConverter` — sai path (đúng: `tactical_view`), package đích cũng tự broken (mục 2) |
| `main.py:10` | `from speed_and_distance_calculator import ...` — sai path (đúng: `speed_and_distance_calculation`) |
| `main.py:14` | `CourtKeypointDrawer` — tên sai, thật ra là `CourtKeypointsDrawer` (có "s") |
| `main.py:16,19` | `FrameNumberDrawer`, `SpeedAndDistanceDrawer` — class chưa từng được viết |
| `drawers/ball_tracks_drawer.py:1` | `from .utils import draw_traingle` — typo, hàm thật tên `draw_triangle` trong `drawers/utils.py` |
| `configs/configs.py` | trỏ tới `models/*.pt` — thư mục `models/` không tồn tại |

**Kết luận Phase 0 (mục 8 PRD)**: pipeline gốc **không chạy được** — vừa do môi trường (numpy/opencv conflict, thiếu package), vừa do chính repo có nhiều lỗi import path/tên class/class chưa tồn tại (bảng trên). `requirements.txt` hiện tại (`ultralytics>=8.0.0`, `roboflow`, và dòng `python_version >=3.8` — dòng này không phải cú pháp pip hợp lệ, nên bị bỏ qua khi `pip install`) **thiếu** hầu hết dependency thật sự được import trong code: `supervision`, `transformers`, `opencv-python`, `pandas`, `numpy`, `Pillow`. Việc dọn dẹp `requirements.txt` chắc chắn sẽ cần làm ở Phase 2 — sẽ báo cáo mọi thay đổi vào file đó khi thực hiện, theo yêu cầu của Phuc.

---

## 3b. Đính chính (2026-09-17) — sau khi có log train thật

Log W&B thật (`training/wandb_yolov8n_output.zip`, `training/yolov8m_final.zip`,
seed=42, cùng epoch/batch/imgsz) cho thấy **giả định ở mục 3/6 phía trên là
sai**: weight yolov8n/yolov8m thật train trên dataset **10 class**
(`ball, ball-in-basket, number, player, player-in-possession,
player-jump-shot, player-shot-block, referee, rim`, `data.yaml` nội bộ Kaggle),
không phải chỉ class `player`. Riêng class `player`: mAP50 ≈ 0.97,
precision 0.88–0.94, recall 0.94–0.98 (yolov8n @ epoch 86, yolov8m @ epoch 61).
Class `ball` cũng có mặt, mAP50 0.61–0.80.

Đã xác nhận với Phuc (2026-09-17): **giữ nguyên quyết định bỏ
ball/pass/interception** dù weight có hỗ trợ — mục tiêu PRD là chủ động thu
hẹp phạm vi còn 2 chức năng (tracking + heatmap), không phụ thuộc việc weight
có class gì. Phần scaffold ở mục dưới không cần làm lại.

---

## 4. Việc cần Phuc xác nhận trước khi sang Phase 1

1. Weight mới (yolov8m/yolov8n/rtdetr-l) sẽ được đặt ở đâu trong repo, và tên class bên trong có đúng là `"player"` (chữ thường) không?
2. Có đồng ý bỏ hẳn `FrameNumberDrawer`/`SpeedAndDistanceDrawer` khỏi kế hoạch (vì chưa từng tồn tại), hay đây là tính năng Phuc muốn giữ/viết mới?
3. Các lỗi import path/class-name liệt kê ở mục 3 — có phải sửa (dù nằm trong module out-of-scope như `pass_interception_detector`) hay cứ xoá thẳng các module đó nên không cần fix riêng lẻ?

Chờ Phuc xác nhận đã đọc báo cáo này trước khi sang **Phase 1 — Lập kế hoạch adapt** (vẫn chưa code ở Phase 1, chỉ đề xuất phương án + đánh đổi).
