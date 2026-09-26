# PRD — Adapt cloned repo thành Player Tracking + Heatmap System

**Bối cảnh**: Repo cũ đã bị bỏ. Giữ lại: notebook training + file weight
`.pt` của các model đã train (yolov8m, yolov8n, rtdetr-l). Codebase mới =
clone từ `https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1` và sửa lại
theo mục tiêu hẹp hơn nhiều so với repo gốc.

**Quy tắc làm việc bắt buộc (đọc trước khi làm bất cứ gì khác)**:
Phuc cần được tham vấn về kiến trúc trước khi viết code, và làm việc theo
từng phiên có xác nhận rõ ràng trước khi qua bước tiếp theo — KHÔNG tự ý viết
code cho nhiều module cùng lúc mà chưa được duyệt plan. Ở mọi bước có quyết
định kiến trúc (giữ hay bỏ module, đổi tracker, đổi cách team-assign...),
trình bày phương án + đánh đổi, rồi DỪNG LẠI chờ xác nhận, không tự quyết
định và code tiếp.

---

## 1. Mục tiêu

Thu hẹp phạm vi repo gốc (vốn có: detection bóng+cầu thủ, tracking, team
classification, court keypoint detection, phân tích pass/interception/
possession, speed/distance) xuống còn đúng 2 chức năng cốt lõi:

1. **Track từng cầu thủ** — ID nhất quán xuyên suốt clip, chịu được camera
   pan liên tục (broadcast footage) và occlusion giữa các cầu thủ.
2. **Tạo heatmap workrate cho từng cầu thủ VÀ từng đội** — thể hiện vùng hoạt
   động/cường độ di chuyển, ánh xạ lên toạ độ sân thực (top-down), không phải
   toạ độ pixel thô.

## 2. Ngoài phạm vi (Out of scope — cân nhắc loại bỏ khỏi repo gốc)

- Ball detection/tracking.
- Pass/interception detection (`pass_interception_detector/`).
- Ball possession analysis (`ball_acquisition/`).
- Bất kỳ phân tích nào phụ thuộc vào việc detect bóng.

Speed/distance calculation (`speed_and_distance_calculation/`) — **giữ lại**,
gộp chung logic với heatmap module vì cùng nguồn dữ liệu (lịch sử vị trí
court-space theo track_id).

## 3. Assets mang từ dự án cũ sang

- Notebook training gốc (EDA, training cells) — dùng để tham chiếu, không
  cần chạy lại.
- Weight `.pt` đã train: `yolov8m`, `yolov8n`, `rtdetr-l` — train trên
  Roboflow NBA player detector dataset, **chỉ có class player, KHÔNG có
  class ball**. Đây là điểm khác biệt quan trọng với model gốc của repo
  (repo gốc detect cả player + ball).

## 4. PHASE 0 — Discovery (bắt buộc làm trước, KHÔNG code gì ở bước này)

Trước khi sửa bất kỳ file nào, khảo sát toàn bộ repo vừa clone và báo cáo lại
cho Phuc:

1. Liệt kê đầy đủ cấu trúc thư mục thật (`find . -type f -not -path
   "*/\.git/*"`).
2. Với từng module sau, đọc code và trả lời rõ:
   - `trackers/` — đang dùng tracker gì (ByteTrack/BoT-SORT/khác)? Có xử lý
     camera motion compensation không? Có Re-ID/appearance matching không?
   - `team_assigner/` — cách hoạt động thật (README nói dùng zero-shot
     Fashion CLIP) — có cần API key/internet để chạy không? Tốc độ thế nào?
   - `Court_keypoint_detection/` — có **weight đã train sẵn** cho keypoint
     model không, hay chỉ có code + cần tự train? Nếu cần train, dataset
     keypoint có kèm theo repo không?
   - `tactical_view/` — cách vẽ top-down view hiện tại hoạt động ra sao,
     input/output format là gì.
   - `drawers/` — những gì đang được vẽ lên video output.
   - Model detection gốc của repo — class nào (`player`, `ball`,...), input/
     output format, có tương thích trực tiếp với weight `.pt` mới (chỉ có
     class player) không, hay cần sửa code xử lý output.
3. Chạy thử pipeline gốc (`main.py` hoặc entry point tương đương) trên 1
   video mẫu có sẵn trong `input_videos/` (nếu có) để xác nhận repo chạy
   được trước khi sửa gì — báo cáo lỗi nếu có.
4. Tổng hợp báo cáo Phase 0 thành 1 file `DISCOVERY_REPORT.md`, KHÔNG code gì
   thêm, trình bày cho Phuc và **chờ xác nhận** trước khi sang Phase 1.

## 5. PHASE 1 — Lập kế hoạch adapt (chờ duyệt trước khi code)

Dựa trên kết quả Phase 0, đề xuất (không code) phương án cho từng điểm quyết
định sau — với mỗi điểm, nêu rõ **đánh đổi**, không tự chọn:

- **Detection**: giữ nguyên cấu trúc `training/` của repo hay thay hẳn bằng
  weight `.pt` mới? Nếu class mismatch (repo detect cả ball), cần sửa code
  xử lý output ở đâu?
- **Tracker**: nếu repo đang dùng ByteTrack thuần (theo README), đề xuất
  nâng cấp sang BoT-SORT + Camera Motion Compensation (`gmc_method:
  sparseOptFlow`) + Re-ID — đây là kinh nghiệm rút ra từ dự án cũ: ByteTrack
  không xử lý được camera pan liên tục, gây ID fragmentation nghiêm trọng
  (1 cầu thủ mang nhiều ID). Cần xác nhận lại: repo gốc có sẵn cấu hình
  BoT-SORT chưa hay chỉ có ByteTrack.
- **Team assignment**: repo dùng zero-shot CLIP (Fashion CLIP qua Hugging
  Face) — cân nhắc giữ nguyên (nếu chạy ổn, không cần internet/API phức
  tạp) hay thay bằng unsupervised K-means color clustering (cách đã làm ở
  dự án cũ, không cần model ngoài, nhẹ hơn). Quyết định dựa trên: tốc độ,
  độ ổn định thực tế khi test, và có cần offline không.
- **Court mapping**: NẾU repo đã có keypoint model + weight train sẵn — đây
  là giải pháp tốt hơn nhiều so với cách "propagation qua optical flow" đã
  cân nhắc ở dự án cũ (dự án cũ chọn propagation vì thiếu thời gian train
  keypoint model — nếu giờ có sẵn, nên dùng luôn, không cần propagation).
  NẾU repo chỉ có code chưa có weight, cân nhắc lại giữa: tự train keypoint
  model (cần dataset + thời gian) vs quay lại phương án propagation.
- **Heatmap module**: xác nhận vị trí đặt (module mới, không có trong repo
  gốc) — đề xuất tận dụng lại output của `tactical_view/` (đã có toạ độ
  court-space) làm input, tránh làm lại phần homography.

Trình bày phương án cho từng điểm, đợi Phuc chọn, rồi mới sang Phase 2.

## 6. PHASE 2 — Implementation (chỉ làm sau khi Phase 1 được duyệt)

- Xoá/disable các module ngoài phạm vi (mục 2).
- Sửa detection để dùng weight `.pt` mới (chỉ class player).
- Áp dụng các quyết định đã chốt ở Phase 1 cho tracker/team assigner/court
  mapping.
- Viết module heatmap mới: tổng hợp vị trí theo track_id + team_id, xuất
  heatmap ảnh cho từng cầu thủ và cho từng đội.
- Cập nhật `main.py`/entry point để pipeline chạy end-to-end: video input →
  video output đã annotate + heatmap ảnh riêng (không phải live dashboard —
  xem mục 7).

## 7. Output mong đợi

- Video output đã annotate (bbox, ID, team color) — theo đúng style repo
  gốc (`output_videos/`).
- Heatmap ảnh: 1 ảnh tổng hợp toàn trận cho mỗi cầu thủ, 1 ảnh cho mỗi đội.
- (Không làm live dashboard trong repo mới này — nếu muốn thêm lại, đó là
  quyết định cần bàn riêng, không mặc định.)

## 8. Tiêu chí hoàn thành Phase 0

Phase 0 coi là xong khi Phuc nhận được `DISCOVERY_REPORT.md` trả lời đầy đủ
các câu hỏi ở mục 4, và pipeline gốc của repo đã được xác nhận chạy được
(hoặc báo lỗi cụ thể nếu không chạy được) trên ít nhất 1 video mẫu.
