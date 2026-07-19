# Task: Pilot pose pipeline (agent tokens) cho tập con sports của OmniVideo-100K trên JUPITER

**Bối cảnh:** Step A (video tokenization: Seed2/Cosmos/AVC-LM) cho toàn bộ 5,214 video OmniVideo-100K đã xong (xem `JUPITER_STEP_A_TASK.md`). Task này **tách biệt, không phụ thuộc Step A** — mục tiêu là **thử nghiệm (pilot)** áp dụng pipeline pose 3D hiện có của project (HRNet → MotionBERT → kinematics → YOLO cleaner → adaptive PCHIP → merge, dùng để tạo `<agent>` block cho FineVideo-VLA) lên một **tập con** OmniVideo-100K, để đánh giá tính khả thi + chất lượng/số lượng pose thu được trên nguồn video mới này. **Đây là pilot/test, không phải lệnh chạy full production** — quyết định mở rộng full-scale để sau khi xem kết quả pilot.

**Vì sao chỉ tập con, không phải cả 5,214 video:** đã phân loại nội dung toàn corpus bằng keyword-match trên field `video_summary` — chỉ ~24% (1,256/5,214) là nội dung sports/hoạt động thể chất thật (bóng rổ, bóng đá, nhảy, boxing, gym, vật...), phần còn lại chủ yếu talking-head/tin tức/cartoon/gambling — motion thấp hoặc không có người, chạy pose pipeline lên sẽ lãng phí GPU. Xem mục 1.

**Về giới hạn bàn tay (đã thống nhất với user, không cần giải quyết trong task này):** HRNet hiện dùng (`td-hm_hrnet-w48_8xb32-210e_coco-256x192`) là **COCO-17, body-only** — không có keypoint ngón tay, điểm xa nhất chi trên chỉ là cổ tay. Pilot này **sẽ không cải thiện phần bàn tay** dù chọn video sports — đó là giới hạn kiến trúc của Phase 1, không phải do nguồn video. User đã xác nhận: ổn, tay khả năng cao sẽ là dataset/effort riêng sau. Mục tiêu pilot này chỉ là **đa dạng hoá chuyển động toàn thân/cánh tay** (arm/body motion) ngoài phổ nội dung lifestyle/vlog của FineVideo.

---

## 1. Danh sách video đã chọn sẵn (làm trên JUWELS, đã có trong repo)

`data_prep/omnivideo_100k/select_sports_subset.py` — script phân loại, đọc field `video_summary` trong `omnivideo_100k_segment_captions.jsonl`, regex match các từ khoá:
```
basketball|soccer|football|boxing|dance|dancing|gym|workout|running|
fight|fighting|wrestl|tennis|martial art|gymnast|athlete
```
Đây là **heuristic thô dựa trên text summary**, không phải kiểm tra hình ảnh thật — có thể sai dương (video tag "sports" nhưng thực chất commentary tĩnh) hoặc sai âm. Nếu pilot cho thấy tỷ lệ detect người thấp, cân nhắc lọc thêm ở cấp `segments[].caption` (chi tiết hơn theo từng đoạn ~11-16s, tránh cả đoạn intro/disclaimer không có người).

**Kết quả chạy thật:** 1,256/5,214 video (24.1%) → đã ghi ra `data_prep/omnivideo_100k/sports_subset_video_ids.txt` (1 video_id/dòng, không đuôi `.mp4`). File này **đã có trong repo** — nếu JUPITER pull cùng repo git, chỉ cần `git pull`; nếu không, cần copy thủ công (file nhỏ, text thuần, ~15KB).

## 2. Dữ liệu cần có trên JUPITER

- **Video mp4**: đã có sẵn tại `$DATA/omnivideo_100k/videos/*.mp4` (5,214 file, 49GB) — đã move từ `/p` sang trong lúc làm Step A, không cần chuyển lại. Chỉ cần đúng 1,256 file trong `sports_subset_video_ids.txt` là dùng được, phần còn lại không đụng tới.
- **Không cần** file JSONL caption/speech (`omnivideo_100k_segment_captions.jsonl`) cho task này — đó là phụ trợ cho Step A/video-token, pose pipeline không dùng.
- **Cần**: `sports_subset_video_ids.txt` (mục 1).

## 3. ⚠️ QUAN TRỌNG — `pipeline_pose/phase1_hrnet_gpu.py` KHÔNG dùng thẳng được, đọc kỹ trước khi bắt đầu

Giống hệt bài học ở Step A với `pipeline_video/pipeline.py`: `phase1_hrnet_gpu.py` hiện tại **hard-code đọc video từ FineVideo HF arrow dataset**:
```python
DATASET_PATH = "/e/scratch/reformo/nguyen38/finevideo_disk"
dataset = load_from_disk(DATASET_PATH)
...
video_bytes = item.get('mp4')          # bytes nhúng sẵn trong arrow dataset
vid_id = raw.get("original_video_filename", ...)
```
và partition theo `cached_video_ids.json`. OmniVideo-100K chỉ là **file mp4 phẳng trên disk**, không có arrow dataset — không thể trỏ thẳng script này vào folder video mới.

**Cách đúng — viết driver mới** (đừng sửa `phase1_hrnet_gpu.py` gốc, giữ nguyên để tracking):
- Import/copy lại phần **model-agnostic** của file gốc (dòng 15-54: load `pose_model`/`det_model`, hàm `coco_to_h36m()`) — phần này KHÔNG phụ thuộc FineVideo, giữ nguyên logic.
- Thay phần đọc input: mở trực tiếp `cv2.VideoCapture(f"{VIDEOS_DIR}/{video_id}.mp4")` cho từng `video_id` trong `sports_subset_video_ids.txt`, thay vì `load_from_disk` + `item.get('mp4')`.
- Sharding: dùng pattern đơn giản `video_list[RANK::WORLD_SIZE]` (giống driver Step A `step_a_tokenize_video.py`), không cần `--offset/--total_workers` phức tạp như bản gốc (bản gốc thiết kế cho 200 worker/40K video, quy mô nhỏ hơn nhiều ở đây — 1,256 video).
- Output: giữ đúng format `outputs/2d_json/{video_id}_2d.json` (list `{frame_id, keypoints: [[x,y,conf]×17]}`) để Phase 2 (MotionBERT) đọc được mà không cần sửa gì.

**Các phase sau (2–6), kiểm tra trước khi giả định dùng thẳng được — đừng tin theo quán tính:**
- Phase 2 (`phase2_motionbert_gpu.py`) — đọc theo `video_id` từ `outputs/2d_json/`, khả năng cao dùng thẳng được vì làm việc theo file, không theo dataset — **verify bằng cách đọc code trước khi chạy**, không giả định.
- Phase 3 (kinematics) / Phase 4 (YOLO cleaner) / Phase 5 (adaptive PCHIP) — cùng cách kiểm tra: các script này thao tác theo `video_id`/file, nhiều khả năng tái dùng được thẳng, nhưng đọc qua I/O assumption của từng file trước khi chạy hàng loạt.
- Phase 6 (`phase6_merge_adaptive.py`) — **gần như chắc chắn KHÔNG dùng thẳng được**: script này merge `<agent>` block vào `training_ready_rank_*.jsonl` theo cấu trúc `scenes[].activities[]` của FineVideo — OmniVideo-100K không có cấu trúc đó (1 video = 1 record phẳng, giống Step A). Cần viết script merge mới, tinh thần giống `flatten_step_a_video.py` (đã có trong repo) nhưng chèn thêm `<agent>...</agent>` vào đúng vị trí trong text đã flatten của OmniVideo-100K (`/p/.../omnivideo_100k_video_flattened/` phía JUWELS — cần đưa file agent-token pose từ JUPITER trở lại JUWELS để merge, hoặc merge ngay trên JUPITER nếu có cả 2 dữ liệu ở đó, tuỳ hạ tầng — quyết định lúc làm).

## 4. Môi trường — dùng đúng env pose, KHÔNG dùng env_stable_vla

Pipeline pose dùng `env_motion_final` (khác hẳn `env_stable_vla` của Step A/video-token):
```bash
source /e/project1/reformo/nguyen38/finevideo-vla/setup_motionbert.sh
```
Xem thêm CLAUDE.md mục "Environments" nếu cần cả 2 env trong cùng phiên làm việc.

## 5. Lộ trình đề xuất (thận trọng — pilot nhỏ trước, học từ bài học Step A)

Step A từng gặp 3 bug thật khi vội submit full-scale ngay (2 bug seed2 + 1 bug tràn quota đĩa) — pilot nhỏ trước đã cứu được. Đề xuất áp dụng lại đúng cách:

1. **Pilot cực nhỏ** (~20-30 video đầu trong `sports_subset_video_ids.txt`, 1 node) — verify driver Phase 1 mới chạy đúng, kiểm tra tỷ lệ detect người thật trên nội dung sports (video đông người/chuyển động nhanh có thể làm Faster-RCNN detect nhầm/sót — cần xem log thật, không chỉ tin "chạy không lỗi").
2. Nếu ổn, chạy Phase 1 cho toàn bộ 1,256 video, rồi lần lượt Phase 2→5 (đã verify I/O từng phase theo mục 3).
3. Viết script merge mới cho Phase 6-equivalent (mục 3, ý cuối) — sau khi có agent-token thật, không viết trước khi có dữ liệu để test.
4. Dừng lại ở đây, báo cáo kết quả pilot (số video có pose hợp lệ / tổng, chất lượng mẫu) trước khi quyết định có mở rộng thêm hay không.

## 6. Việc KHÔNG nằm trong scope task này

- Cải thiện keypoint bàn tay/ngón tay (giới hạn kiến trúc HRNet-COCO17 hiện tại, để dành dataset/effort riêng sau).
- Chạy pose pipeline lên phần còn lại của OmniVideo-100K (76% không phải sports) — không có giá trị theo phân tích mục intro.
- Quyết định tỷ lệ trộn với FineVideo-VLA lúc train — để sau khi có agent-token thật.
