> **CẬP NHẬT 19/07/2026 (chiều) — pilot Phase 1 đã chạy that (24 video, job `976467`, COMPLETED, 0 loi), 78.7% frame co detect nguoi. Doc truoc khi tiep tuc.**
> Driver da viet: `data_prep/omnivideo_100k/pose/phase1_hrnet_omnivideo.py` (+ `submit_phase1_pilot.sbatch`) — doc mp4 phang truc tiep, GIU NGUYEN confidence lien tuc (khong binarize nhu ban goc).
> 2 bug ha tang phat hien + fix truoc khi chay duoc: symlink `outputs/` bi dut (da tro lai `/e/data1/.../nguyen38/outputs/`), path env sai trong `setup_hrnet_gpu.sh`/`setup_motionbert.sh` (da sua tro `/e/data1/.../3d-human-pose/env_*`).
> **Dieu tra 2 video ty le detect thap trong pilot -> phat hien filter goc (`select_sports_subset.py`) co sai duong voi noi dung animation** (tu khoa chung chung nhu "dancing"/"running" khop nham video commentary/hoat hinh). Nghiem trong hon ca video 0% detect: 1 video animation trong pilot (`Ncl93lkMpJM`, phim nhac hoat hinh khung long) van ra **56.3% detect** — HRNet/Faster-RCNN co the nham nhan vat hoat hinh thanh nguoi that, MotionBERT (train tren Human3.6M nguoi that) lift len se ra pose 3D SAI chu khong chi la thieu du lieu. Da viet `filter_animation_content.py` loc tiep tren 1,256 video: loai 130 video co video_summary tu nhan la animation/cartoon → con **1,126 video** trong `sports_subset_video_ids_filtered.txt` (driver da doi default sang file nay). Luu y: filter nay CHUA bat het — video hoat hinh kieu nhan vat co ten rieng ("Redhead Girl"...) khong tu nhan "animated" van lot qua, chi co the dua vao ty le detect that tu Phase 1 de loc not.
> Theo quyet dinh cua user, **da dung lai o pilot 24 video de xem xet, CHUA mo rong full-scale** — xem `PROGRESS_VI.md`/`REPORT.md` muc "§26" de biet chi tiet day du + trang thai job moi nhat truoc khi quyet dinh buoc tiep theo (vd: chay lai pilot nho tren `sports_subset_video_ids_filtered.txt` de kiem tra filter moi, hay mo rong thang full 1,126).
>
> **⚠️ BAT BUOC truoc khi chay Phase 2.5 cho OmniVideo-100K (chua can lam ngay o Phase 1):** pipeline hien tai giu dung convention cu — Phase 1 (`phase1_hrnet_omnivideo.py`) va Phase 2 (MotionBERT) chay o **native fps** cua tung video (khong ep 30fps), roi Phase 2.5 (`phase2_5_resample_30fps.py`) moi noi suy ve 30fps dung nhat, dua vao `outputs/fps_lookup.json`. Da verify that: OmniVideo-100K co fps goc KHONG dong nhat (vd `07WqS-ccIrw`/`0OxHEDu5dFE` la 25fps that, `0GPO9qLraB8`/`iGVvChGEQdM` la 30fps that, do bang `cv2.CAP_PROP_FPS`). `fps_lookup.json` hien tai (43,751 entry) **chi co video FineVideo, chua co video_id nao cua OmniVideo-100K**. Code Phase 2.5 tu ghi ro trong docstring: "Videos missing from fps_lookup.json are skipped with a warning" — neu chay thang ma khong bo sung, **toan bo video OmniVideo-100K se bi am tham bo qua o buoc 30fps**, mat trang du Phase 1-2 da chay dung. Truoc khi chay Phase 2.5: chay `tools/extract/extract_fps.py` tren `$DATA/omnivideo_100k/videos/`, roi **merge** (khong ghi de) ket qua vao `outputs/fps_lookup.json` chung (dung symlink `outputs/` -> `/e/data1/.../nguyen38/outputs/`, da fix trong phien nay — xem REPORT.md §26).

---

# Task: Pilot pose pipeline (agent tokens) cho tập con sports của OmniVideo-100K trên JUPITER

**Bối cảnh:** Step A (video tokenization: Seed2/Cosmos/AVC-LM) cho toàn bộ 5,214 video OmniVideo-100K đã xong (xem `JUPITER_STEP_A_TASK.md`). Task này **tách biệt, không phụ thuộc Step A** — mục tiêu là **thử nghiệm (pilot)** áp dụng pipeline pose 3D hiện có của project (HRNet → MotionBERT → kinematics → YOLO cleaner → adaptive PCHIP → merge, dùng để tạo `<agent>` block cho FineVideo-VLA) lên một **tập con** OmniVideo-100K, để đánh giá tính khả thi + chất lượng/số lượng pose thu được trên nguồn video mới này. **Đây là pilot/test, không phải lệnh chạy full production** — quyết định mở rộng full-scale để sau khi xem kết quả pilot.

**Vì sao chỉ tập con, không phải cả 5,214 video:** đã phân loại nội dung toàn corpus bằng keyword-match trên field `video_summary` — chỉ ~24% (1,256/5,214) là nội dung sports/hoạt động thể chất thật (bóng rổ, bóng đá, nhảy, boxing, gym, vật...), phần còn lại chủ yếu talking-head/tin tức/cartoon/gambling — motion thấp hoặc không có người, chạy pose pipeline lên sẽ lãng phí GPU. Xem mục 1.

**Về giới hạn bàn tay (đã thống nhất với user, không cần giải quyết trong task này):** HRNet hiện dùng (`td-hm_hrnet-w48_8xb32-210e_coco-256x192`) là **COCO-17, body-only** — không có keypoint ngón tay, điểm xa nhất chi trên chỉ là cổ tay. Pilot này **sẽ không cải thiện phần bàn tay** dù chọn video sports — đó là giới hạn kiến trúc của Phase 1, không phải do nguồn video. User đã xác nhận: ổn, tay khả năng cao sẽ là dataset/effort riêng sau. Mục tiêu pilot này chỉ là **đa dạng hoá chuyển động toàn thân/cánh tay** (arm/body motion) ngoài phổ nội dung lifestyle/vlog của FineVideo.

---

## 1. Danh sách video đã chọn sẵn (làm trên JUWELS, đã có trong repo)

`data_prep/omnivideo_100k/dataset_prep/select_sports_subset.py` — script phân loại, đọc field `video_summary` trong `omnivideo_100k_segment_captions.jsonl`, regex match các từ khoá:
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
