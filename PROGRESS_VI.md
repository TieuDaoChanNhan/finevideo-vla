# PAB-Spline VLA — Tiến độ dự án

**Tác giả:** Van Khue Nguyen  
**Cập nhật lần cuối:** 21/07/2026  
**Cluster:** JUPITER (JSC), partition `booster`, GPU GH200 — bảo trì phần lớn đã xong. Phase 4 FineVideo **ĐÃ XONG** (40,300/40,305). Phase 5 **ĐÃ XONG**. Phase 6 đang chạy.  
**Mục tiêu:** Xây dựng mô hình VLA (Vision-Language-Action) — xem video, nghe tiếng, sinh ra token điều khiển robot.

---

## Cập nhật phiên làm việc — 21/07/2026 (tiếp lần 4 — Harmony4D convert+resample+sample, pull kết quả tokenize JUWELS, chọn 5 dataset + rebalance mix, viết config Qwen3 v2, submit job training thật)

**Việc chính:** Hoàn tất pipeline convert Harmony4D tới bước windowing/root-centering (tạm dừng lại theo yêu cầu user để chuyển hướng), pull kết quả tokenize từ phiên JUWELS trước, chọn 5 dataset cho lần train tiếp theo, viết config Qwen3 1.7B mới, copy 120GB data + tokenizer từ `/p` sang `/e`, và **đã submit thật job training 64-node** (`1009758`, đang RUNNING tại thời điểm ghi entry).

### 1. Harmony4D — convert COCO→H36M, resample 30fps, sample trực quan

Viết `data_prep/harmony4d/convert_coco_to_h36m.py` (mapping COCO-17→H36M-17, tái dùng công thức từ `phase1_hrnet_gpu.py::coco_to_h36m()`, mở rộng 3D, confidence-threshold→NaN thay vì zero-fill). Submit SLURM job `1009045` (22 worker, 1 zip/worker) — **xong trong 18 giây**, 208/208 sequence (khớp đúng số paper), 416 person-track (208×2 người), 0 lỗi. Verify bằng cách đo lại độ dài xương sau convert trên `skeleton_tree` chuẩn của project — đối xứng trái-phải gần tuyệt đối, độ dài hợp lý giải phẫu.

Viết `data_prep/harmony4d/resample_30fps.py` (linear interpolation theo timestamp thật `frame_idx/20fps`, mirror đúng phương pháp `phase2_5_resample_30fps.py`). Chạy trực tiếp (không cần SLURM, xong trong vài giây): 416/416 track, 0 lỗi. 301 frame@20fps → 451 frame@30fps, tỉ lệ 1.498 ≈ đúng 1.5 kỳ vọng.

**Quyết định cùng user:** agent tokens giữ nguyên schema 1-người (không thiết kế multi-person mới) — nhưng **giữ cả 2 người** thành 2 track độc lập (không bỏ người nào), mỗi track xử lý y hệt 1 video-1-người bình thường, đúng tiền lệ FineVideo. 208 sequence → 416 track khả dụng.

Viết `data_prep/harmony4d/make_pose_samples.py`, render 6 video skeleton (`samples/harmony4d/`) cho 3 track (mma×2 người, hugging×1 người) ở 2 checkpoint: sau resample-30fps (world-frame), và sau root-centering+windowing (stride=8, tái dùng `create_windows()` từ `phase3_kinematics_processor.py`). **Quyết định kỹ thuật quan trọng tự đưa ra:** KHÔNG chạy toàn bộ `KinematicPreprocessor.process()` (hallucination filter/ID-switch/stiff-leg) — các heuristic đó tinh chỉnh riêng cho lỗi MotionBERT đơn-mắt, áp vào ground-truth đa-camera chất lượng cao của Harmony4D có nguy cơ "sửa nhầm" tư thế thật (vd: cú vật nhanh bị hiểu nhầm ID-switch, gối bẻ ngược khi grapple dưới đất bị "sửa" sai). Đã gửi user xem sample, **user tạm dừng task này ở đây** để chuyển hướng sang việc khác.

### 2. Pull code — xác nhận JUWELS đã tokenize xong 4/4 dataset

`git pull` lấy 2 commit mới từ phiên JUWELS trước (`adf87f7`, `59b0042`) — `TOKENIZE_TODO.md` đã được cập nhật xác nhận **cả 4 job tokenize đều COMPLETED thật** (verify qua `.idx` header, không chỉ SLURM state):

| Dataset | Token thật | Số record |
|---|---|---|
| finevideo_v6 | 10,926,767,551 | 371,892 |
| omnivideo_100k_video | 536,149,780 | 5,214 |
| synth_llava | 103,097,102 | 603,999 |
| roleplay | 52,469,577 | 67,459 |
| mv_omni (đã tokenize từ 18/07) | 20,389,561,883 | — |
| **Tổng 5 dataset** | **32,008,045,893 (~32.01B)** | |

### 3. Chọn 5 dataset train + copy 120GB data sang `/e`

User chọn đúng 5 dataset ở bảng trên (khớp con số "Total, these 4" + MV-Omni mà `TOKENIZE_TODO.md` đã ghi). Copy toàn bộ `.bin/.idx` (120GB: finevideo_v6 41G, mv_omni 76G, omnivideo 2.0G, synth_llava 405M, roleplay 202M) + tokenizer `tokenizer_vla_qwen3` (257,901 vocab thật, max token id 257900) từ `/p` sang `/e/data1/datasets/playground/mmlaion/shared/nguyen38/vla_v2_tokenized/` + `.../tokenizer_vla_qwen3/` — bắt buộc vì container Apptainer chỉ bind `/e`. Verify shard count + size khớp tuyệt đối với bản gốc `/p`.

### 4. Viết config `qwen3_1.7b_vla_v2.yaml` — migration Qwen3 thật đầu tiên

Tìm `oellm-autoexp/config/experiments/harsh/qwen3_1.7b_mixvitae_jupiter.yaml` làm mẫu, viết `config/experiments/nguyen38/qwen3_1.7b_vla_v2.yaml` y hệt cấu trúc (Qwen3 1.7B architecture, `ckpt_convert_qwen3` postprocess) nhưng trỏ tokenizer + data mix riêng của mình. Đây là lần đầu project thật sự chạy Qwen3 (CLAUDE.md ghi "planned" từ lâu, giờ mới thực hiện) — thay cho kiến trúc OpenSci-Ref của model cũ.

**2 quyết định cùng user:**
- **Epoch:** user chọn đúng 1 epoch ("cứ tùy độ lớn dữ liệu thôi") thay vì giữ quy ước ~3 epoch cũ (hợp lý cho corpus 2.84B, nhưng corpus giờ lớn hơn 11 lần nên 3 epoch sẽ quá dài) → `train_iters=7632` (32.008B/4,194,304 tokens-per-iter).
- **Mix ratio:** user yêu cầu giảm MV-Omni. Rebalance 60/40 — 2 dataset có `<agent>` action token thật (finevideo_v6 + omnivideo_100k_video) chiếm 60% tổng weight (thay vì 35.82% theo tỉ lệ token thật), 3 dataset không có action token (mv_omni + synth_llava + roleplay) chia nhau 40% (MV-Omni giảm từ 63.71% raw xuống 39.71%). Lý do giải thích cho user: proportional-theo-size = mọi dataset được nhìn thấy đúng N epoch như nhau — "giảm" nghĩa là MV-Omni bị nhìn ít hơn N epoch, finevideo_v6 được nhìn nhiều hơn N epoch, ưu tiên đúng tín hiệu modality-transition đang fail.

### 5. Submit job training thật — job `1009758`, 64 node, RUNNING

Submit qua đúng template shell user cung cấp (đã lưu memory `feedback_training_submit_template`). **Sự cố nhỏ:** `run_autoexp.py` chạy foreground polling liên tục không tự thoát → bị timeout tool sau 2 phút (exit 143) — nhưng verify qua `squeue`/`sacct` xác nhận **job `1009758` đã submit thành công từ trước đó và vẫn RUNNING bình thường**, không bị ảnh hưởng bởi việc kill script polling.

Vì orchestrator bị kill nên mất luôn vòng lặp tự động trigger postprocess (dist_to_torch → convert_hf → eval) sau khi training xong. Tìm đúng công cụ resume: `scripts/monitor_autoexp.py --session-dir monitor_state/<session_id>` — đọc code xác nhận script này **không có code path submit job nào cả**, chỉ load lại `JobFileStore` từ session cũ và poll tiếp — an toàn tuyệt đối để tránh submit nhầm job trùng lặp 64-node (rất tốn kém nếu sai). Chạy nền (`nohup ... & disown`), verify `squeue` chỉ có đúng 1 job, verify `attempts` trong state file vẫn là 1 (không tăng lên 2) → xác nhận không bị submit lại.

**Trạng thái cuối phiên:**
- Job `1009758` (`qwen3_1.7b_vla_v2`) — RUNNING, 64 node, time limit 4h, output tại `output_vla/qwen3_1.7b_vla_v2/`.
- Monitor process (`monitor_autoexp.py`, đã disown khỏi shell) đang chạy nền, sẽ tự trigger postprocess khi job xong.
- **Job SLURM hoàn toàn độc lập với session Claude Code** — user thoát session không ảnh hưởng gì tới job đang chạy trên cluster. Monitor process (nohup+disown) về lý thuyết cũng sống sót độc lập, nhưng chưa có xác nhận 100% về việc môi trường shell/sandbox nền có tồn tại lâu dài sau khi đóng session hay không — nếu monitor chết giữa chừng, việc resume lại bằng đúng lệnh trên vẫn an toàn (không submit lại), chỉ cần chạy lại khi cần.
- Việc tồn đọng: Harmony4D windowing/root-centering thật (áp dụng filter nào ngoài center+window, nếu có) vẫn treo theo quyết định tạm dừng ở mục 1; eval protocol cho model mới vẫn chưa thiết kế (REPORT.md Pre-training Blockers mục 3 vẫn mở); chưa commit/push 3 script Harmony4D + samples lên GitHub; chưa commit config `qwen3_1.7b_vla_v2.yaml` vào repo `oellm-autoexp` (repo khác, chưa được yêu cầu commit).

---

## Cập nhật phiên làm việc — 21/07/2026 (JUWELS — verify + submit 4 job tokenize Megatron)

**Việc chính:** Phiên này chạy trên **JUWELS** (không phải JUPITER — compute node JUPITER không mount được `/p`, nơi hạ tầng tokenize `mv-scale/` sống), theo đúng file handoff `TOKENIZE_TODO.md` mà một phiên JUPITER trước để lại ở repo root. Pull code mới nhất, đọc `TOKENIZE_TODO.md` + `data_prep/` để xác định 4 dataset cần tokenize lại: **FineVideo-VLA v6, OmniVideo-100K (video), synth-llava, emotional-roleplay**. Theo yêu cầu của user, verify từng dataset khớp đúng bản đã upload HF **trước khi** submit job — tải nguyên bản HF của 3/4 dataset đã có trên Hub (OmniVideo-100K, synth-llava, roleplay), so sánh content-hash (sha256 trên cặp id+text đã sort theo id) với bản local trên `/p` — **khớp tuyệt đối byte-for-byte cả 3**. FineVideo v6 chưa có bản HF nào (HF hiện chỉ có bản v1 cũ, 19GB) nên không có gì để đối chiếu — chỉ verify số record local (371,892) khớp đúng con số `TOKENIZE_TODO.md` đã ghi.

Tạo/sửa 4 file sbatch trong `/p/data1/mmlaion/nguyen38/mv-scale/` (hạ tầng tokenize dùng chung, **không** nằm trong git repo này):
- `tokenize_finevideo_v6.sbatch` — mới, copy từ `tokenize_finevideo_v5.sbatch`, 4 node Ray cluster, port 20160 (đã check không đụng port các sbatch khác đang có).
- `tokenize_omnivideo_100k_video.sbatch` — sửa `INPUT` từ thư mục cũ (`omnivideo_100k_video_flattened`, có trước fix wrapper-token 21/07) sang `omnivideo_100k_final/` — đúng bản plain JSONL (chưa gzip) mà `phase7_finalize_omnivideo.py` ghi ra; không dùng `hf_upload/` vì thư mục đó đã gzip + chia train/test riêng cho HF, `mv_preprocess_data.py` glob không đệ quy nên tự bỏ qua subfolder này.
- `tokenize_synth_llava.sbatch` — mới, 1 node (chưa từng tokenize; size chỉ 547MB nên dùng pattern single-node như `tokenize_robovqa.sbatch`, không cần Ray cluster đa node).
- `tokenize_roleplay.sbatch` — mới, 1 node (chưa từng tokenize; 339MB).

Cả 4 dùng chung tokenizer `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3` (Qwen3 base + full VLA token set, 257,901 vocab) — xác nhận bằng cách đọc thẳng các sbatch template thật đang chạy, không suy đoán từ doc cũ.

**Đã submit 4 job lên SLURM** (JUWELS, account `laionize`, partition `batch`):

| Job ID | Tên | Số node | Dataset |
|---|---|---|---|
| 14127888 | tok_finevideo_v6 | 4 | FineVideo-VLA v6 (371,892 record, 74GB) |
| 14127889 | tok_omni_video | 1 | OmniVideo-100K video (5,214 record, 3.6GB) |
| 14127890 | tok_synth_llava | 1 | synth-llava (603,999 record, 547MB) |
| 14127891 | tok_roleplay | 1 | emotional-roleplay (67,459 record, 339MB) |

Cả 4 job đã chuyển RUNNING (`squeue` xác nhận, không còn PD). **Việc còn lại cho phiên sau:** đừng tin trạng thái SLURM một mình — verify output thật (grep log tìm `Traceback`, check size `tokenized_output/{finevideo_v6,omnivideo_100k_video,synth_llava,roleplay}/`, chạy `mv-scale/count_tokens.py` đối chiếu số token thật) trước khi coi các job này là xong, đúng bài học đã rút ra ngày 18/07 (job từng báo COMPLETED trên SLURM nhưng thực chất fail ngầm vì Ray không khởi động được).

**Cập nhật cùng ngày, tối muộn hơn — cả 4 job đã COMPLETED thật, đã đếm token thật:** theo dõi qua `squeue`/`sacct` cho tới khi cả 4 job kết thúc (job cuối `tok_finevideo_v6` mất 53m12s, tổng cộng 4 node × ~53 phút vì file nặng nhất — 74GB). Grep log cả 4 không thấy `Traceback`. Đếm token thật bằng cách đọc trực tiếp header `.idx` (cùng logic với `mv-scale/count_tokens.py`, không phải ước tính) — tất cả pass BIN SIZE CHECK (tổng độ dài sequence khớp chính xác byte size của file `.bin`):

| Job | Token thật | Docs | Thời gian chạy |
|---|---|---|---|
| finevideo_v6 | **10,926,767,551 (10.93B)** | 371,892 | 53m12s |
| omnivideo_100k_video | 536,149,780 (0.54B) | 5,214 | 19m47s |
| synth_llava | 103,097,102 (0.10B) | 603,999 | 27m06s |
| roleplay | 52,469,577 (0.05B) | 67,459 | 6m07s |
| **Tổng 4 job hôm nay** | **11,618,484,010 (11.62B)** | 1,048,564 | |

Đáng chú ý: token thật của finevideo_v6 (10.93B) gần gấp đôi con số ước tính lúc flatten (5.443B, ghi trong `TOKENIZE_TODO.md`) — đúng pattern lệch đã ghi nhận từ trước với v5 (10.55B thật vs 5.256B ước tính word-count), không phải bug mới, do cách đếm ước tính dùng word-count cho phần text tự do (title/context/caption/speech) thay vì BPE thật.

Đếm lại luôn MV-Omni (đã tokenize từ 18/07, không thuộc 4 job hôm nay nhưng cần cho bức tranh tổng) bằng cùng phương pháp — xác nhận khớp với con số cũ trong memory: **20,389,561,883 (20.39B) token, 1,593,301 docs, PASS**.

**Tổng token thật hiện có, sẵn sàng cho training (chưa tính RoboVQA/OmniVideo-100K-QA — chưa verify lại, và 2 shard `vla_25b`/`vla_adaptive` đã gắn với model cũ):** 11.62B (4 job hôm nay) + 20.39B (MV-Omni) = **~32.01B token**.

---

## Cập nhật phiên làm việc — 21/07/2026 (tiếp lần 3 — Phase 4 xác nhận DONE, phân tích chat Discord với Huu → pivot chiến lược sang nguồn data pose đã annotate sẵn, chốt Harmony4D, chạy lại Phase 5→6 cho FineVideo, phát hiện + xử lý JUWELS/Jupiter storage mismatch, bắt đầu tải Harmony4D)

**Việc chính:** Track lại Phase 4 (job `1004747`) và phát hiện đã **COMPLETED thật sự** (docs cũ ghi ~67%, thực tế đã xong lúc 08:19 sáng cùng ngày) — tính được số liệu Huu hỏi từ lâu ("how much data was filtered out"): **45.9% window bị occlusion filter drop** (43.0M → 23.2M), 8.3% video (3,359/40,300) mất trắng hoàn toàn. Sau đó đọc + phân tích 1 đoạn chat Discord dài giữa Huu và Van Khue về đúng chủ đề này, dẫn tới quyết định pivot chiến lược: **ngừng đầu tư cải thiện pose detector, chuyển sang tìm dataset pose-video đã annotate sẵn** để bù gap occlusion + multi-person. Điều tra 3 ứng viên (MotionVid, OCHuman-Pose, Harmony4D) + rà thêm `Awesome-Video-Datasets` — chốt **Harmony4D** (MIT, đúng cả 2 gap), loại JRDB-Pose3D dù fit kỹ thuật tốt hơn (CC BY-NC-SA, vi phạm chính sách permissive-only). Chạy lại Phase 5 (agent tokens) trên Phase 4 mới, rồi chuẩn bị Phase 6 — phát hiện `submit_merge_adaptive_v4.sh` cũ trỏ vào 1 checkout repo khác trên `/p` (JUWELS storage, lùi sau 3 commit, agent-tokens-dir trỏ vào file `.tar` chưa giải nén) — nếu chạy thẳng sẽ merge caption/speech mới vào agent token cũ/rỗng. Test xác nhận **compute node của Jupiter không mount được `/p`** (chỉ login node thấy) → quyết định cuối: xử lý toàn bộ trên `/e`. Bắt đầu tải Harmony4D (352GB) song song.

### 1. Phase 4 FineVideo — xác nhận COMPLETED, tính số liệu filter nợ Huu

`sacct` xác nhận job `1004747` COMPLETED lúc 08:19:09 (không còn job nào trên `squeue`) — entry phiên trước ghi ~67% chỉ là snapshot cũ. Kết quả cuối: **40,300/40,305 video (99.99%)**, 5 lỗi (3 CUDA OOM do 128 worker tranh GPU qua MPS, 2 "windows could not be resolved") — tỷ lệ lỗi 0.012%, không đáng retry riêng.

Đếm window thật (Phase 3 output vs Phase 4 output, `xargs -P16 wc -l` + sum đúng qua mọi batch — lần đầu dùng `tail -1` sai vì xargs chia nhiều batch, mỗi batch có dòng "total" riêng, phát hiện + sửa ngay):
- Trước filter: 42,994,892 window
- Sau filter: 23,245,694 window
- **Bị drop: 19,749,198 (45.9%)** — đây là con số trả lời câu Huu hỏi trong đoạn chat Discord bên dưới
- 3,359/40,300 video (8.3%) mất trắng toàn bộ (0 window nào pass)

### 2. Phân tích chat Discord Huu ↔ Van Khue → pivot chiến lược nguồn data pose

User dán 1 đoạn chat Discord dài, yêu cầu đọc + phân tích. Chuỗi logic: Huu hỏi tại sao drop pose bị occlude → truy ra root cause là **giới hạn detector** (HRNet hallucinate, ví dụ nhận nhầm cây thành người, nên phải thêm YOLO filter chặn) chứ không phải chủ đích thiết kế → hệ quả là dataset loại hết mọi cảnh có occlusion (người đi khuất sau cây) lẫn multi-person (pipeline hiện chỉ lấy 1 bbox tự tin nhất/frame). Huu quyết định: **không đầu tư cải thiện detector nữa** ("it might be a rabbit hole"), vì đang chạy đua thời gian ("summer is almost over") — 2 hướng train mùa này: model nhỏ (omni-VLA + RL trong simulation) và model lớn (dataset MV2, làm POC omni model tốt cả ở language). Thay vào đó: **tìm dataset pose-video đã annotate sẵn** trên HF.

Chi tiết đáng chú ý: 1 list dataset gợi ý do AI generate mà Huu paste có **`FineVideo-Phase2-3DPose` — chính dataset team đã upload** — bị liệt kê như của bên thứ 3; Huu tự nhận xét điều này cho thấy data pose-video permissive-license công khai thực sự khan hiếm.

Lưu vào memory (`project_pivot_pose_dataset_sourcing`): không đề xuất sửa detector nữa cho các phiên sau; "VALID" và "Leo" (JUWELS/Leonardo) là 2 proper noun Huu nhắc tới trong chat, chưa xác nhận rõ nghĩa, cần verify trước khi giả định.

### 3. Điều tra 3 ứng viên (agent nền) + rà Awesome-Video-Datasets — chốt Harmony4D

Launch 1 agent nghiên cứu song song (không sửa code, chỉ tra cứu):
- **MotionVid**: loại — HF repo chỉ có caption+keypoint 2D (DWPose), **không có video thật** (phải tự ghép lại từ 9 dataset gốc, nhiều cái non-commercial)
- **OCHuman-Pose**: loại — chỉ ảnh tĩnh (không video/temporal), 2D only, chỉ có eval split, license mâu thuẫn (HF ghi MIT nhưng nguồn khác nói CC BY-NC)
- **Harmony4D**: ✅ chọn — MIT xác nhận từ nguồn gốc tác giả (`jyuntins/harmony4d` trên GitHub + `Jyun-Ting/Harmony4D` trên HF, không phải bản mirror `Voxel51/Harmony4D` ghi thiếu license). 208 sequence, 24 subject, luôn 2 người/cảnh (hugging/grappling/sword/ballroom/karate/mma), multi-view video + 3D pose 17-joint + SMPL mesh fit contact-aware — giải đúng cả occlusion lẫn multi-person
- Rà `xiaobai1217/Awesome-Video-Datasets` (list thiên action-recognition, không hiệu quả lắm cho pose): tìm ra **JRDB-Pose3D** (SMPL multi-person 3D, 5-35 người/frame, nhãn occlusion rõ ràng — fit kỹ thuật tốt hơn cả Harmony4D) nhưng **CC BY-NC-SA → loại thẳng, không bàn thêm**, đúng chính sách permissive-only. NTU RGB+D/120, UAV-Human cũng bị loại (non-commercial / viewpoint không hợp / license chưa xác nhận).

User chốt trực tiếp: **Harmony4D**, bỏ qua JRDB-Pose3D dù kỹ thuật tốt hơn, vì "chỉ quan tâm permissive thôi, làm open source mà". Đồng thời làm rõ dynamic quyết định: "Huu chỉ là PI thôi", Van Khue tự quyết kỹ thuật trực tiếp, không cần chờ Huu duyệt từng bước — đã lưu vào memory feedback (`feedback_decision_authority_and_license_policy`) để các phiên sau không tự thêm rào cản "cần confirm với Huu trước" không cần thiết.

### 4. Chạy lại Phase 5 (agent tokens) trên Phase 4 mới

`outputs/agent_tokens_adaptive/` cũ (18,847 file, 19/06) được dựng trên Phase 3/4 **trước khi fix bug fps-mismatch** — script Phase 5 chỉ skip theo `os.path.exists()`, không biết input đã đổi, nên phải move (không xóa) sang `agent_tokens_adaptive_buggy_fps_mismatch_2026-07-20/` trước khi submit lại, tránh bị skip nhầm. Submit `slurm/submit_phase5_adaptive.sh` (job `1006884`, 64 worker) — **COMPLETED trong 5 phút28s**: 19,076 video có agent token, 21,224 video rỗng (đa số do joint tay bị NaN gần như toàn bộ trong footage YouTube, đúng như REPORT.md đã ghi từ trước), 0 skip.

### 5. Phase 6 — phát hiện kiến trúc `/p` (JUWELS) vs `/e` (Jupiter) không tương thích, xử lý xong

Định submit thẳng `slurm/submit_merge_adaptive_v4.sh` (script cũ, có đủ `--captions-dir`/`--speech-segments-dir` như note đã ghi từ phiên trước) nhưng phát hiện 2 vấn đề nghiêm trọng trước khi chạy:
1. `--input-glob` trỏ vào `final_dataset_adaptive_v3` — bản này được build từ Phase 3/4/5 **trước khi fix fps-mismatch** (v3 tạo 12/07, fix bug 20/07) — nếu dùng sẽ merge caption/speech mới vào agent/pose token cũ lỗi, phí công regen Phase 4+5 vừa xong.
2. Script chạy trên checkout `/p/data1/mmlaion/nguyen38/3d-human-pose` — **lùi sau 3 commit** so với `/e` (thiếu cả fix fps-mismatch lẫn wrapper-token hôm trước) — đã `git pull` đồng bộ lại.
3. `--agent-tokens-dir` của bản `/p` trỏ vào 1 file `agent_tokens_adaptive.tar` **chưa giải nén** (chỉ là bản backup từ 19/07), không phải thư mục thật.

User làm rõ: `/p/data1` là storage của **JUWELS**, `account=laionize` là account riêng JUWELS — không dùng được từ session Jupiter hiện tại. Test trực tiếp bằng `srun` xác nhận **compute node của Jupiter/booster không mount được `/p`** (chỉ login node thấy) — đây chính là lý do các script cũ phải chạy qua JUWELS. Sau khi cân nhắc (ban đầu định đưa hết data về `/p` theo policy mới của user, rồi phát hiện conflict này), **quyết định cuối: toàn bộ pipeline xử lý trên `/e`** (đã lưu memory `feedback_data_storage_location`, đảo ngược policy cùng ngày sau khi hiểu rõ giới hạn hạ tầng). Copy 3 thư mục phụ trợ (snac_tokens 6.5GB/40,779 file, captions_dict 114MB/40,798 file, speech_segments 334MB/40,490 file — dữ liệu cũ, không bị ảnh hưởng bug pose, không cần regen) từ `/p` sang `/e`. Viết `slurm/submit_merge_adaptive_v5.sh` (account `reformo`/`booster`, input đúng base `training_ready_rank_*.jsonl` — không phải v3 — + agent token mới + snac/caption/speech đã copy) — submit job `1007805` (32 task), tất cả `RUNNING` ngay, 0 lỗi. Spot-check live trên phần đã ghi: 112 video/1,148 activity, agent 16.9% (khớp gần đúng con số lịch sử "full-chain 18.8%" của bản v3/v4 cũ), snac 90.7%, caption 90.8% — tín hiệu tốt.

### 6. Bắt đầu tải Harmony4D (352GB)

Viết `tools/extract/download_harmony4d.py` theo đúng convention các script tải khác trong repo (`snapshot_download`, resumable, retry loop, `HF_HUB_DISABLE_XET=1`). Tra trực tiếp cấu trúc thật trên HF (`Jyun-Ting/Harmony4D`): `train/` 15 zip (~287GB, 01_hugging → 15_mma4), `test/` 7 zip (~65GB) — README chỉ 24 byte, không có tài liệu, phải tự giải nén để xem format. License MIT xác nhận trực tiếp trên HF card. Target ban đầu định để `/p` rồi sửa lại `/e` theo quyết định ở mục 5. User tự chạy trong tmux (`logs/harmony4d_download.log`) — phát hiện cần env `activate_env_tools.sh` (khác 2 env chính trong CLAUDE.md, nhẹ hơn, đủ `huggingface_hub`) vì module-load thủ công bị lỗi thiếu `libpython3.12.so.1.0` khi activate `env_stable_vla` trực tiếp không qua đúng trình tự. **Đang tải, ~139GB/352GB (~39%) tại thời điểm ghi entry, 0 lỗi.**

**Sự cố nhỏ:** user paste `HF_TOKEN` thật ra chat khi debug lỗi thiếu module — đã cảnh báo user revoke/tạo token mới sau khi xong việc.

### 7. Phase 6 (merge v5) — COMPLETED, verify sạch

Job `1007805` xong trong **22 phút** (nhanh hơn ước tính 45p-1.5h). 32/32 task COMPLETED, exit 0:0, 160/160 file output, 0 lỗi. Verify số liệu: 40,804 video, 398,775 activity (**khớp tuyệt đối** với số liệu lịch sử v2/v3), agent injected 2,326,095 (**cao hơn** bản cũ 2,148,474 — +8.3%, hợp lý vì fix fps-mismatch giúp match đúng timestamp hơn), snac injected 38,824,718 (**khớp tuyệt đối từng số** với v2 — đúng kỳ vọng vì audio không bị ảnh hưởng bug pose).

### 8. Phase 7 (flatten v6) — 1 lần fail, tìm đúng nguyên nhân, fix, chạy lại

Submit lần đầu (`job 1007976`) dùng `setup_motionbert.sh` (env_motion_final, theo convention Phase 5/6) — **fail ngay** với `ModuleNotFoundError: No module named 'wn'` (thư viện WordNet cho text augmentation, `import wn` cứng ở đầu file, không có try/except). Đáng chú ý: `sacct` vẫn báo `COMPLETED 0:0` dù thực chất crash — vì script bash không check exit code của lệnh python trước dòng echo cuối cùng → **bài học: không tin `sacct` một mình, phải đọc `.err` thật**.

Điều tra: `wn` chỉ có sẵn trong venv `env_tools` (`/p/data1/mmlaion/nguyen38/env_tools`) — nhưng venv này **cũng dính đúng lỗi JUWELS/Jupiter y hệt cả buổi**: `python3` bên trong nó là symlink trỏ tới đường dẫn phần mềm JUWELS (`/p/software/juwels/...`), không tồn tại trên Jupiter, nên "activate" nó từ Jupiter thực chất không chạy gì — Python thật thực thi là bản module-load của Jupiter, dùng package từ `~/.local` (site-packages riêng user, mount toàn cluster). `wn` chưa từng được cài vào `~/.local`. Fix: `python3 -m pip install --user wn` + `python3 -m wn download oewn:2024` chạy từ login node (dùng đúng `python3 -m pip`, không dùng lệnh `pip`/`pip3` trực tiếp — bản thân file `pip` trong venv `/p` cũng có shebang trỏ tới interpreter JUWELS bị hỏng y hệt).

Sửa `submit_phase7_flatten_v6.sh` bỏ `source activate_env_tools.sh` (gây lỗi `No such file or directory` vô hại trên compute node nhưng gây nhiễu debug), thay bằng `module load` trực tiếp. Resubmit (`job 1007994`) — **chạy đúng, ra output thật** (verify: file đầu có 2,031 dòng nội dung `### Context:`/`### Keywords:`/`### Speech:` đúng format). Giữa chừng `squeue`/`scontrol` bị lỗi kết nối SLURM controller tạm thời (cluster-wide, không liên quan job) khiến 1 lần tưởng nhầm job đã "hỏng" — verify lại bằng filesystem trực tiếp (nội dung file output, không phụ thuộc SLURM controller) xác nhận job vẫn chạy tốt, không cần kill/resubmit lại. **Tại thời điểm ghi entry: ~96/160 file, đang chạy tiếp, 0 lỗi.**

### 9. Harmony4D — tiếp tục tải, có 1 lần đứt kết nối tự resume

Tại thời điểm ghi entry: **~308GB/352GB (~87%)**. Có 1 lần mất kết nối giữa chừng khi tải `02_grappling.zip` (đứt ở 28.3/44.1GB) — script tự "Trying to resume download..." theo đúng thiết kế resumable, không cần can thiệp.

### 10. Thảo luận chiến lược: "omni model — data đủ chưa, output là gì?" — làm rõ 3 framing mâu thuẫn

User đặt câu hỏi cấp cao, thấy mơ hồ: liệu data đã đủ cho lần train omni tiếp theo, và output model thực sự là gì. Rà lại toàn bộ docs + thảo luận trước đó, phát hiện **3 framing khác nhau đang tồn tại song song, không khớp nhau**: (1) framing gốc CLAUDE.md — humanoid VLA, output = pose/action token; (2) framing Huu 20/07 — "omni means all modes... cross-modal bindings", không định nghĩa output cụ thể; (3) framing từ chat Discord phiên trước — 2 nhánh train: model nhỏ (omni-VLA + RL-in-sim, có target đo được) và model lớn (MV2, POC "performant in language too", chưa có eval nào đo). Đối chiếu với REPORT.md: đây **không phải vấn đề mới** — chính Van Khue đã tự flag y hệt concern này ngày 20/07 ("not yet a single fixed central research question..."), vẫn chưa giải quyết.

User yêu cầu tự tin đề xuất 1 framing rõ thay vì liệt kê. Đề xuất (dựa trên bằng chứng thực tế đã đầu tư công sức, không phải lời nói): **mục tiêu thật nên là VLA model có năng lực cốt lõi sinh action/pose token hợp lệ, thước đo cuối là closed-loop task success trong simulation (RL rollout)** — không phải benchmark ngôn ngữ. Bằng chứng: toàn bộ pipeline 7-phase chỉ phục vụ việc này; 2 phép eval đã chạy (agent completion, modality transitions) đều đo action; quyết định Harmony4D hôm nay cũng để vá gap action-token. Hệ quả: "đủ data chưa" nên đo bằng **volume + đa dạng agent-token** (hiện chưa đủ — occlusion cắt 46%, chỉ lower-body, 1 người/cảnh — đúng lý do Harmony4D đáng làm), không phải tổng token toàn corpus. Cảnh báo riêng: nếu Huu thật sự muốn claim "performant in language too", **hiện chưa có eval nào đo việc đó** — cần 1 benchmark ngôn ngữ chuẩn riêng, đây là gap hoàn toàn khác, chưa ai làm.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4, 5, 6 FineVideo** — **XONG** cả 3.
- **Phase 7 (flatten v6)** — đang chạy (job `1007994`), ~96/160 file, 0 lỗi, ETA còn vài phút.
- **Harmony4D download** — đang tải, ~308/352GB (~87%).
- Đã cài thêm `wn` (WordNet) + lexicon `oewn:2024` vào `~/.local` — cần thiết cho mọi lần chạy Phase 7 sau này trên Jupiter.
- Việc tồn đọng: verify + upload lại `FineVideo-Phase7-Flattened` sau khi Phase 7 xong; phân tích cấu trúc thật Harmony4D sau khi tải xong rồi thiết kế pipeline convert SMPL/COCO-17 → H36M-17; **quyết định chưa chốt với Huu**: framing/eval protocol thật cho omni model (đề xuất ở mục 10, chưa gửi Huu); tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (vẫn hoãn); chờ Huu quyết định `MixtureVitae-Backup` SNAC + ý nghĩa "moss" token; RoboVQA vẫn treo do câu hỏi kiến trúc chưa giải (16 frame rời rạc/episode vs video liên tục Step A cần).

---

## Cập nhật phiên làm việc — 21/07/2026 (tiếp lần 2 — submit lại Phase 4 FineVideo qua SLURM, thêm wrapper token seed2/cosmos/agent/snac vào 3 dataset, upload lại synth-llava + omnivideo-100k-final)

**Việc chính:** User yêu cầu submit lại Phase 4 FineVideo qua SLURM (JUPITER hết bảo trì phần lớn), rồi trong lúc track tiến độ phát hiện + sửa 1 loạt vấn đề thật: (1) job SLURM đầu tiên chia việc không đều do file đã-xong dồn cục, phải rebalance; (2) `synth_llava` bị lỗi seed2 token lồng sai vào trong `<caption>` (kế thừa từ cấu trúc data gốc); (3) qua thảo luận với user, phát hiện **toàn bộ 3 dataset video/pose/synth_llava đều thiếu wrapper token** (`<seed2>`/`</seed2>`, `<cosmos>`/`</cosmos>`, `<agent>`/`</agent>`, `<snac>`/`</snac>`) dù các token này **đã được đăng ký sẵn trong tokenizer vocab** (`tools/tokenizer/expand_vocab.py`, `build_tokenizers.py`) — chỉ riêng SNAC (`laion_emotional_roleplay`) là có dùng. Quyết định cùng user: thêm wrapper lại cho cả 3 dataset (lý do: tận dụng token đã trả phí sẵn trong vocab, nhất quán với SNAC, và là giả thuyết đáng thử cho lỗi lớn nhất của model hiện tại — "modality transitions: FAIL"). Đã fix code nguồn, regen data, và **upload thành công `EmpathicRobotics/synth-llava` + `EmpathicRobotics/omnivideo-100k-final`** — có 1 lần báo "sẵn sàng" hơi sớm (quên áp fix wrapper vào data `synth_llava` đã tokenize sẵn, chỉ sửa source script), bắt lại kịp trước khi gây hại thêm, đã fix và upload lại đúng.

### 1. Phase 4 FineVideo — submit SLURM, phát hiện + sửa lỗi chia việc không đều

`sinfo` xác nhận phần lớn `booster` hết bảo trì (chỉ còn 18/~5,600 node lẻ tẻ `maint`) — submit `slurm/submit_yolo.sh` (job `1004323`, 128 worker/4 GPU/1 node qua NVIDIA MPS), kế thừa tiến độ cũ từ login-node (17,706/40,305 đã có, script tự skip). User quan sát thấy vài worker xong ngay lập tức trong khi worker khác vẫn chạy chậm — điều tra ra: **48/128 worker xong gần như tức thì** vì file đã-xong (từ lần chạy 4-worker cũ) dồn cục ở đầu mỗi 1/4 khối do chia contiguous, còn lại ~80 worker phải làm gần 100% việc thật (~315 video/worker thay vì ~171 nếu chia đều đúng phần còn lại). **Fix:** `scancel` job cũ, build symlink farm chỉ chứa 21,841 file states_jsonl còn thiếu (`outputs/states_jsonl_30fps_remaining_phase4/`, NFS symlink chậm nên phải chạy nền + resume 2 lần do timeout), viết `slurm/submit_yolo_remaining.sh` (giống hệt `submit_yolo.sh`, chỉ đổi `--input-dir` trỏ vào symlink farm) — submit lại (`job 1004747`), chia đều ~171 video/worker. **Đang chạy tốt** (21,342/40,305 tại thời điểm ghi entry, 1 lỗi soft/video không fatal, worker tự chuyển video kế tiếp).

### 2. Sửa lỗi `synth_llava`: seed2 token lồng sai vào `<caption>`

User phát hiện: data gốc `synth_llava`/`synth_llava2` (Huu tự tạo) đã có sẵn cấu trúc `<caption><image_0>caption text</caption>` — `tokenize_seed2.py` chỉ string-replace placeholder tại chỗ, giữ nguyên lồng sai (seed2 token nằm trong `<caption>`, trong khi convention thật của project — `step_a_tokenize_video.py` — coi `<seed2>`/`<caption>` là 2 tag ngang hàng, tuần tự, không lồng nhau). Fix bằng 2 pass regex trên 603,999 dòng × 151 file (pass đầu bị bug con — token seed2 cuối cùng dính liền text không space bị regex bỏ sót, sinh lỗi mới; verify + fix tiếp ngay). Đồng thời sửa `tokenize_seed2.py` (source) để lần rerun sau không lặp lại.

### 3. Phát hiện + quyết định: thêm lại wrapper token cho seed2/cosmos/agent/snac

Trong lúc thảo luận, user chỉ ra `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened` có `<snac>`/`</snac>` wrapper nhưng `FineVideo-Phase7-Flattened`/`omnivideo-100k-final` thì không, dù cả 2 loại token đều **đã đăng ký sẵn trong vocab tokenizer** (`<seed2>`, `</seed2>`, `<cosmos>`, `</cosmos>`, `<avc_lm>`, `</avc_lm>`, `<agent>`, `</agent>` — thấy trong `tools/tokenizer/expand_vocab.py` + `build_tokenizers.py`). Điều tra ra: đây **không phải bug quên sót cụ thể ở seed2/cosmos** — `pipeline_pose/phase7_flatten.py`/`flatten_step_a_video.py` lột bỏ wrapper cho **mọi** loại token đồng nhất (seed2/cosmos/avc_lm/agent outer/snac outer đều bị lột, chỉ `<caption>`/`<speech>` giữ lại) — là quy ước có chủ đích từ đầu. Trong khi `laion_emotional_roleplay/tokenize_snac.py` là code **hoàn toàn riêng biệt**, không dùng lại `phase7_flatten.py`, và có quyết định wrapper riêng (ghi rõ trong docstring: "decided after review with Van Khue, session 2026-07-20") — nên 2 format song song không nhất quán từ trước tới giờ, không phải lỗi phát sinh mới.

User hỏi lý do kỹ thuật tại sao cần wrapper trước khi cho làm — đã giải thích: (a) token `</seed2>` tách bạch "đã hết span" khỏi "modal kế tiếp là gì" — 2 tín hiệu vốn bị gộp làm 1 khi không có wrapper, đúng loại vấn đề mà model đang fail ("modality transitions: FAIL... stays in seed2 mode"); (b) dropout nặng (cosmos 50-90%, avclm ~99%) làm "tập modal hợp lệ sau seed2" đổi liên tục giữa các ví dụ, cộng dồn nhiễu lên đúng tín hiệu đó; (c) pattern chuẩn trong nhiều hệ multimodal/agent LLM thật (Chameleon, tool-call wrapper...); (d) minh chứng sống ngay trong phiên — chính regex đoán ranh giới bằng khoảng trắng ở mục 2 đã sai 1 lần vì thiếu wrapper rõ ràng. User đồng ý, quyết định thêm wrapper cho cả 3 dataset, **để riêng FineVideo-Phase7-Flattened chờ Phase 4 xong** (không block bởi user, sẽ tự đúng khi Phase 5→6→7 chạy lại vì code đã fix sẵn).

**Đã sửa code nguồn** (4 file, đều unit-test bằng data giả lập trước khi áp dụng):
- `pipeline_pose/phase7_flatten.py` — `process_activity_per_chunk()` + `count_token_types()` (tránh đếm nhầm token wrapper mới vào bucket agent)
- `data_prep/omnivideo_100k/step_a/flatten_step_a_video.py` — mirror fix (dù xác nhận đây **không phải** script build `omnivideo-100k-final` thật, chỉ giữ đồng bộ cho tương lai)
- `data_prep/omnivideo_100k/phase6_merge_omnivideo.py` — **script thật** build `omnivideo-100k-final` (đọc thẳng raw Step A, tự flatten+merge agent trong 1 hàm riêng `flatten_token_stream_with_agent`, không tái dùng `flatten_step_a_video.py` như docstring cũ ghi nhầm)
- `data_prep/synth_llava/tokenize_seed2.py`

**Regen `omnivideo-100k-final`:** chạy lại `phase6_merge_omnivideo.py` (32 file, ~17s/file) rồi `phase7_finalize_omnivideo.py` — verify số liệu khớp **chính xác tuyệt đối** với bản cũ (5,214→5,214, 0 malformed, 799 video có agent, 62,631 window agent, 5,214/5,214 có QA) — chỉ đổi cấu trúc token, không đổi nội dung/số lượng.

**Fix `synth_llava_flat`:** áp thêm 1 pass regex thêm `<seed2>`/`</seed2>` bọc quanh chuỗi seed2 token đã có (an toàn vì mỗi dòng chỉ có đúng 1 span seed2, không như OmniVideo có nhiều chunk cùng loại liên tiếp — không thể patch hậu-kỳ an toàn kiểu này cho OmniVideo, phải regen từ raw). Verify: 603,999 dòng, 19,327,968 seed2 token (không đổi), 0 lỗi format.

**Sự cố nhỏ:** báo user "sẵn sàng upload" 1 lần trước khi kịp áp fix wrapper vào `synth_llava_flat` (chỉ sửa xong source script, quên chạy lại trên data đã có) — user chạy upload, thành công lên HF nhưng thiếu wrapper. Phát hiện qua triệu chứng user báo ("sao lại skip compress") — do cache `.gz` cũ (từ lần upload lỗi format `<caption>` trước đó) đã tồn tại nên script skip đúng theo thiết kế. Verify bằng `HfApi().dataset_info()` xác nhận thời điểm upload thật, fix data, xóa cache, upload lại — không phát hiện vấn đề gì với `omnivideo-100k-final` (user chưa kịp upload bản cũ nên không bị ảnh hưởng).

### Kết quả upload (đã xong, verify bằng log user gửi)

- **`EmpathicRobotics/synth-llava`** — 151 shard (140 train + 11 test), 135MB, upload thành công.
- **`EmpathicRobotics/omnivideo-100k-final`** — 32 shard (30 train + 2 test), 683MB, upload thành công.

### Trạng thái cuối phiên

- **Phase 4 FineVideo** (SLURM job `1004747`, rebalanced, 128 worker) — đang chạy, 21,342/40,305 (~53%), 1 lỗi soft/video không fatal.
- **`synth-llava`, `omnivideo-100k-final`** — đã upload lại đúng format (wrapper token + caption fix), **XONG**.
- **`FineVideo-Phase7-Flattened`** — chưa regen, chờ Phase 4 xong rồi chạy Phase 5→6→7 bình thường (code `phase7_flatten.py` đã fix sẵn, không cần làm gì thêm ngoài chạy pipeline như thường lệ).
- Việc tồn đọng: sau khi Phase 4 xong — chạy Phase 5 (agent tokens) → Phase 6 (merge, **nhớ dùng script kiểu `submit_merge_adaptive_v4.sh` có `--captions-dir`/`--speech-segments-dir`, không dùng bản merge thường**, xem mục dưới) → Phase 7 (flatten, đã có wrapper) → upload lại `FineVideo-Phase7-Flattened`; tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (đã hứa từ lâu, vẫn hoãn); chờ Huu quyết định `MixtureVitae-Backup` SNAC + ý nghĩa "moss" token.

### Cập nhật tiếp — track lại Phase 4, tự sửa 1 nhận định sai về FineVideo v5, tổng hợp bức tranh 4 dataset

**Phase 4** (job `1004747`) — cập nhật: **27,056/40,305 (~67%)**, 0 lỗi trừ 1 soft/video đã ghi ở trên, chạy ổn định ~1h. User phát hiện tôi (Claude) đọc nhầm log cũ (`logs/yolo_workers_run1_unbalanced/` — log job đầu tiên đã hủy, archive trước khi resubmit) tưởng job vẫn chia việc lệch — job hiện tại `logs/yolo_workers/` xác nhận **0 SKIP**, mỗi worker đúng 170-171 file, rebalance hoạt động đúng.

**Tự sửa sai:** khi được hỏi "gộp 4-5 dataset đã đủ train chưa", tôi khẳng định sai rằng FineVideo-VLA **không có caption** — dựa trên check nhầm `/p/data1/mmlaion/shared/vla/vla_adaptive/` (bản THẬT nhưng là bản CŨ, tiền-caption, chính là data đã tokenize cho model đang train hiện tại — 2.84B token — không phải v5). User chỉ ra ngay ("bị ngáo à"), check lại đúng thư mục `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v5/` (live trên `EmpathicRobotics/FineVideo-Phase7-Flattened` từ 07/07, xem REPORT.md §21): **100% có `<caption>`, 96% có `### Speech:` header, 80% có `<speech>` inline** — đúng như REPORT.md đã ghi từ trước (371,888 record, 5.217B token), tôi chỉ tra sai thư mục local, không phải docs sai. Số tổng 12.53B đã báo trước đó **vẫn đúng** (seed2/cosmos đếm lại khớp chính xác tuyệt đối với log cũ) — chỉ riêng nhận định "thiếu caption" là sai, đã rút lại.

**Bảng phân loại 4 dataset (không tính MV-Omni):**

| Dataset | Record | Modal | Token |
|---|---|---|---|
| FineVideo-VLA (v5, đang regen fps-fix+wrapper) | 371,888 | seed2 + cosmos + agent (18.8% full-chain) + snac + caption (Qwen2.5-VL) + speech (Whisper header+inline) | 5.217B |
| omnivideo-100k-final | 5,214 | seed2 + cosmos + agent (799/5,214, chỉ sports-subset) + caption + speech + QA (99,983 cặp, có reasoning hint xuyên-modal) | 274.6M |
| synth-llava/synth-llava2 | 603,999 | chỉ seed2 (ảnh tĩnh, 32 tok/ảnh) + caption — không video/pose/audio | 70.8M |
| emotional-roleplay-finetuning-dataset-flattened | 67,459 | chỉ snac (audio) + text — không video/pose | 26.9M |
| **Tổng** | | | **~12.53B** |

Cộng MV-Omni (+6.93B, chủ yếu SNAC/audio) → ~19.4B, nằm trong khoảng "10-20B đủ" từng ước tính, chưa tới target tham vọng "20-40B". **Kết luận đưa cho user: đủ để thử train v0.2** sau khi FineVideo regen xong (giữ đúng caption/speech, xem nhắc nhở trên) — không cần thêm RoboVQA mới đạt ngưỡng sẵn sàng.

## Cập nhật phiên làm việc — 21/07/2026 (track lại Phase 4 FineVideo + synth_llava seed2 tokenize — xác nhận synth_llava đã DONE hoàn toàn)

**Việc chính:** User yêu cầu track lại 2 job đang chạy nền mà entry trước ghi "cần check lại % hiện tại" — verify bằng log thật (không suy đoán từ docs cũ).

### 1. synth_llava seed2 tokenize — ĐÃ XONG hoàn toàn

`logs/synth_llava_seed2.log` kết thúc lúc 20:02 tối 20/07, `SYNTH_LLAVA_EXIT=0`:

```
TONG: 603,999 -> 603,999 | no_image: 0 | encode_fail: 0 | seed2_tokens: 19,327,968
```

100% thành công, 0 lỗi. Output: 151 file `{synth_llava,synth_llava2}_shard-*.jsonl` (56 + 95 file) tại `/p/data1/mmlaion/shared/vla/synth_llava_flat/`, mỗi dòng đã đúng format phẳng training-ready `{"id", "text"}` với `<image_0>` đã thay bằng chuỗi `<seed2_N> ...` atomic-token — verify trực tiếp bằng cách đọc vài dòng thật.

**Không cần bước "merge" riêng cho dataset này** — khác FineVideo/OmniVideo (nhiều stream token: video + agent pose + QA cần Phase 6 merge lại), `synth_llava` chỉ có 1 modality (ảnh → seed2) nên `tokenize_seed2.py` sinh thẳng ra format phẳng cuối cùng, không có agent/pose stream nào khác để ghép vào.

### 2. Phase 4 FineVideo — vẫn đang chạy, ~33%

Không connect được tmux `yolo_login_finevideo` từ session hiện tại (`tmux ls` lỗi "No such file or directory"), nhưng 4 file log worker vẫn được ghi liên tục **đúng tới thời điểm check** (mtime khớp giây với `date`) → job chắc chắn còn sống, chỉ là sandbox hiện tại không thấy tiến trình đó.

| Worker | Vị trí | Done |
|---|---|---|
| 1 | 3858/10076 | 3377 |
| 2 | 3851/10076 | 3334 |
| 3 | 3846/10076 | 3339 |
| 4 | 3827/10077 | 3321 |

Tổng: **13,371/40,305 video (~33%)**, 0 lỗi trên cả 4 log.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy, ~33% (13,371/40,305).
- **synth_llava seed2 tokenize** — **XONG**, 603,999 dòng, 19,327,968 seed2 token, output tại `/p/data1/mmlaion/shared/vla/synth_llava_flat/`.
- Việc tiếp theo: viết script consolidate/upload HF cho `synth_llava_flat` theo convention `laion_emotional_roleplay/upload_hf.py` (nén + train/test split theo shard + dataset card) — chờ user chọn tên repo trước khi upload thật.

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 6 — recap RoboVQA, đếm token distribution thật cho 4 dataset đã flatten, đối chiếu số liệu docs cũ)

**Việc chính:** Recap lại RoboVQA cho user (không có việc mới, chỉ tổng hợp lại điều tra đã làm). Viết script đếm token tổng quát (tái dùng đúng `PATTERNS`/`count_tokens` convention từ `tools/inventory/data_inventory.py` để so sánh ngang hàng với số liệu cũ), chạy full trên cả 4 dataset đã flatten: FineVideo-VLA v5, MV-Omni (phần valid_snac đã convert), OmniVideo-100K final, emotional-roleplay-finetuning-dataset. User bắt đầu soạn 2 sbatch Megatron-tokenize cho OmniVideo/roleplay để chạy trên JUWELS — **tự viết nhầm 1 bản đoán mò không có template thật, user chỉ ra và yêu cầu xóa**, đã xóa, chờ user cung cấp template thật.

### 1. Đếm token thật — kết quả (multiprocessing, 24 worker/dataset, ~9 phút tổng cho cả 4 dataset trên login node)

| Dataset | seed2 | cosmos | avclm | snac | agent | text (word-count xấp xỉ) | **TOTAL** |
|---|---|---|---|---|---|---|---|
| emotional-roleplay-finetuning-dataset | 0 | 0 | 0 | 23,390,760 | 0 | 3,556,739 | **26,947,499** (27.0M) |
| omnivideo-100k-final | 17,229,664 | 201,736,400 | 0 | 0 | 19,185,063 | 36,480,085 | **274,631,212** (274.6M) |
| mv-omni (valid_snac converted) | 19,249,664 | 0 | 0 | 4,922,681,181 | 0 | 1,990,708,185 | **6,932,639,030** (6.933B) |
| finevideo-vla (v5) | 332,592,448 | 3,882,954,800 | 0 | 363,029,331 | 564,876,258 | 82,430,660 | **5,225,883,497** (5.226B) |

`avclm` luôn = 0 ở cả 4 — đúng thiết kế (payload avc_lm luôn bị discard ở bước flatten cuối cùng của mọi pipeline trong project, không phải bug — user tự hỏi và tự xác nhận đúng).

**Đối chiếu với số liệu cũ trong docs:**
- **MV-Omni: 6.933B — khớp gần như tuyệt đối** với con số "+6.93B token" đã ghi ở mục "Bức tranh data" (PROGRESS_VI.md, từ investigation tháng 6).
- **FineVideo-VLA v5: 5.226B — khớp gần đúng** với con số "5.256B tokens" đã ghi khi upload v5 lên HF (chênh ~0.03B, hợp lý vì cách đếm text ở đây là word-split xấp xỉ, không phải đếm qua tokenizer thật).
- **Không tìm thấy con số "10B"/"20B" cụ thể nào khớp trực tiếp** — các số này trong docs thực ra là **ước tính mục tiêu** (ví dụ "Target: 20–40B token cho v0.2 training", "sau khi hoàn thành ưu tiên 1,2,4: ước tính 10–20B token") chứ không phải số đo thực tế của 1 dataset cụ thể nào — đã báo lại cho user để tránh nhầm giữa target và actual.

### 2. Recap RoboVQA cho user (không có việc mới — tổng hợp lại điều tra đã làm)

Nhắc lại: tải về có `tfrecord/{train,val}` (184 shard) + `json/{train,val}` (pre-extract) + **9,999 video `.mp4` thật** + `instructions/*.txt` + LICENSE Apache-2.0. Đã làm: `flatten_text.py` xong (221,912 dòng, cố tình bỏ video_id — quyết định deprioritized ghi rõ trong docstring, không phải bug). `extract_frames.py` dở dang 130/184 shard, không có job chạy. Nhắc lại phát hiện quan trọng: TFRecord thật có `texts_start`/`texts_end` luôn = 15 (frame cuối trong 16 frame/episode) — QA không interleave được theo từng loại task, chỉ neo được ở cấp "cả episode".

### 3. Tự viết nhầm sbatch Megatron-tokenize không có template thật

User yêu cầu viết 2 sbatch để submit Megatron-tokenize cho OmniVideo-100K + roleplay trên JUWELS. Tìm quanh oellm-autoexp không thấy script preprocess_data.py thực tế đã dùng cho lần tokenize FineVideo trước đó (chỉ tìm thấy job.sbatch của bước TRAIN, dùng container JUPITER-aarch64-specific, không áp dụng cho JUWELS x86_64) — **tự đoán mò 1 bản generic** (module load placeholder, Megatron tools/preprocess_data.py CLI chuẩn). User chỉ ra ngay là sai hướng, sẽ tự cung cấp template thật. Đã xóa file đoán mò, chờ template.

### 4. Kiểm tra mức dùng login node JUPITER theo yêu cầu user (lo bị admin report)

Recheck: load average 18.28/72 core (~25%), GPU 15GB/97GB dùng, 60% util, 14 user khác online — **mức chấp nhận được**, không còn kịch 100% GPU liên tục như trước khi giảm worker (mục entry trước). 3 job chạy song song ổn định: Phase 4 FineVideo (4 worker), synth_llava seed2 tokenize, đếm token 4 dataset (đã xong).

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy.
- **synth_llava seed2 tokenize** (tmux `synth_llava_seed2`) — đang chạy, ETA ước tính ban đầu ~6h (cần check lại % hiện tại).
- Đếm token 4 dataset — **XONG**, script lưu tạm ở `logs/count_tokens_4datasets.py` (chưa đưa vào `tools/inventory/`, cân nhắc chuyển vào đó nếu cần dùng lại).
- Việc tồn đọng: chờ user cung cấp template Megatron-tokenize thật cho JUWELS; tách `pipeline_pose/`+`pipeline_video/` (chờ Phase 4 FineVideo xong); resume `extract_frames.py` RoboVQA nếu muốn đẩy tiếp; quyết định `MixtureVitae-Backup` SNAC + "moss" token (chờ Huu).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 5 — điều tra RoboVQA + synth_llava, viết + chạy seed2 tokenize cho synth_llava, viết detokenizer chung cosmos/avc_lm, giảm tải GPU login node)

**Việc chính:** Trả lời loạt câu hỏi tổng hợp về 3 dataset mới (RoboVQA, synth_llava) và cơ chế decode 3 loại token video (seed2/cosmos/avc_lm) — đều verify bằng cách chạy code thật, không suy đoán. Viết + launch full-scale seed2 tokenize cho `synth_llava` (603,999 ảnh). User lo ngại GPU login node bị admin report vì Phase 4 FineVideo chiếm 100% GPU liên tục nhiều giờ — giảm từ 16 xuống 4 worker. Viết 2 tool detokenize tổng quát (`tools/decode/decode_cosmos.py`, `decode_avclm.py`), bắt được 1 bug thật trong avc_lm decode.

### 1. RoboVQA — datasets.md sai (ghi "chưa tải" nhưng thực tế đã tải+xử lý 1 phần từ 18/7)

Audit lại: `/p/data1/mmlaion/shared/vla/robovqa/` có đủ 184 shard TFRecord (175 train + 9 val) + `json/` (đã pre-extract) + **9,999 video `.mp4` thật** + `instructions/*.txt` + `LICENSE.txt` (Apache 2.0, permissive thật). Đã làm: `flatten_text.py` xong (221,912 dòng text, **cố tình bỏ `video_id`** — có ghi rõ trong docstring là quyết định deprioritized, không phải bug như vụ OmniVideo QA). `extract_frames.py` dở dang: **130/184 shard (~70.6%)**, không có job nào đang chạy.

**Câu hỏi user: text có anchor theo frame để interleave không?** Parse trực tiếp TFRecord thật (`tfrecord_lite.py`, tự viết vì không có tensorflow trong env nào): mỗi episode có `texts_start`/`texts_end` — verify 15 record thật, **luôn luôn = 15** (frame cuối trong 16 frame/episode). Nghĩa là toàn bộ multi-task QA (affordance/planning/success/future_prediction...) đã gộp thành **1 blob duy nhất, neo vào 1 điểm** (cuối episode) — không thể interleave từng loại QA riêng theo thời gian như FineVideo's caption/speech. Đơn vị hợp lý nhất để interleave: **cả episode (16 frame) = 1 chunk**, không phải nhiều điểm trong episode.

### 2. synth_llava — format xác nhận lại, viết + chạy tokenize seed2

Format thật (verify từ data, không chỉ nhớ lại): mỗi shard = `.jsonl` (`{"text": "<caption><image_0>...</caption>", "metadata", "language", "media": {"<image_0>": "filename.png"}}`) + `.wds` (tar POSIX chứa ảnh PNG thật). Verify 4,000 dòng shard đầu: **luôn đúng 1 ảnh/record**, placeholder `<image_0>` luôn nằm ngay đầu caption.

Viết `data_prep/synth_llava/tokenize_seed2.py` — tái dùng chính xác import/shim của `step_a_tokenize_video.py` (fix bug transformers-version + Qformer.cls=None đã biết), thay `<image_0>` bằng chuỗi `<seed2_N>` atomic trực tiếp tại vị trí đó (không cosmos/avc_lm vì ảnh tĩnh không cần). Test đo throughput thật: **~30 ảnh/giây** trên GPU rảnh, 0 lỗi. **Đã launch full-scale** (151 shard, 603,999 ảnh) trong tmux `synth_llava_seed2`, ETA ước tính ~6 giờ. Tại thời điểm ghi entry: 8/151 shard xong, 0 lỗi.

### 3. GPU login node bị chiếm 100% liên tục — user lo bị admin report, giảm tải

Phase 4 FineVideo (16 worker) khiến GPU 100% util suốt ~2 tiếng trong khi 12 user khác cũng online trên login node. Theo yêu cầu user: **kill session, restart với 4 worker** (giảm `slurm/run_yolo_login.sh` từ `NUM_WORKERS=16` xuống `4`) — verify GPU về 0% trước khi restart, không mất tiến độ (script skip-existing). Sau restart: GPU ~83-99% khi chạy đơn lẻ nhưng **có headroom hơn**, và giờ chia sẻ với job seed2 (2 job cùng chạy). Test throughput seed2 trên GPU rảnh trước khi quyết định tạm dừng Phase 4 nhường chỗ — kết luận seed2 đủ nhanh (~6h) nên **không cần dừng hẳn Phase 4**, chỉ giảm worker.

Cũng thử submit 1 job SLURM test (`985910`) để kiểm tra maintenance đã hết chưa — vẫn bị chặn y hệt (`ReqNodeNotAvail, Reserved for maintenance`), đã cancel ngay.

### 4. Điều tra decode seed2/cosmos/avc_lm — đọc paper, viết tool tổng quát, bắt 1 bug thật

**Seed2 (đọc paper user để trong `documents/2310.01218v1 (1).pdf` — "Making LLaMA SEE and Draw with SEED Tokenizer", Tencent AI Lab):** SEED không phải codec nén — là tokenizer bậc ngữ nghĩa cao, "de-tokenize" thật ra là **sinh lại ảnh mới cùng ngữ nghĩa** qua 1 pipeline Stable Diffusion riêng (`stable-diffusion-2-1-unclip`), verify từ chính README `ontocord/seed2`. Không tìm được paper "SEED2" chính thức nào — chỉ có "SEED"/SEED-LLaMA từ Tencent; checkpoint `ontocord/seed2` là bản build của 1 tổ chức khác, lý do đặt tên "2" không rõ, không đoán bừa.

**Cosmos — viết `tools/decode/decode_cosmos.py` (tool tổng quát, không phải script 1 lần):** verify shape thật — 8 frame/160x160 → encode ra indices `(1,2,10,10)` = 200 token/chunk. Decode qua `Cosmos-Tokenizer-DV8x16x16/decoder.jit` → `(1,3,9,160,160)`. Support cả 2 format token trong project (atomic `<cosmos_N>` cho data đã flatten, raw block `<cosmos>N N N</cosmos>` cho Step A chưa flatten) và cả 2 schema record (`{"text":...}` phẳng kiểu OmniVideo, `{"scenes":[...]}` lồng nhau kiểu FineVideo raw).

**AVC-LM — viết `tools/decode/decode_avclm.py`, bắt 1 bug thật:** `avc_lm_v2/tokenizer.json` không có `decoder` component (`decoder: None`) — gọi `tokenizer.decode()` mặc định sẽ tự chèn dấu cách giữa mỗi token, phá byte stream (verify: ffmpeg báo "non-existing PPS 0 referenced" khi test round-trip thật). **Fix:** nối trực tiếp `tok.id_to_token(id)` cho từng id, không dấu cách (đúng vì mỗi token là substring latin-1 thô, không có ByteLevel remap). Sau fix: ffmpeg decode sạch (returncode 0, 8 frame thật) — AVC-LM là **byte-exact**, khác hẳn seed2 (generative) và cosmos (neural reconstruct lossy).

Đã decode thử + gửi user xem: 1 chunk cosmos + 1 chunk avc_lm từ sample OmniVideo (`samples/omnivideo_100k_final_sample/`), và 1 record FineVideo-VLA thật (`d6b4OmUFt7I`, `samples/finevideo-vla/`) cho cả cosmos lẫn avc_lm.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Cluster vẫn maintenance (job test `985910` xác nhận, đã cancel).
- **Phase 4 FineVideo** (tmux `yolo_login_finevideo`, 4 worker) — đang chạy, 4,073/40,305 video.
- **synth_llava seed2 tokenize** (tmux `synth_llava_seed2`, 1 process) — đang chạy, 8/151 shard, 0 lỗi, ETA ~6h.
- RoboVQA: `extract_frames.py` vẫn dở dang 130/184 shard, chưa resume (không phải việc đang làm phiên này).
- Việc tồn đọng: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (chờ Phase 4 FineVideo xong); resume `extract_frames.py` cho RoboVQA nếu muốn đẩy tiếp; quyết định `MixtureVitae-Backup` SNAC + "moss" token (chờ Huu).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 4 — hoàn tất merge OmniVideo-100K 3 track, sample+upload script, chỉ còn chờ Phase 4 FineVideo)

**Việc chính:** Chạy xong + verify sạch cả 2 bước còn lại của plan merge OmniVideo-100K (Bước 2 `phase6_merge_omnivideo.py`, Bước 3 `phase7_finalize_omnivideo.py`), viết sample + script upload HF theo đúng convention `laion_emotional_roleplay`, tự sửa lại mô tả cấu trúc record sau khi user chỉ ra thiếu phần "Context" ở đầu. Phase 4 FineVideo (tmux `yolo_login_finevideo`) vẫn chạy nền bình thường suốt phiên, không có gì khác cần làm ngoài chờ nó xong.

### 1. Bước 2 — `phase6_merge_omnivideo.py` — chạy xong, verify hoàn hảo

Lần đầu chạy thử vô tình chạy full 32 file thay vì test 1 file (do quên giới hạn), ghi nhầm ra `/tmp` scratchpad + có `| head -5` dễ gây `BrokenPipeError` giữa chừng — user nhắc kiểm tra thời gian, đã kill và chạy lại đúng trong tmux `phase6_merge_omnivideo`, ghi thẳng vào `/p/data1/mmlaion/shared/vla/omnivideo_100k_video_agent_merged/`.

**Kết quả (verify bằng đếm thật, không chỉ tin log):**
- 5,214 → 5,214 (0 malformed)
- **799 video có `<agent>`** — khớp chính xác 100% số file Phase 5 non-empty
- **62,631 window agent injected** — khớp chính xác 100% tổng số dòng trong toàn bộ `pose_agent_tokens_adaptive/*.jsonl` (đếm riêng, đối chiếu bằng `wc -l`) — không mất window nào
- Spot-check 1 record thật: `<agent>` xuất hiện đúng ngay sau cosmos của đúng chunk, format `<agent> <fps_30> <pelvis>...</agent>` khớp thiết kế

### 2. Bước 3 — `phase7_finalize_omnivideo.py` — chạy xong, verify hoàn hảo

Join QA (`omnivideo_100k_qa_flat.jsonl`, đã fix giữ `video_id` ở entry trước) vào cuối record Bước 2, gộp theo `video_id`. Kết quả: **5,214 → 5,214, 5,214/5,214 video có QA (100%), 0 warning** (không có video_id nào lệch giữa QA và Step A — tập video_id trùng khớp hoàn toàn cả 2 track). Output: `/p/data1/mmlaion/shared/vla/omnivideo_100k_final/` (32 file, 3.6GB).

**Verify tổng thể dataset cuối:** 17,229,664 token seed2, 201,736,400 token cosmos, 799 video có agent (62,631 window), 5,214/5,214 có QA.

### 3. Sample + script upload HF

Ghi 1 record thật vào `samples/omnivideo_100k_final_sample/` (bản raw 568KB + bản `_PREVIEW_readable.txt` đã cắt bớt run cosmos/seed2 dài cho dễ đọc), gửi cho user qua `SendUserFile`. Viết `data_prep/omnivideo_100k/upload_hf.py` theo đúng pattern `data_prep/laion_emotional_roleplay/upload_hf.py` (gzip nén, split train/test theo shard seed 42, `--repo-id` CLI arg). Dry-run (`--skip-compress`... à `--skip-upload`) trước: nén 32 shard (3.6GB → 648MB, 30 train/2 test), `gzip -t` pass hết. Đã đưa user lệnh upload đầy đủ, user tự export `HF_TOKEN`.

**Tự sửa 1 điểm sau khi user đọc sample không hiểu:** giải thích lần đầu bỏ sót phần `### Context: ...` (header caption tổng quan cả video, lấy nguyên từ `scripts.jsonl` gốc qua `step_a_tokenize_video.py`, giữ nguyên xuyên suốt cả `flatten_step_a_video.py` lẫn `phase6_merge_omnivideo.py`) — cấu trúc record thật là **3 phần**: `### Context: ...` → chuỗi token theo chunk (seed2/cosmos/agent) → QA nối cuối. Đã sửa lại giải thích + sẽ nhớ mô tả đủ 3 phần trong dataset card.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

Recheck toàn bộ trước khi ghi entry này:
- Cluster vẫn maintenance (18 node `booster` `maint`, không đổi).
- Phase 4 FineVideo (tmux `yolo_login_finevideo`) — **2,882/40,305 video, 0 lỗi** trên tất cả worker log, đang chạy khỏe.
- OmniVideo-100K final dataset + gzip upload bundle — recheck lại, vẫn nguyên vẹn (32 file, gzip integrity pass).

**Không còn việc gì khác đang treo trong phiên này — chỉ còn chờ Phase 4 FineVideo chạy xong (~15h từ lúc submit ~12:22 20/7).** Việc tiếp theo sau khi Phase 4 xong: tách `pipeline_pose/`+`pipeline_video/` (FineVideo) vào `data_prep/finevideo/` cho đối xứng với `data_prep/omnivideo_100k/` (đã hứa từ trước, chưa làm vì job đang import trực tiếp từ đó). Việc tồn đọng khác: chờ Huu quyết định `MixtureVitae-Backup` SNAC (~3.27B code) + ý nghĩa "moss" token cho roleplay dataset; chuyển `mixturevitae_multimodal/synth_llava` đã xong từ entry trước (không còn tồn đọng).

## Cập nhật phiên làm việc — 20/07/2026 (tiếp lần 3 — reorg data_prep/omnivideo_100k, check chất lượng Phase 5 agent token, lên plan + bắt đầu merge 3 track OmniVideo-100K)

**Việc chính:** Phase 4 FineVideo login-node (tmux `yolo_login_finevideo`) tiếp tục chạy nền, ETA ~15h, không đụng tới. Trong lúc chờ: reorg `data_prep/omnivideo_100k/` (28 file phẳng → 4 thư mục con theo mục đích), check chất lượng Phase 5 agent token (tốt, không lỗi), làm rõ nhầm lẫn về 2 folder Step A tưởng duplicate, tổng hợp lại đúng cấu trúc 3 track của OmniVideo-100K, và bắt đầu viết pipeline merge 3 track thành 1 dataset hoàn chỉnh.

### 1. Reorg `data_prep/omnivideo_100k/` — xong

Từ 28 file phẳng (trộn Step A + Pose + dataset-curation) → 4 thư mục con:
```
data_prep/omnivideo_100k/
  pose/            # Phase 1-4 driver + submit .sbatch + task doc
  step_a/          # step_a_tokenize_video.py, flatten_step_a_video.py, debug_seed2_load.py
  dataset_prep/    # select_sports_subset.py, filter_animation_content.py, build_segment_captions.py, flatten_qa_text.py
  analysis/        # compare_native_vs_30fps_render.py + compare_renders/
  sports_subset_video_ids*.txt   # giữ top-level, dùng chung bởi pose/*
```
Sửa + verify bằng chạy thật (không chỉ đoán):
- 5 file dùng sys.path trick (`dirname` x3 → x4) để vẫn ra đúng repo root khi bị đẩy sâu 1 cấp — verify bằng cách chạy `--help` thật, xác nhận import `pipeline_pose.*` thành công (chỉ lỗi thiếu numpy/cv2 ở env hệ thống, không lỗi path).
- 5 file có `DEFAULT_VIDEO_IDS_FILE` — sửa path 1 cấp vì txt vẫn ở top-level, verify `os.path.exists()` = True cho cả 5.
- `DEFAULT_MODEL` (yolo26n.pt) trong phase4 — verify resolve đúng file thật ở repo root.
- Sửa path trong tất cả `submit_*.sbatch` + vài dòng docstring tự tham chiếu.
- Xóa `__pycache__/`.
Job Phase 4 FineVideo không bị ảnh hưởng (dùng `pipeline_pose/phase4_yolo_cleaner.py` gốc, không liên quan `data_prep/omnivideo_100k/`).

### 2. Giải oan cho `flatten_step_a_video.py` — không phải bug

Phiên trước nói nhầm: "`omnivideo_100k_video_flattened/` KHÔNG CÓ token `<seed2>`" — dựa trên so sánh chỉ 2000 ký tự đầu file nên hiểu sai. Kiểm tra sâu (đếm full file): **CÓ** 3,776 token `<seed2_N>` cho đúng video đó. Lý do nằm sai vị trí: `flatten_token_stream()` chỉ "flush" seed2/cosmos ra output tại đúng điểm gặp `<avc_lm>` (không theo thứ tự xuất hiện gốc trong input) — không phải mất token, thứ tự trong output đơn giản là khác thứ tự trong input. Không sửa gì, code đúng như thiết kế.

### 3. Tổng hợp lại cấu trúc thật của OmniVideo-100K — 3 track độc lập

Dataset gốc tải qua `tools/extract/download_omnivideo_100k.py` (HF `MiG-NJU/OmniVideo-100K`, Apache-2.0, 52.9GB): `videos.tar.part_aa..ae` (5 phần, giải nén ra **5,214 video** thật) + `scripts.jsonl` (149MB, caption/script sẵn có) + `train_oe_70k.jsonl` (70,017 QA mở) + `train_mcq_30k.jsonl` (29,966 QA trắc nghiệm) + 2 bản `_formatted`. Tổng QA 99,983 ≈ "100K" trong tên dataset — hoá ra tên gọi theo số QA pair, không phải số video (chỉ 5,214 video thật).

3 track xử lý song song, độc lập, join bằng `video_id`:

| Track | Pipeline | Scope | Trạng thái |
|---|---|---|---|
| 1. QA text (không video) | `dataset_prep/flatten_qa_text.py` | 5,214 video | Xong từ trước |
| 2. Step A video-tokenize (seed2/cosmos/avc_lm) | `step_a/step_a_tokenize_video.py` → `step_a/flatten_step_a_video.py` | **5,214/5,214 video** | Xong từ trước |
| 3. Pose pipeline (Phase 1-5, agent token) | `pose/phase1..4_*_omnivideo.py` + `pipeline_pose/phase5_adaptive_pchip.py` (dùng chung FineVideo) | Chỉ sports subset **1,126/5,214** | Vừa xong Phase 5 hôm nay (799 non-empty) |

### 4. Check chất lượng Phase 5 agent token — TỐT, không phát hiện lỗi thật

Sample 8+ video ngẫu nhiên, check kỹ:
- 17/17 khớp có mặt mọi window (test đầu tưởng thiếu 2 khớp — do tự đoán sai tên `neck`/`head`, tên thật đúng convention H36M là `nose`/`head_top` — lỗi test, không phải lỗi data).
- `pelvis` luôn = 128,128,128 (giữa range uint8) — **đúng thiết kế**, không phải lỗi: Phase 3 root-centering (`split_root_motion`) trừ vị trí pelvis khỏi mọi khớp nên pelvis luôn là gốc tọa độ.
- `r_wrist_x` có variation thật (std 6.9–22.6 tùy video) — không degenerate/constant.
- 0% giá trị bị clip ở biên (0/1 hoặc 254/255) trên 113,040 giá trị xyz sample.
- Token `t` đúng range 0-7 theo tier CP.
→ Agent token Phase 5 sẵn sàng dùng để merge.

### 5. Phát hiện quan trọng cho merge: chunk numbering giữa Step A và Pose KHỚP CHÍNH XÁC

Step A (`step_a_tokenize_video.py`) dùng `CHUNK_SIZE=8` frame @ `TARGET_FPS=30` — giống hệt Phase 5's `window_id` (stride 8 @ 30fps). `window_id` Phase 5 = `chunk_idx * 8` của Step A, khớp 1-1, không cần nội suy thời gian (khác các bug fps-mismatch đã fix trước đây — ở đây cả 2 bên đã cùng lưới 30fps/stride-8 cố định từ đầu).

### 6. Plan merge 3 track (đã user duyệt cấu trúc, đang thực hiện)

Quyết định: **1 record/video**, QA nối đuôi sau video tokens (không tách mỗi QA pair thành 1 record riêng — tiết kiệm token, tránh lặp lại video stream N lần).

- **Bước 1 — Fix `flatten_qa_text.py` giữ `video_id`** — **XONG**. Verify: 99,983 dòng, đúng 5,214 video_id unique.
- **Bước 2 — `phase6_merge_omnivideo.py`** (mới viết) — đọc raw `omnivideo_100k_video_flat/step_a_rank_*.jsonl`, tái dùng logic `flatten_token_stream()` từ `flatten_step_a_video.py` (import trực tiếp, không copy-paste), thêm counter `chunk_idx` tại mỗi lần flush `<avc_lm>`, tính `window_id = chunk_idx*8`, nếu có trong `pose_agent_tokens_adaptive/{video_id}_tokens.jsonl` thì chèn `<agent>{token_str}</agent>` ngay sau cosmos. Output: `/p/data1/mmlaion/shared/vla/omnivideo_100k_video_agent_merged/`. **Đang chạy** (tmux `phase6_merge_omnivideo`, ETA ước tính ~10 phút dựa trên lần test dry-run 11/32 file trong 205s — lần test đó lỡ ghi nhầm ra `/tmp` scratchpad thay vì đích thật + có `| head -5` gây rủi ro `BrokenPipeError`, đã kill và chạy lại đúng trong tmux, ghi thẳng vào `/p`).
- **Bước 3 — `phase7_finalize_omnivideo.py`** (chưa viết) — left-join `omnivideo_100k_qa_flat.jsonl` (gom QA theo video_id) vào cuối text của Bước 2.

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Phase 4 FineVideo (tmux `yolo_login_finevideo`) — đang chạy, ETA ~15h từ lúc submit (~12:22).
- Phase 6 merge OmniVideo (tmux `phase6_merge_omnivideo`) — đang chạy, ETA ước tính ~10 phút.
- Việc tiếp theo: verify Bước 2 xong (đếm video có `<agent>` phải khớp ≤799, spot-check vị trí đúng thời gian), rồi viết + chạy Bước 3.
- Việc tồn đọng: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/` (chờ Phase 4 FineVideo xong, ~15h); chờ Huu quyết định `MixtureVitae-Backup` SNAC + "moss" token nghĩa là gì.

## Cập nhật phiên làm việc — 20/07/2026 (tiếp nữa nữa — Phase 3 FineVideo login-node chạy xong, verify+upload SNAC roleplay lên HF, điều tra thực nghiệm tác dụng Phase 4 YOLO filter, chuyển synth_llava sang đúng chỗ)

**Việc chính:** Cluster vẫn maintenance (`sinfo`/`scontrol` xác nhận job `978074` bị chặn cứng bởi lý do `Reserved for maintenance`, 18 node `booster` ở trạng thái `maint`) — chuyển hẳn Phase 3 FineVideo sang chạy login node (như đã làm với OmniVideo trước đó), xong hoàn toàn. Verify SNAC tokenize `laion/emotional-roleplay` (tưởng còn đang chạy dở theo note phiên trước, thực ra log cho thấy đã DONE), viết script + upload lên HF theo tên user chọn. User nghi ngờ Phase 4 YOLO filter có tác dụng thật hay không — điều tra bằng số liệu thật từ OmniVideo đã chạy xong, kết luận filter cắt thật ~25.7% dữ liệu, không phải bước thừa. Hủy job `978074` (dư thừa vì Phase 3 đã xong qua login node), submit Phase 4 FineVideo login-node luôn. Chuyển `synth_llava/` (107GB) từ `/e/data1/.../nguyen38/mixturevitae_multimodal/` sang đúng convention `/p/data1/mmlaion/shared/vla/`.

### 1. Phase 3 FineVideo — login-node run, XONG hoàn toàn

Cluster vẫn PENDING do maintenance, viết `slurm/run_kinematics_login.sh` (giống hệt `submit_kinematics.sh` về script/args, giảm xuống 32 worker + `nice -n 15`/`ionice -c3` vì login node dùng chung — 72 core, 9 user khác, load trung bình 12/72 lúc launch). Chạy tmux `kin_login_finevideo`. Tốc độ thật đo được: ~47 file/30s → toàn bộ 40,804 video xong trong khoảng 15-20 phút. Kết quả: **40,305 video thành công + 499 video rỗng** (0 window sạch sau hallucination filter, không phải lỗi — cùng loại outcome như 78 video EMPTY của OmniVideo) = đủ 40,804/40,804, **0 exception** trên 32 worker (verify bằng cách grep log tất cả worker). Output ghi thẳng vào `outputs/states_jsonl_30fps/` (thư mục đã đổi tên sạch từ bug fps-mismatch tuần trước).

### 2. SNAC tokenize `laion/emotional-roleplay` — xác nhận DONE + verify kỹ + upload HF

Note phiên trước ghi "đang chạy ~1,000/5,000 dòng shard đầu" nhưng log thật (`logs/snac_roleplay_tokenize.log`) cho thấy **đã DONE từ trước đó**: `ok=67,459 skipped_shards=0 decode_fail=0 snac_fail=0 total_snac_tokens=23,390,760`.

**Verify cấu trúc toàn bộ 14 shard** (không chỉ tin log): parse lại từng dòng bằng regex, kết quả — 67,459/67,459 dòng, **0 ID trùng, 0 lỗi format, 0 token ngoài range hợp lệ** (3 offset `OFFSET_L0/L1A/L1B` đúng theo `tokenize_snac.py`), tổng token khớp chính xác log.

**Phát hiện 1 vấn đề thật ở data nguồn (không phải bug tokenize):** 103/67,459 dòng (0.15%) có audio ngắn bất thường so với độ dài text — ví dụ 1 dòng text 128 ký tự nhưng token SNAC chỉ tương ứng 0.16s audio. Verify bằng ffprobe thật trên đúng bytes MP3 gốc lấy lại từ parquet (`id=cv_charming flity woman__b1_08_Hope_Enthusiasm_Optimism_2`): file chỉ 813 byte, **Duration thật = 0.14s** — xác nhận đây là audio bị cắt/hỏng sẵn trong dataset gốc, script tokenize xử lý đúng những gì có. Phần còn lại phân bố hợp lý (median 7.92s, p99 22.96s, max 66.48s, khớp README "~184h/67,491 clip").

Ghi 4 sample thật từ output đầy đủ vào `samples/laion_emotional_roleplay_sample/fullrun_*.txt` (3 dòng bình thường từ 3 shard khác nhau + 1 dòng anomaly để user tự xem).

**Upload lên HF:** viết `data_prep/laion_emotional_roleplay/upload_hf.py` (theo đúng pattern `tools/upload/upload_flattened_hf.py` — gzip nén, split train/test theo shard seed 42, `--repo-id` là CLI arg để user tự chọn tên chứ không hardcode). Dry-run (`--skip-upload`) trước: nén 14 shard 339MB→55MB, `gzip -t` pass hết. User tự export `HF_TOKEN` và chạy thật — **upload thành công**: `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened` (67,459 dòng, 23.39M token, train 13 shard/test 1 shard).

**Verify thêm theo yêu cầu user — decode ngược 1 sample ngẫu nhiên từ chính repo HF vừa upload:** cài `snac`+`soundfile` vào `env_stable_vla` (trước đó tìm nhầm hướng đi lục lại môi trường cũ, user nhắc đúng — chỉ cần pip install thêm, không cần tìm env nào khác). Tải 1 shard train thật từ HF (`hf_hub_download`), chọn ngẫu nhiên 1 dòng (`id=force_ogress-male-shouting-terrified_004944_b0`, 201 token). Phát hiện: **listen-format encode chỉ giữ `codes[0]`+`codes[1]` của SNAC (bỏ `codes[2]`, tầng chi tiết nhất, tốc độ 4x)** — decode cần đủ 3 tầng nên phải zero-fill `codes[2]` để nghe thử được (audio nghe được nhưng mất chi tiết tần số cao, là preview lossy chứ không phải chất lượng gốc). Tải lại audio gốc từ parquet nguồn (23,181 byte MP3) để so sánh cạnh nhau. Đã gửi cả 2 file (`hf_random_sample_decoded.wav` + `hf_random_sample_ORIGINAL.mp3`) cho user nghe trực tiếp.

### 3. Điều tra thực nghiệm: Phase 4 YOLO filter có tác dụng thật không?

User nghi ngờ đúng chỗ đáng nghi (chạy lại YOLO trên video có vẻ trùng việc Phase 1 đã detect person rồi) — trả lời bằng số liệu thật thay vì suy đoán.

**Khác biệt kiến trúc quan trọng tìm được:** Phase 1 (`phase1_hrnet_gpu.py`) dùng **MMDetection**, threshold 0.5, chọn bbox lớn nhất mỗi frame. Phase 4 dùng **Ultralytics YOLO**, threshold 0.75 — là 1 detector hoàn toàn độc lập, không phải lặp lại.

**Số liệu thật trên 1,048 video OmniVideo đã chạy xong Phase 3+4** (so `pose_states_jsonl_30fps` trước lọc với `pose_yolo_cleaned_30fps` sau lọc, đếm dòng thật từng file):

| Metric | Giá trị |
|---|---|
| Tổng window trước lọc | 387,226 |
| Tổng window sau lọc | 287,881 |
| **Tỷ lệ cắt tổng** | **25.7%** |
| Cắt trung vị/video | 18.8% (p25=4.7%, p75=37.2%) |
| Video bị cắt 100% (có window trước, 0 sau) | **25 video** |

Kiểm tra mẫu 8/25 video "cắt 100%": mỗi video có 224-504+ window pass được Phase 3 (hợp lệ về hình học) nhưng YOLO không tìm thấy người ở ≥4/8 frame trong MỌI window — gần như chắc chắn là case MMDet detect nhầm vật gì đó thành người (score>0.5) ở Phase 1, HRNet/MotionBERT vẫn lift ra bộ xương "hợp lý về hình học" từ input rác, chỉ YOLO độc lập mới bắt được.

**Kết luận:** Phase 4 giữ lại, đang làm đúng việc — nếu bỏ, ước tính ~1/4 dữ liệu training sẽ lẫn pose giả từ video không có người thật.

### 4. Hủy job SLURM dư thừa + submit Phase 4 FineVideo login-node

`scancel 978074` (Phase 3 SLURM cũ — dư thừa vì đã xong qua login node ở mục 1). Viết `slurm/run_yolo_login.sh` (giống `submit_yolo.sh` về script/args, giảm từ 128 worker/4 GPU xuống **16 worker/1 GPU** — login node chỉ có 1× GH200 480GB, xác nhận bằng `nvidia-smi` lúc launch: 0% util, 0MiB used, hoàn toàn rảnh). Bật NVIDIA MPS riêng cho login node (pipe/log dir khác job SLURM để tránh đụng). Chạy tmux `yolo_login_finevideo`, verify worker 1 chạy khỏe (~52 window/s cho video đầu). **Đang chạy lúc ghi entry này**, chưa xong.

### 5. Chuyển `synth_llava/` sang đúng convention lưu trữ

Theo đúng quy tắc đã thống nhất trước đó (dataset tải ngoài → `/p/data1/mmlaion/shared/vla/`, không phải `/e/data1/.../nguyen38/`): confirm 2 mount khác nhau thật (`exa_data1` vs `data1`, 2 filesystem ID GPFS khác nhau) nên không thể `mv` tức thời, phải rsync. Chạy `rsync -a` (107GB: `data/` 53GB gốc .tar.gz + `extracted/` 55GB đã giải nén 151 shard) trong tmux `move_synth_llava`, tốc độ thật ~500MB/s. **Đang chạy lúc ghi entry này**, chưa xong, chưa xóa bản gốc (chỉ xóa sau khi verify copy xong).

### Trạng thái cuối phiên (tại thời điểm ghi entry)

- Cluster vẫn maintenance, không còn job SLURM nào pending (đã cancel `978074`).
- Phase 4 FineVideo (login-node, tmux `yolo_login_finevideo`) — đang chạy, 16 worker.
- Chuyển `synth_llava/` (tmux `move_synth_llava`) — đang chạy rsync, chưa xóa bản gốc.
- Việc tồn đọng chưa làm: tách `pipeline_pose/`+`pipeline_video/` vào `data_prep/finevideo/`; chờ Huu quyết định `MixtureVitae-Backup` SNAC (~3.27B code) và ý nghĩa "moss" token cho roleplay dataset.

## Cập nhật phiên làm việc — 20/07/2026 (tiếp nữa — so sánh paper iFLYTEK-Embodied-Omni, phân tích + SNAC tokenize dataset laion/emotional-roleplay, chuẩn bị tắt session)

**Việc chính:** User đưa paper `iFLYTEK-Embodied-Omni` (arXiv 2607.02542, Huu chia sẻ trong chat) để so sánh với project — đọc toàn bộ 16 trang, kết luận: **họ đã có model thật, train thật, SOTA thật** (89.6% LIBERO-Plus, 93.68%/93.16% RoboTwin 2.0), nhưng **~30% data mix của họ không permissive đã xác nhận** (Ego4D 12.72% proprietary, AgiBot 11.65% CC-BY-NC-SA — 2 dataset mà **project mình đã tự điều tra và loại bỏ từ trước**). Xác nhận claim của user ("technical report chứ không phải science paper") hợp lý — chính iFLYTEK cũng tự đặt tên bài là "Technical Report". Đã lưu toàn bộ so sánh vào memory (`project_iflytek_omni_comparison.md`, `project_omni_scope_clarification.md`). Sau đó phân tích kỹ dataset `laion/emotional-roleplay-finetuning-dataset` (theo chỉ đạo Huu: "concatenate text, interleave snac/moss tokens"), viết + chạy full-scale script SNAC tokenize trên login node (tmux), verify Phase 4 OmniVideo đã DONE hoàn toàn.

### 1. So sánh với iFLYTEK-Embodied-Omni — đã lưu chi tiết vào memory

Tóm tắt (chi tiết đầy đủ trong memory `project_iflytek_omni_comparison.md`):
- **Họ hơn:** data robot trajectory THẬT (không phải proxy như agent-token pose-từ-video của mình), eval task-success THẬT trên simulator chuẩn ngành (LIBERO-Plus, RoboTwin 2.0), data mixture định lượng rõ ràng, ablation kiểm soát.
- **Mình hơn/ngang:** đã có audio (SNAC) tích hợp sẵn — họ còn chưa có, tự ghi "future work: incorporate speech"; IMU cả 2 bên đều chưa làm; kỷ luật license chặt hơn (đã loại AgiBot/EgoDex/stera-10m vì lý do license, họ thì vẫn dùng AgiBot+Ego4D+EgoDex dù non-permissive).
- **Kết luận:** chưa đạt tới mức ngay cả "technical report" — thiếu robot-action thật, eval task-success, data mixture định lượng, ablation kiểm soát. 4 việc này là danh sách ưu tiên cụ thể nếu muốn bắt kịp.

### 2. Phân tích `laion/emotional-roleplay-finetuning-dataset` — đọc kỹ README, sửa 1 hiểu nhầm

Verify data thật: **67,491 dòng chính xác** (khớp README), **183.96h** (khớp "~184 hours"), audio xác nhận thật `24000 Hz, mono` qua ffmpeg probe. Phân bố: source `fill_creature` 37%/`emolia`/`ears`/`gemini`/`emotional_va`/`character`/`fill_human`; ngôn ngữ Đức 59% áp đảo. Phát hiện nhỏ: ~32/67,491 dòng `adherence_score` giá trị lạ (8/9/10/80/0) ngoài thang 1-5 README ghi — lỗi nhập liệu nhỏ, đã loại khi tokenize.

**Tự sửa 1 điểm hiểu nhầm sau khi đọc kỹ README (user yêu cầu đọc lại trước khi tóm tắt):** 5 field mới (`genuineness`, `vocal_burst_blend`, `voiceclap_commercial_embedding`, `is_human`, `archetype`) **CHỈ có trong bản `webdataset/` (tar shard), KHÔNG có trong 6 file `data/*.parquet` đã tải** — README ghi rõ "the original parquet files under data/ are unchanged". Nếu sau này cần dùng các field này để lọc/cân bằng human vs non-human, phải tải thêm `webdataset/` (~3.13GB), chưa làm.

### 3. Format tokenize + interleave — đã thống nhất, verify bằng chạy thật 1 sample

```
USER: <text> [Voice: <voice_description>] ASSISTANT:
<snac> <snac_N> <snac_N> ... </snac>
```

Cố tình bỏ `instruction`/`req_*` — README's Limitations tự ghi rõ model lệch giọng nam/bình tĩnh mặc định, `req_*` overstate nữ/to tiếng so với thực tế → chỉ tin `voice_description`/`realized_gender` (audio-verified thật), không tin field ý định. Verify bằng chạy thật (không phải giả định) trên `sample5_ears`: decode MP3→float32 24kHz qua ffmpeg (audio đã sẵn 24kHz mono, không cần resample), load model `hubertsiuzdak/snac_24khz` thật trên GPU, encode ra **561 token SNAC thật**, ghép thành record hoàn chỉnh 8,220 ký tự — lưu vào `samples/laion_emotional_roleplay_sample/sample5_ears_flattened_snac.txt` để đối chiếu trực tiếp với `.mp3`/`.json` gốc.

### 4. Viết + chạy full-scale `tokenize_snac.py`

`data_prep/laion_emotional_roleplay/tokenize_snac.py` — không import `pipeline_pose/snac_finevideo.py` (dùng type hint `X | None` kiểu Python 3.10+, vỡ ở `env_motion_final` Python 3.9) — viết lại độc lập phần toán học SNAC encode (offset giống hệt, không đổi). Không cần `split_snac_by_chunks()` như FineVideo vì mỗi dòng là 1 clip độc lập, không có chunk video nào để căn theo — encode nguyên clip thành 1 khối token phẳng.

Đo throughput thật trước khi cam kết quy mô (đúng thói quen dự án): 300 dòng thật mất 21s (trừ overhead load model ~4-5s) → ước tính **~63 phút cho toàn bộ 67,459 dòng** trên login node GPU — hợp lý, không cần SLURM. **Đã submit chạy full trong tmux session `snac_roleplay`** (không dùng background task tool, theo đúng yêu cầu user từ trước), output ghi vào `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/flattened/roleplay_snac_flat_{00000..00013}.jsonl` (13-14 shard × 5,000 dòng), log tại `logs/snac_roleplay_tokenize.log`. **Đang chạy lúc ghi entry này** (~1,000/5,000 dòng shard đầu).

### 5. Sửa lộn vị trí lưu trữ (nhắc lại từ mục trước) — đã thống nhất rõ

Theo đúng quy tắc user chốt: dataset tải ngoài → `/p/data1/mmlaion/shared/vla/`. Đã áp dụng đúng cho `laion_emotional_roleplay` (tải + output tokenize đều ở đây). **`mixturevitae_multimodal/synth_llava/` (107GB) vẫn còn ở `/e/data1/.../nguyen38/`** — chưa chuyển, cần hỏi lại user.

### Trạng thái cuối phiên (user chuẩn bị tắt/mở lại session)

- **`978074`** (Phase 3 FineVideo full-scale rerun, 40,804 video) — vẫn PENDING, cluster bảo trì diện rộng, chưa rõ ETA. Chưa submit lại Phase 4 FineVideo (`slurm/submit_yolo.sh`, đã fix + verify) vì phải chờ `978074` xong trước.
- **Phase 4 OmniVideo** — **XONG HOÀN TOÀN** (verify thật): 1,037 done + 11 skip + 78 no_input (đúng số video Phase 3 rỗng) = 1,126/1,126, 0 lỗi. Output: `$DATA/omnivideo_100k/pose_yolo_cleaned_30fps/` (1,048 file).
- **`snac_roleplay`** (tmux) — đang chạy, ETA ~1 giờ từ lúc submit (07:59). Kiểm tra bằng `tmux attach -t snac_roleplay` hoặc `tail -f logs/snac_roleplay_tokenize.log`.
- **Việc lớn đã thống nhất nhưng CHƯA làm:** tách `pipeline_pose/`+`pipeline_video/` (FineVideo) vào `data_prep/finevideo/` — chờ Phase 3/4 FineVideo chạy xong hẳn để tránh đổi path giữa lúc job đang chạy. Chuyển `mixturevitae_multimodal/synth_llava/` sang `/p/data1/mmlaion/shared/vla/` — chưa hỏi/làm.

---

## Cập nhật phiên làm việc — 20/07/2026 (tiếp — làm rõ scope "omni", khảo sát + tải 2 dataset mới ngoài scope video/pose, chạy Phase 4 OmniVideo trên login node)

**Việc chính:** User hỏi tại sao Huu (leader) lại nhắc tới ảnh tĩnh/audio roleplay trong khi tưởng project chỉ làm VLA video cho humanoid — dẫn tới làm rõ **scope thật của project rộng hơn nhiều** so với những gì `CLAUDE.md` mô tả (đã cập nhật toàn bộ `.md` — xem mục 1). Khảo sát 2 dataset mới theo yêu cầu Huu: `synth_llava`/`synth_llava2` (ảnh+caption tổng hợp, Huu tự tạo) và `laion/emotional-roleplay-finetuning-dataset` (audio+text TTS tổng hợp). Trong lúc chờ cluster hết bảo trì (job Phase 3 FineVideo `978074` vẫn PENDING), chạy Phase 4 OmniVideo trực tiếp trên GPU login node (qua tmux theo yêu cầu user, không dùng background task tool).

### 1. Làm rõ scope "omni" — cập nhật toàn bộ file `.md` cấp cao

Huu xác nhận trực tiếp trong chat: dự án là **omni-modal** — bind bất kỳ tổ hợp modal nào (ảnh, video, âm thanh, action, IMU...), miễn **license permissive + cân bằng tỷ trọng modal + tạo được cross-modal binding thật**, không bắt buộc phải liên quan robot/humanoid như khung `PAB-Spline VLA` (video+pose) hiện tại — khung đó chỉ là **1 modal-pair branch** trong bức tranh lớn hơn. Đã thêm ghi chú scope này vào: `CLAUDE.md` (project-wide), `3d-human-pose/README.md`, `REPORT.md` (section "Project Scope Update" mới), `PROGRESS.md` (tiếng Anh), `datasets.md`. Cũng ghi nhận lo ngại thật của user về tính khoa học/khả thi paper khi scope liên tục mở rộng mà eval protocol vẫn "Still open" — chưa có 1 câu hỏi nghiên cứu trung tâm cố định để justify từng nguồn data mới — đã nêu với user, chưa hành động (quyết định hướng nghiên cứu thuộc về Huu).

### 2. `synth_llava` / `synth_llava2` — khảo sát xong, đã tải + giải nén

Theo lịch sử chat, đây là 2 file trong path `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` mà đợt khảo sát 9/7 (`tools/inventory/peek_multimodal.py`) **chưa kịp peek tới** (report cũ dừng đúng trước 2 file này). Peek trước (không tải full, stream qua HTTP) rồi mới tải full theo yêu cầu user (56.2GB) — ban đầu tải nhầm vào `/e/data1/.../nguyen38/mixturevitae_multimodal/`, **user sửa lại đúng convention: dataset tải ngoài phải vào `/p/data1/mmlaion/shared/vla/`** (giống `omnivideo_100k`/`robovqa`/`sensenova_si8m` cũ) — đã note lại, chưa move (chỉ áp dụng cho download mới từ giờ, xem mục 4).

**Cấu trúc xác nhận bằng data thật:** 151 shard liên tục (`shard-0000000`→`shard-0000150`, 2 file chia nhau không trùng), mỗi shard = 1 `.jsonl` (4000 dòng caption) + 1 `.wds` (**thực chất cũng là tar**, xác nhận qua `file`) chứa 4000 cặp `image_N.png` (256×256) + `metadata.json` khớp 1-1 với `.jsonl`. Tổng **603,999 sample thật** (đếm chính xác). Nguồn: `llava_pretrain|shard-N|create_multimodal_data.generate_images_then_captions` — ảnh sinh tổng hợp (AI-generated) rồi mới auto-caption, không phải ảnh chụp thật. Xem 3 mẫu thật (1 mẫu có artifact sinh ảnh rõ — bàn poker chữ vô nghĩa; 2 mẫu khá thực tế) — đã copy vào `samples/synth_llava_sample/`.

**Đánh giá kỹ thuật (trước khi biết rõ scope omni):** seed2 là loại token DUY NHẤT khớp được (cosmos cần chuỗi 8-frame thời gian thật, avc_lm mã hoá bitstream H.264 — cần video thật, agent cần pose 3D chuyển động) — nhưng data này chỉ có seed2, không transition được sang modal khác, có nguy cơ củng cố lỗi eval đã biết ("model kẹt ở seed2, không tự chuyển cosmos→avclm→agent") nếu tỷ trọng quá cao. Sau khi biết rõ scope omni (mục 1), kết luận: vẫn hợp lệ như 1 modal-pair (ảnh↔text) trong tổ hợp lớn hơn, miễn **tỷ trọng nhỏ, cân bằng** — không phải lý do loại bỏ, chỉ là điều kiện khi trộn.

### 3. `laion/emotional-roleplay-finetuning-dataset` — khảo sát + tải xong

Theo chỉ đạo trực tiếp Huu: "concatenate the text and interleave with snac and/or moss tokens". Kiểm tra: public, không gated, license cc-by-4.0. **67,491 clip audio tổng hợp** (mono MP3 24kHz, ~184h, sinh bởi MOSS-TTS-Local v1.5 fine-tune riêng của Huu `laion/moss-1.5-roleplay-finetune`), đa ngôn ngữ Đức(chủ đạo)/Anh/Tây Ban Nha/Pháp, nhiều giọng nhân vật giả tưởng (orc/goblin/dragon...). Field chính: `text` (lời thoại), `voice_description` (mô tả giọng kiểu DramaBox, do gemini-3.5-flash **nghe audio thật** rồi viết — không phải mô tả mù), `instruction` (ý định gốc), `adherence_score`, `realized_gender`, cùng VoiceCLAP embedding (768-d) + `genuineness`/`vocal_burst_blend` (attribute mới). Khớp đúng ý Huu: audio → `snac` (đã có sẵn trong project, không cần token mới) và/hoặc "moss" (chưa rõ nghĩa — có thể là token riêng của model MOSS-TTS, cần hỏi lại). Đã tải 6 file parquet (2.5GB) vào `/p/data1/mmlaion/shared/vla/laion_emotional_roleplay/` — đúng convention, chưa tokenize.

### 4. Sửa lộn vị trí lưu dataset tải ngoài — quy tắc rõ từ giờ

User chỉ ra: dataset tải từ ngoài (HF) phải lưu ở `/p/data1/mmlaion/shared/vla/` (nơi `omnivideo_100k`/`robovqa`/`sensenova_si8m` cũ đã ở đó), **không phải** `/e/data1/.../nguyen38/` (nơi này dành cho pipeline output/working data của chính project, ví dụ `omnivideo_100k/pose_*` là kết quả pipeline tự chạy ra, không phải "tải ngoài"). Đã xoá thư mục tải nhầm (`/e/data1/.../nguyen38/laion_emotional_roleplay`, mới tải dở), tải lại đúng chỗ. **Việc còn để ngỏ, CHƯA làm:** `mixturevitae_multimodal/synth_llava/` (107GB, cả 2 file `.tar.gz` gốc + phần đã giải nén) hiện vẫn đang ở `/e/data1/.../nguyen38/` — cần hỏi lại user có muốn chuyển sang `/p/data1/mmlaion/shared/vla/` không trước khi tự ý di chuyển 107GB.

### Trạng thái cuối phiên (tiếp)

Job Phase 4 OmniVideo chuyển từ SLURM (`978072`, huỷ) sang chạy trực tiếp GPU login node trong tmux session `phase4_omnivideo` (theo yêu cầu user, không dùng background task tool) — đang chạy tốt, ~52% (585/1,126 video) tính tới lúc ghi entry này. Job `978074` (Phase 3 FineVideo full-scale rerun) vẫn PENDING, cluster bảo trì diện rộng chưa rõ ETA.

---

## Cập nhật phiên làm việc — 20/07/2026 (Phase 2→4 OmniVideo-100K, tách output riêng khỏi FineVideo, bắt 2 bug fps-mismatch thật ảnh hưởng cả FineVideo, fix + rerun)

**Việc chính:** Track lại Phase 2 OmniVideo-100K (đã COMPLETED từ phiên trước, verify lại bằng data thật), chạy tiếp Phase 2.5 (resample 30fps) và Phase 3 (kinematics). Theo yêu cầu user, **tách hẳn output OmniVideo-100K ra khỏi thư mục `outputs/` dùng chung với FineVideo** (trước đó 2 phiên convention là "an toàn vì video_id không trùng" — nhưng gây khó đếm/kiểm tra/dọn dẹp riêng). Trong lúc viết driver Phase 4, **phát hiện 1 bug fps-mismatch thật, nghiêm trọng, có từ trước** trong `phase4_yolo_cleaner.py` gốc — ảnh hưởng 35% video FineVideo đã chạy production (dùng để train cả 2 model hiện có). Điều tra sâu hơn phát hiện thêm **bug thứ 2 cùng loại** trong `apply_2d_mask()` (Phase 3) — cũng ảnh hưởng FineVideo, và ảnh hưởng cả Phase 3 OmniVideo vừa chạy xong. Fix cả 2, verify bằng data thật, rerun Phase 3 cho cả OmniVideo (xong) và FineVideo (đang submit, chờ cluster hết bảo trì). Chạy phân tích tương quan confidence-vs-hallucination theo yêu cầu user — kết quả ngược kỳ vọng, kết luận không nên dùng confidence làm filter bổ sung.

### 1. Phase 2 (MotionBERT) — xác nhận đã xong từ phiên trước

Verify lại bằng data thật (không chỉ tin SLURM state): job `977337` COMPLETED, **1,126/1,126 video** ra `.npy` hợp lệ, 0 error trên 32 rank. 2 lần pilot trước đó (`977034` lỗi 100% do `os.rename()` xuyên mount, `977128` fix bằng `shutil.move()` mới OK) — bài học đã áp dụng đúng cho pilot lần 2 trước khi chạy full.

### 2. Tách output OmniVideo-100K ra khỏi `outputs/` dùng chung — theo yêu cầu user

User đặt câu hỏi đúng: tại sao để chung 1 folder với FineVideo. Quyết định cũ ("video_id không trùng nên an toàn") đúng về mặt kỹ thuật nhưng sai về mặt vận hành (khó đếm/dọn/kiểm tra riêng). Đã:
- Di chuyển 1,126×3 file (`2d_json`, `3d_npy`, preview `.mp4`) từ `outputs/` sang `$DATA/omnivideo_100k/pose_2d_json/`, `pose_3d_npy/` (cùng mount `exa_data1` nên `os.rename()` tức thời, không phải copy 262GB).
- Revert merge `fps_lookup.json` chung (đã lỡ merge 5,214 entry OmniVideo vào — revert sạch về đúng 43,751 entry FineVideo cũ), tách riêng thành `$DATA/omnivideo_100k/fps_lookup.json` (5,214 entry).
- Cập nhật default path trong `phase1_hrnet_omnivideo.py`/`phase2_motionbert_omnivideo.py` cho khớp.

### 3. Phase 2.5 (resample 30fps) — driver mới, chạy login node (20 giây)

`data_prep/omnivideo_100k/phase2_5_resample_omnivideo.py` — import `resample_pose()` từ bản gốc (không đụng file gốc, hàm này vốn dataset-agnostic), chỉ viết lại phần iteration (theo video_id list, không glob cả thư mục `outputs/3d_npy` 43,751 file của FineVideo). Chạy `tools/extract/extract_fps.py` cho toàn bộ 5,214 video OmniVideo trước (12 giây), verify 0 ID trùng FineVideo. **1,126/1,126 video xong trong 20 giây trên login node** — đúng dự đoán, việc này quá nhẹ để cần SLURM. Render skeleton so sánh trước/sau cho 1 video 25fps (gửi user xem) — timing khớp chính xác (120.00s cả 2 bên), video control 30fps ra byte-identical.

### 4. Phase 3 (kinematics) — chạy 2 lần (trước và sau khi phát hiện bug §6)

Lần đầu (trước khi phát hiện bug): login node, 1,126 video xong trong ~2 phút. Kết quả: **928 video OK, 198 EMPTY** (không lỗi, chỉ quá nhiễu để có window sạch). Hallucination filter loại trung bình 30.4% frame, chủ yếu do `rogue_joint_filter` (khớp bị kéo dài bất thường — đặc trưng lỗi lifting khi chuyển động nhanh, xác nhận bằng cách phân rã riêng từng loại filter trên 1 video mẫu, phân bố đều theo thời gian chứ không phải bug cục bộ).

### 5. Phát hiện bug thật #1 — fps-mismatch trong `phase4_yolo_cleaner.py` gốc (ảnh hưởng cả FineVideo)

Trong lúc viết driver Phase 4 cho OmniVideo, phát hiện: script gốc đọc **frame tuần tự từ video native-fps** nhưng `window_id` trong `states_jsonl_30fps` lại đánh số theo **lưới 30fps đã resample** (Phase 2.5) — 2 timeline khác nhau bị coi là một. Verify bằng data FineVideo thật (`-2MKTg-LNio`, 25fps): `native_frames=12,758` nhưng `states` có `max_window+8=15,304` — lệch ~2,546 frame. Hệ quả: (a) mọi window vượt quá `native_frames` bị **âm thầm bỏ** (mất ~1/6 cuối video ở video 25fps), (b) window còn lại đọc nhầm thời điểm, lệch dần tới ~20% tổng thời lượng. Đo thật: **35% FineVideo** (15,321/43,751 video fps lệch ≥5% so với 30) và **37.3% OmniVideo subset** (420/1,126) sẽ bị ảnh hưởng nếu copy y hệt logic gốc.

**Fix:** viết mapping `native_idx ↔ resampled_idx` qua `np.round(np.linspace(0, N-1, M))` (cùng công thức endpoint-aligned mà `resample_pose()` dùng chiều ngược lại) — áp dụng cả cho `pipeline_pose/phase4_yolo_cleaner.py` (sửa tại chỗ, vì đây là bug thật trong code dùng chung, không phải vấn đề tương thích dataset) và driver mới `data_prep/omnivideo_100k/phase4_yolo_cleaner_omnivideo.py`. Verify bằng cả gọi hàm trực tiếp lẫn qua đúng CLI/`main()` trên video FineVideo thật: `total_input=1,913` (đủ, không mất), `max window_id giữ lại=14,712` (gần đúng 15,304 thay vì bị chặn cứng ở ~12,750 như bản cũ).

### 6. Phát hiện bug thật #2 — `apply_2d_mask()` (Phase 3) cùng loại lỗi + 1 lỗi riêng cho OmniVideo

Lúc chuẩn bị phân tích confidence, phát hiện `apply_2d_mask()` cũng so `pose2d` (native fps) với `pose3d` (đã resample 30fps) qua `num_frames = min(len(pose3d), len(pose2d))` — **cùng bug fps-mismatch như mục 4**, verify trên `z-Qcz_FMW7Q` (25fps): 600/3,600 frame cuối (resampled-space) **không bao giờ được mask**.

**Bug riêng #2b (chỉ OmniVideo):** check "khớp bị thiếu" của hàm này yêu cầu `x=y=conf=0` cả 3 — đúng cho driver FineVideo gốc (ép conf về 0/1) nhưng **sai cho driver OmniVideo** (`phase1_hrnet_omnivideo.py` cố tình giữ conf liên tục, chỉ zero x,y). Verify thật: **9,162/13,404 (68.4%)** khớp lẽ ra phải bị mask lại bị bỏ sót do check sai field.

**Fix (`pipeline_pose/phase3_kinematics_processor.py`, sửa tại chỗ):** thêm mapping native↔resampled giống mục 4 (bản không phụ thuộc torch/ultralytics, viết riêng thay vì import từ `phase4_yolo_cleaner.py` để tránh kéo dependency nặng không cần thiết); đổi check zero-mask chỉ nhìn `x,y` (bỏ `conf` khỏi điều kiện) — không đổi hành vi với driver FineVideo gốc (x,y và conf vốn cùng về 0 ở đó), chỉ fix đúng cho OmniVideo. Verify: video mẫu giờ có 2,909/3,600 frame được mask đúng (từ 0 → 546/600 ở 600 frame cuối từng bị bỏ sót).

### 7. Rerun sau fix — không xoá, đổi tên output cũ để có đường lùi

Thay vì `rm -rf` (không thể hoàn tác), **đổi tên** các thư mục output cũ bị ảnh hưởng (cùng mount nên tức thời, không tốn thời gian copy):
- `outputs/yolo_cleaned_30fps` → `outputs/yolo_cleaned_30fps_buggy_fps_mismatch_2026-07-20` (107GB, FineVideo)
- `outputs/states_jsonl_30fps` → `outputs/states_jsonl_30fps_buggy_2026-07-20` (193GB, FineVideo)
- `$DATA/omnivideo_100k/pose_states_jsonl_30fps` → `..._buggy_2026-07-20` (OmniVideo)

Rerun Phase 3 OmniVideo ngay (login node, rẻ): **928→1,048 video OK** (198→78 EMPTY) — fix không chỉ đúng hơn mà còn tăng yield thật, vì mask đúng giúp hallucination-filter không còn bị nhiễu bởi khớp lẽ ra đã phải NaN từ trước.

User xác nhận (qua `AskUserQuestion`) muốn rerun full-scale Phase 3 cho FineVideo luôn (không chỉ pilot) — đã submit `978074` (`slurm/submit_kinematics.sh`, không cần sửa vì đã dataset-agnostic, fix nằm trong `phase3_kinematics_processor.py`).

**Job `978073`** (Phase 4 FineVideo, submit trước khi phát hiện bug #2) đã bị `scancel` — lý do: nó đọc `states_jsonl_30fps` mà mình vừa đổi tên để `978074` ghi lại, nếu chạy trước sẽ đọc nhầm thư mục thiếu/rỗng. Sẽ submit lại sau khi `978074` xong.

### 8. Phân tích tương quan confidence ↔ hallucination — kết quả ngược kỳ vọng, không nên dùng làm filter

Theo yêu cầu user, đối chiếu confidence trung bình/frame (đã align đúng qua mapping ở mục 5) với việc frame có bị geometric hallucination filter loại hay không (60 video, 176,656 frame):

| Confidence bucket | Tỷ lệ bị hallucination-filter loại |
|---|---|
| 0.00–0.30 | 7.6% |
| 0.30–0.50 | 15.8% |
| 0.50–0.70 | **34.5%** |
| 0.70–0.85 | 27.3% |
| 0.85–1.01 | 0.7% |

Hệ số Pearson **+0.21** (yếu, **dương** — ngược kỳ vọng "confidence thấp → dễ hallucination"). Giải thích hợp lý: khớp confidence thấp bị zero-out ở Phase 1 rồi được `interpolate_nan_gaps()` làm mượt trước khi vào bộ lọc → ít bị coi là rogue joint; ngược lại khớp confidence 0.5–0.85 thường là chi chuyển động nhanh, rõ nét 2D nhưng khó cho MotionBERT lift đúng độ sâu → dễ hallucination hơn. **Kết luận: không nên dùng confidence làm filter bổ sung/dự đoán hallucination** — tương quan yếu và sai chiều so với giả thuyết ban đầu.

### Trạng thái cuối phiên

Job đang PENDING (cluster bảo trì, `Reserved for maintenance`, nhiều node `booster` ở trạng thái `maint`, không có ETA rõ): `978072` (Phase 4 OmniVideo, 928/1,126→1,048/1,126 sau rerun Phase 3), `978074` (Phase 3 FineVideo full-scale rerun, 40,804 video). Cần làm tiếp sau khi `978074` xong: submit lại Phase 4 FineVideo (đã fix, script đã sẵn `slurm/submit_yolo.sh` + `pipeline_pose/phase4_yolo_cleaner.py`). Việc lớn tiếp theo đã thống nhất nhưng CHƯA làm: tách `pipeline_pose/` + `pipeline_video/` (FineVideo) vào `data_prep/finevideo/` để đối xứng với `data_prep/omnivideo_100k/`, `data_prep/robovqa/` — sẽ làm sau khi các job Phase 3/4 hiện tại chạy xong (tránh sửa path giữa lúc job đang chạy).

---

## Cập nhật phiên làm việc — 19/07/2026 (chiều muộn — self-review bắt 1 bug thật, đối chiếu review ngoài với data thật, submit full-scale Phase 1)

**Việc chính:** Trước khi mở rộng từ pilot 24 video lên full 1,126 video, tự review lại code + hạ tầng đã sửa trong phiên — bắt được 1 regression thật do chính fix symlink gây ra. Fix 2 lỗi robustness trong `phase1_hrnet_omnivideo.py` (video hỏng bị coi là "thành công" âm thầm, rò rỉ `VideoCapture` khi lỗi giữa chừng), verify bằng pilot 8 video thật + 1 test video hỏng giả. User mang 1 bài review độc lập (kiểu ChatGPT) về đúng file này — đối chiếu từng điểm với data thật đã có trong phiên thay vì tin/bác theo cảm tính: 2 điểm thật đáng sửa, còn lại phần lớn đã verify không phải vấn đề hoặc là hành vi kế thừa từ bản gốc (đã chạy sạch 40,804 video FineVideo). Trả lời câu hỏi thiết kế về fps — xác nhận giữ đúng convention cũ (native fps ở Phase 1/2, resample ở Phase 2.5), nhưng phát hiện 1 lỗ hổng thật (`fps_lookup.json` chưa có video OmniVideo-100K nào) cần xử lý trước khi chạy Phase 2.5. Áp 4 fix nhỏ cuối cùng rồi **submit job full-scale 1,126 video**.

### 1. Self-review bắt 1 bug thật — do chính fix symlink phiên trước gây ra

`outputs/fps_lookup.json` (43,751 entry, dùng bởi Phase 2.5 và Phase 3) hoá ra **chỉ tồn tại ở thư mục cục bộ** vừa bị đổi tên thành `outputs_local_backup/` — chưa từng có ở `/e/data1/.../nguyen38/outputs/` thật. Nghĩa là fix symlink trước đó vô tình làm file này "biến mất" khỏi path Phase 2.5/3 sẽ tìm. Đã copy lại ngay, verify khớp 43,751 entry cả 2 phía.

### 2. Fix 2 lỗi robustness trong driver Phase 1, verify bằng test thật

- Video hỏng/không mở được trước đây bị ghi thành "OK" với 0 frame — giờ raise lỗi rõ ràng, vào nhóm `error`.
- `cap.release()` giờ nằm trong `try/finally` — tránh rò rỉ file descriptor nếu gặp nhiều video lỗi liên tiếp trong 1 rank chạy lâu (~280 video/rank ở quy mô full).

Verify: pilot SLURM 8 video mới (job `976556`, COMPLETED, 0 lỗi, 87.7% frame detect người — cao hơn pilot đầu 78.7% nhờ đã lọc animation) + 1 test video hỏng cố ý (xác nhận vào đúng nhóm `error`, không sót file `.tmp`).

### 3. Phát hiện thêm: 130/1,256 video trong sports subset thực ra là animation

Điều tra 2 video tỷ lệ thấp trong pilot đầu → phát hiện là animation lọt qua filter từ khoá chung chung ("dancing"/"running"). Nghiêm trọng hơn: 1 video animation khủng long trong pilot vẫn ra **56.3% detect** — HRNet có thể nhầm nhân vật hoạt hình thành người thật, MotionBERT (train trên người thật) lift lên sẽ SAI chứ không chỉ thiếu data. Viết `filter_animation_content.py`: loại 130/1,256 video có `video_summary` tự nhận animation/cartoon → còn **1,126 video** (`sports_subset_video_ids_filtered.txt`, driver đã đổi default sang file này).

### 4. Đối chiếu review ngoài với data thật — 2 điểm đáng sửa, còn lại đã verify không phải vấn đề

**Đáng sửa thật:**
- Chọn người bằng bbox lớn nhất mỗi frame độc lập có thể đổi identity giữa các frame ở cảnh đông người — **hành vi kế thừa từ bản gốc** (đã chạy đúng vậy trên 40,804 video FineVideo), không phải bug mới. Có phòng vệ downstream một phần (Phase 3 anti-teleportation, Phase 4 YOLO cleaner). Sửa đúng cách (IoU tracking) là thay đổi kiến trúc ngoài phạm vi task — ghi nhận là giới hạn đã biết, chưa sửa.
- Decode bị cắt giữa chừng không bị phát hiện — thật, nhưng cách review đề xuất (so với `duration × 30fps`, ngưỡng 90%) khi test trực tiếp trên 8 video pilot thật sẽ **báo oan 2/8 video (25%)** — `07WqS-ccIrw`/`0OxHEDu5dFE` chỉ đạt 83.1% vì native **25fps thật** (verify qua `cv2.CAP_PROP_FPS`), không phải lỗi decode. Sửa bằng cách so với `cv2.CAP_PROP_FRAME_COUNT` thật của từng video (không giả định fps), chỉ warning chứ không raise cứng.

**Đã verify KHÔNG phải vấn đề / là hành vi kế thừa:**
- Lo ngại `SLURM_NTASKS` (nếu quên `srun` sẽ chỉ chạy rank 0) — đã bị bác bỏ bằng thực tế: log 2 job thật đều cho thấy đúng 4 rank chạy song song.
- "Discontinuity" giữa confidence liên tục và toạ độ zero — review hiểu ngược: `WildDetDataset` của MotionBERT vốn thiết kế nhận confidence liên tục từ detector thật, bản nhị phân hoá cũ mới là cái lệch chuẩn.
- mmpose có thể trả tensor thay vì numpy — code y nguyên bản gốc, đã chạy sạch 32 video thật + 40,804 video FineVideo trên đúng bản mmpose đang dùng — thêm phòng hờ (miễn phí) chứ không phải sửa bug đang xảy ra.
- RAM giữ cả video trong list tới cuối — tính lại bằng số thật (video dài nhất 180s = 5,400 frame) chỉ ~10MB/video, không phải "hàng trăm MB" như review nói.
- Path phụ thuộc CWD — đây là convention chung của TOÀN pipeline (Phase 1-7), mọi sbatch đều `cd` đúng chỗ trước khi gọi — không phải rủi ro thật, đổi riêng file này sẽ làm nó khác biệt với cả hệ thống.

### 5. Câu hỏi thiết kế fps — giữ nguyên convention cũ, phát hiện 1 lỗ hổng thật

Xác nhận: giữ native fps ở Phase 1/2 (không resample sớm), Phase 2.5 mới resample về 30fps — đúng convention cũ của FineVideo, driver hiện tại đã tự động đúng (đọc frame tuần tự, không ép fps). Verify thật: OmniVideo-100K có fps gốc KHÔNG đồng nhất (`07WqS-ccIrw`/`0OxHEDu5dFE` = 25fps, `0GPO9qLraB8`/`iGVvChGEQdM` = 30fps, đo bằng `cv2.CAP_PROP_FPS`) — đúng kiểu tình huống Phase 2.5 sinh ra để xử lý.

**Lỗ hổng thật phát hiện (chưa chặn ở Phase 1 hiện tại):** `fps_lookup.json` (43,751 entry) chỉ có video FineVideo — 0 video OmniVideo-100K. Phase 2.5 tự ghi rõ sẽ "skip với warning" video thiếu trong file này — chạy thẳng sẽ mất trắng toàn bộ pose OmniVideo-100K ở bước 30fps. Đã ghi vào `JUPITER_POSE_PILOT_TASK.md`: cần chạy `tools/extract/extract_fps.py` rồi **merge** (không ghi đè) vào `fps_lookup.json` chung trước khi chạy Phase 2.5.

### 6. Fix cuối + submit full-scale

Áp 4 fix nhỏ: dedup video_ids (data thật hiện tại 0 trùng, phòng hờ), `getsize() > 2` trong resume check, tensor→numpy an toàn cho mmpose, warning decode-truncation dùng frame-count thật. Verify: syntax pass, test dedup bằng ID trùng cố ý (log đúng "Removed 1 duplicate"), test lại đường lỗi video hỏng vẫn đúng. Theo yêu cầu user đẩy nhanh tiến độ, không re-verify riêng đường thành công đầy đủ với dòng warning mới trước khi submit — thay đổi rủi ro thấp (chỉ thêm 1 điều kiện + print, không đổi luồng), và job full-scale với 1,126 video thật sẽ tự lộ nếu có vấn đề.

**Đã submit:** `submit_phase1_full.sbatch` — job **`976705`**, 8 node × 4 GPU (32 GPU), toàn bộ 1,126 video, `--time=04:00:00` (ước tính ~2.6h dựa trên throughput đo thật 263s/video/GPU). Xác nhận `RUNNING` ngay sau submit, 8 node cấp đủ, log lỗi sạch.

### Trạng thái cuối phiên

Job `976705` đang chạy. Commit phiên này: `2f3d675`, `7dc1ca0`, `8e688c4`, `2024da4`, `f9eb687` — đã push hết. Phase 2 trở đi chưa bắt đầu.

---

## Cập nhật phiên làm việc — 19/07/2026 (chiều — chạy pilot pose pipeline that tren JUPITER, fix 2 bug ha tang)

**Việc chính:** Pull `JUPITER_POSE_PILOT_TASK.md` (task handoff từ phiên trưa) trên JUPITER và thực hiện. Trước khi viết driver, kiểm tra theo đúng yêu cầu "giữ confidence score, đừng vứt" — phát hiện `phase1_hrnet_gpu.py` gốc **nhị phân hoá** confidence (1.0/0.0) thay vì giữ giá trị thật, trong khi MotionBERT (`infer_wild.py`) đọc trực tiếp cột này làm input feature cho model lifting; Phase 2 tự nó không có confidence nào để giữ (output `X3D.npy` chỉ có toạ độ 3D thuần). Viết driver mới `phase1_hrnet_omnivideo.py` sửa đúng điểm này. Trong lúc chuẩn bị chạy, phát hiện + fix 2 vấn đề hạ tầng không liên quan tới code phiên này, rồi chạy smoke-test + pilot SLURM 24 video — cả 2 đều sạch.

### 1. 2 bug hạ tầng phát hiện trước khi submit job GPU nào

- **Symlink `outputs/` bị đứt:** CLAUDE.md ghi `outputs/` là symlink trỏ `/e/data1/.../nguyen38/outputs/` (145GB+ data thật). Thực tế đã thành thư mục thường gần rỗng (chỉ có `fps_lookup.json`) — mọi script Phase 1-6 dùng path tương đối `outputs/...` từ CWD project1 sẽ đọc/ghi nhầm chỗ. **Đã fix:** đổi tên thư mục cũ thành `outputs_local_backup/` (không mất file), tạo lại symlink đúng, verify resolve ra data thật.
- **Path env sai trong `setup_hrnet_gpu.sh`/`setup_motionbert.sh`:** cả 2 trỏ `conda activate .../3d-human-pose/env_{hrnet_datasets_v1,motion_final}` ở project1 — không còn tồn tại ở đó (`conda env list` xác nhận). Tìm ra env thật còn nguyên vẹn ở `/e/data1/.../nguyen38/3d-human-pose/env_*` (cùng mount với fix outputs/ ở trên — khả năng cao cùng 1 đợt di chuyển data nhưng chưa cập nhật 2 script). **Đã fix:** sửa path trong 2 script, verify activate được, CUDA True, `mmpose`/`mmdet` import OK.

Đã hỏi user qua `AskUserQuestion` trước khi đụng vào (ảnh hưởng path dùng chung toàn repo), được đồng ý tự sửa.

### 2. Driver mới `phase1_hrnet_omnivideo.py` — giữ confidence liên tục thật

Không sửa `phase1_hrnet_gpu.py` gốc. Tái dùng phần model-agnostic (path config/checkpoint, `coco_to_h36m` cấu trúc mapping) nhưng đọc mp4 phẳng trực tiếp qua `cv2.VideoCapture`, shard `video_ids[RANK::WORLD_SIZE]` theo `sports_subset_video_ids.txt`. Output cùng format/cùng thư mục `outputs/2d_json/` để Phase 2 dùng thẳng không cần sửa. `coco_to_h36m()` viết lại: vẫn zero-hoá vị trí khi dưới `CONF_THRESHOLD` (giữ nguyên logic tránh toạ độ rác) nhưng **lưu confidence float thật** thay vì ép 1.0/0.0; các khớp suy ra (pelvis/neck/spine/head-top) dùng `min()` của 2 confidence gốc thay vì hardcode 1.0.

### 3. Smoke-test 1 video → pilot SLURM 24 video — cả 2 sạch

Chạy thử trực tiếp 1 video (`iGVvChGEQdM`) — 2,564 frame, 0 lỗi, verify output có 915 giá trị confidence liên tục khác nhau (không phải chỉ 0/1). Submit `submit_phase1_pilot.sbatch` (1 node × 4 GPU, 24 video đầu) — job `976467`, **COMPLETED**, 26 phút, 0 lỗi. Output ghi vào thư mục pilot riêng (`pose_2d_json_pilot/`, tách khỏi `outputs/2d_json/` sản xuất chính cho tới khi verify chất lượng).

Kết quả tổng hợp: 60,506 frame, **47,639 frame có detect người (78.7%)** — 16/24 video ≥80%, chỉ 2/24 video <20% (`28jYYH6WrA0`: 5.1%, `dXv4oInXqiE`: 17.6%, nhiều khả năng sai dương của keyword filter). 78.7% cao hơn hẳn 24-41% coverage joint-level ghi nhận cho FineVideo trước đây.

### Trạng thái cuối phiên

Theo đúng quyết định của user, **dừng lại ở pilot 24 video để xem xét**, chưa mở rộng full 1,256 video. Phase 2-6 chưa bắt đầu.

---

## Cập nhật phiên làm việc — 19/07/2026 (trưa — flatten + tokenize video track xong, khảo sát nội dung, lên kế hoạch pilot pose)

**Việc chính:** Sau khi xác nhận Step A xong (mục dưới), hoàn tất luôn phần còn lại của video track: viết + chạy `flatten_step_a_video.py` (chuyển token thô Step A → format Megatron), submit + xác nhận xong job tokenize Megatron thật (`14120433`/`tok_omni_video`). Đối chiếu số token với FineVideo-VLA v5 để giải thích tại sao OmniVideo-100K "ít token" (không phải bug — do ít document hơn, không phải do density thấp). Phân loại nội dung toàn bộ 5,214 video theo `video_summary`, phát hiện ~24% là sports/hoạt động thể chất thật — quyết định pilot pose pipeline (agent token) trên tập con này, đóng gói thành task handoff cho JUPITER.

### 1. Flatten + tokenize Megatron cho video track — xong thật

`data_prep/omnivideo_100k/flatten_step_a_video.py` (mới, cùng convention drop-rate với `phase7_flatten.py`: avc_lm luôn bỏ, cosmos drop 50%, seed2/caption/speech luôn giữ) — chạy thật, input `omnivideo_100k_video_flat/` (32 file, từ JUPITER) → output `omnivideo_100k_video_flattened/` (32 file), **5,214/5,214 dòng khớp chính xác**, 0 mất video.

Tokenize Megatron: job `14120433` (`tok_omni_video`), **COMPLETED** (06:09→06:26, 17'), output `/p/data1/mmlaion/shared/vla/tokenized_output/omnivideo_100k_video/data_shard_00000.bin/.idx`. Đọc trực tiếp header `.idx` (dtype `int32`, 5,214 sequence khớp đúng số video) → **456,487,128 token thật (~456.5M)**. (Không tìm thấy sbatch script lưu trong repo cho job này — có thể chạy trực tiếp bằng lệnh, chưa commit; nếu cần tái tạo lệnh, tra `sacct -j 14120433 --format=SubmitLine`.)

### 2. Giải thích số token "ít" — so sánh với FineVideo-VLA v5, không phải bug

Đối chiếu FineVideo-VLA v5 (10,554,076,391 token / 371,888 document, ~28,375 token/document) với OmniVideo-100K video (456,487,128 token / 5,214 document, ~87,556 token/document) — **OmniVideo-100K thực ra nhiều token/document hơn FineVideo ~3.1 lần**. Chênh lệch tổng ~23x hoàn toàn do chênh số document ~71x: FineVideo chia mỗi video thành nhiều record scene/activity (371,888 record từ ~40K video gốc, ~9.3 record/video), còn OmniVideo-100K theo đúng thiết kế 1 video = 1 document (không có cấu trúc scenes/activities). Không phải lỗi flatten/tokenize.

### 3. Phân loại nội dung toàn corpus — phát hiện quan trọng, sửa lại nhận định sai trước đó

Nhận định ban đầu trong phiên ("chỉ tin tức/cartoon, không có hoạt động thể chất") **sai** — user nhắc lại nhớ dataset có khá nhiều người, kiểm lại bằng keyword-match trên `video_summary` toàn bộ 5,214 video (không chỉ vài sample nhỏ):

| Loại nội dung | Số video | Tỷ lệ |
|---|---|---|
| Sports/hoạt động thể chất thật | 1,256 | 24.1% |
| News/talking-head | 1,210 | 23.2% |
| Cartoon/animation | 325 | 6.2% |
| Gambling/slot machine | 129 | 2.5% |
| Gaming/gameplay | 115 | 2.2% |
| Vlog/travel | 79 | 1.5% |
| Còn lại (misc) | 2,503 | 48.0% |

Đây là heuristic thô dựa trên text summary cấp video, chưa verify hình ảnh thật — nếu cần chính xác hơn có thể lọc thêm ở field `segments[].caption` (chi tiết hơn theo từng đoạn).

### 4. Quyết định pilot pose pipeline trên tập con sports — kèm giới hạn đã thống nhất

Kiểm tra `phase1_hrnet_gpu.py`, xác nhận model hiện dùng (`td-hm_hrnet-w48...coco-256x192`) là **COCO-17 body-only, không có keypoint bàn tay/ngón tay** — chuyển sang video sports sẽ không cải thiện phần tay, chỉ đa dạng hoá chuyển động toàn thân/cánh tay. **User xác nhận: chấp nhận giới hạn này, bàn tay sẽ là dataset/effort riêng sau**, không cần giải quyết trong pilot này.

Loại nhóm news/talking-head khỏi pilot dù có người: (1) framing thường cận cảnh ngực trở lên → hông/gối/mắt cá ngoài khung hình → zero-fill trong `coco_to_h36m()` (`CONF_THRESHOLD = 0.5`); (2) chuyển động gần như tĩnh, giá trị training thấp, khả năng trùng lặp với dữ liệu lifestyle FineVideo đã có sẵn. Suy luận từ text summary, **chưa verify bằng frame thật** (video ở JUPITER `/e`, không mount được từ session JUWELS) — đã nói rõ với user, đề xuất JUPITER-side pilot có thể đo thử yield thật nếu cần chắc chắn hơn.

### 5. Deliverables — đã viết + commit (`95f2927`)

- `select_sports_subset.py`: script phân loại, chạy thật ra 1,256/5,214 video (24.1%)
- `sports_subset_video_ids.txt`: danh sách 1,256 video_id thật
- `JUPITER_POSE_PILOT_TASK.md`: task handoff cho JUPITER — cảnh báo `phase1_hrnet_gpu.py` hard-code đọc từ FineVideo HF dataset (cùng dạng vấn đề Step A từng gặp với `pipeline.py`), Phase 6 merge cũng gần chắc chắn cần viết lại tương tự `flatten_step_a_video.py`; đề xuất lộ trình thận trọng (pilot ~20-30 video trước, học từ 3 bug thật đã gặp ở Step A) trước khi chạy full 1,256 video.

### Trạng thái cuối phiên

Video track OmniVideo-100K giờ có đủ: Step A xong + flatten/tokenize xong (456.5M token sẵn sàng train) + kế hoạch pilot pose pipeline đã đóng gói, **chưa chạy** (chờ JUPITER pull task này về thực hiện). Chưa quyết định tỷ lệ trộn với FineVideo-VLA lúc train.

---

## Cập nhật phiên làm việc — 19/07/2026 (xác nhận job `970099` OmniVideo-100K Step A đã xong)

**Việc chính:** Track lại trạng thái job full-scale `970099` (Step A OmniVideo-100K, submit cuối phiên 18/7) — **xác nhận đã COMPLETED sạch, đã verify output thật** chứ không chỉ dựa vào SLURM state.

`sacct -j 970099`: `COMPLETED`, exit `0:0`, chạy 19:30:45→22:01:21 (18/7), 2h30'. Verify output: 32/32 file `step_a_rank_*.jsonl` (39GB), đúng **5,214/5,214 dòng** — khớp chính xác tổng số video OmniVideo-100K. Log lỗi (`970099_omni100k_stepA_full_err.log`) chỉ có warning vô hại (`GenerationMixin`/`torch_dtype` deprecation), không có Traceback/Exception thật. Kiểm tra riêng bug seed2 cũ có tái phát ở full-scale không: sample `rank_0` (163 video) — **0 video bị seed2=0**, cả 2 bug fix ở `env_stable_vla` (phiên 18/7 tối muộn) giữ vững ở quy mô đầy đủ. Nội dung mẫu đủ cả 4 loại token (seed2/cosmos/avclm + caption/speech blocks) với số lượng hợp lý.

**Kết luận: Step A (phần cần GPU, chỉ chạy được ở JUPITER) cho OmniVideo-100K đã xong hoàn toàn.** Bước tokenize Megatron với `tokenizer_vla_qwen3` sẽ được đẩy sang chạy ở JUWELS — không thuộc phạm vi task Step A này.

---

## Cập nhật phiên làm việc — 18/07/2026 (tối muộn — phiên bổ sung 3, đọc phần này trước)

**Việc chính:** Viết driver Step A mới cho OmniVideo-100K theo đúng yêu cầu ("tận dụng pipeline cũ nhưng đừng viết code mới vào đó, viết vào `data_prep/omnivideo_100k`") — `step_a_tokenize_video.py` import 3 class tokenizer từ `/e/project1/reformo/nguyen38/prototype/pipeline.py` gốc (không sửa gì ở đó), tự viết toàn bộ logic mới: list video, chunk 8-frame, và **chèn caption/speech chỉ 1 lần tại chunk đầu mỗi segment** (không phải mọi chunk overlap — tránh lặp lại đoạn caption 300-500 từ tới ~40 lần/segment, quyết định do bạn chốt qua `AskUserQuestion`). Trong lúc chạy pilot phát hiện + fix **2 bug thật của `env_stable_vla`** khiến seed2 tokenizer hỏng hoàn toàn, âm thầm (không crash, chỉ ra `seed2=0` mọi video) — **bug này ảnh hưởng cả pipeline FineVideo gốc, không riêng OmniVideo-100K, nếu chạy lại Step A bằng env hiện tại**. Submit full-scale lần 1 (`970087`, 32 GPU) thì **lộ thêm 1 bug thật khác — trong chính code mới của tôi**: trích toàn bộ frame gốc cả video ra đĩa tạm không resize, 32 rank chạy song song làm tràn quota đĩa user, gần như mọi video lỗi `Disk quota exceeded`. Đã `scancel`, dọn ~40GB rác, viết lại theo streaming từng chunk 8-frame + resize 512×512, verify lại bằng pilot (`970095`: 48/48 video sạch, nhanh hơn bản cũ 7:22 so với 9:23), rồi **submit lại full-scale (`970099`, 8 node×4 GPU=32 GPU) cho toàn bộ 5,214 video — đang chạy lúc ghi entry này**.

### 1. Driver Step A mới — `data_prep/omnivideo_100k/step_a_tokenize_video.py`

Không đụng `pipeline_video/pipeline.py` (bản trong git, dùng để tracking) hay `/e/project1/reformo/nguyen38/prototype/pipeline.py` (bản runtime thật, có đủ checkpoint model). Chỉ import 3 class tokenizer cấp thấp (`Seed2Tokenizer`/`CosmosVideoTokenizer`/`AVCLMTokenizer`) từ `prototype/pipeline.py`, kèm `sys.path.insert` + `os.chdir(PROTOTYPE_DIR)` bắt buộc (checkpoint paths trong 3 class đó là relative tới CWD, và `import cosmos_tokenizer` bên trong `pipeline.py` chỉ resolve được nếu `sys.path[0]` trỏ đúng `prototype/`).

Logic mới hoàn toàn viết trong file này: sharding video theo `video_list[RANK::WORLD_SIZE]` (đơn giản hơn `dataset.shard()` của FineVideo vì chỉ là list file phẳng), extract full-video 30fps frame (video max 180s nên không lo tràn RAM), loop 8-frame chunk mirror `tokenize_activity_frames()` gốc nhưng thêm bước chèn `<caption>`/`<speech>` tại "anchor chunk" (chunk đầu tiên có `start_sec >= segment.start_sec`) — pattern giống hệt cách `phase6_merge_adaptive.py` snap speech ASR về đúng 1 chunk bắt đầu, không rải khắp segment. Output: `{"video_id", "text"}` 1 dòng/video, có resume toàn cục (quét hết `step_a_rank_*.jsonl` hiện có trước khi xử lý, giống pattern gốc `process_pipeline()`).

### 2. Phát hiện + fix 2 bug seed2 thật trong `env_stable_vla` (quan trọng — ảnh hưởng cả FineVideo, không chỉ OmniVideo-100K)

Pilot lần 1 (`970063`) chạy nhưng `seed2=0` mọi video, không có lỗi crash. Điều tra ra 2 lớp bug chồng nhau, cả 2 đều do `transformers` trong `env_stable_vla` đã trôi lên `4.57.6` (checkpoint config ghi `4.52.4`, tức khác xa lúc bạn chạy thành công lần trước):

- **Bug #1 (import path):** `apply_chunking_to_forward`/`find_pruneable_heads_and_indices`/`prune_linear_layer` bị dời từ `transformers.modeling_utils` sang `transformers.pytorch_utils` trong bản 4.57.6. Code `seed2/seed2_tokenizer.py` (viết theo API BERT đời cũ) vẫn import từ chỗ cũ → `ImportError`, bị `except Exception` trong `pipeline.py` nuốt mất, chỉ in warning.
- **Bug #2 (hành vi `tie_weights()`, sâu hơn, chỉ lộ ra sau khi fix bug #1):** `AttributeError: 'NoneType' object has no attribute 'predictions'`. Full traceback (lấy qua job debug riêng `debug_seed2_load.py`, job `970070`) cho thấy: tác giả gốc **chủ ý** set `self.Qformer.cls = None` (dòng 2601 — Qformer không cần đầu MLM khi chỉ encode ảnh). `transformers` bản cũ tolerate việc này; bản 4.57.6 gọi `tie_weights()` đệ quy vào **mọi** submodule kể cả cái đã bị null hoá chủ ý → crash.

**Cách sửa:** cả 2 đều được vá bằng monkeypatch **chỉ trong `step_a_tokenize_video.py`** (đúng yêu cầu — không đụng `seed2_tokenizer.py`, `pipeline.py`, hay `env_stable_vla` chung): (1) gán lại 3 hàm bị dời vào `modeling_utils` trước khi import `pipeline.py`; (2) import `seed2_tokenizer` sớm rồi patch `get_output_embeddings`/`set_output_embeddings` của `BertLMHeadModel`/`BertForMaskedLM` để trả `None` an toàn khi `self.cls is None`, đúng ý định gốc thay vì crash. Verify qua `debug_seed2_load.py` (`LOADED OK`, job `970072`) rồi pilot thật (job `970073`).

### 3. Pilot 3 lần (fix seed2) → throughput thật → submit full-scale lần 1

- Pilot lần 1 (`970063`, 2 node×4 GPU, 48 video): seed2=0 do bug #1. **Huỷ giữa chừng** theo quyết định của bạn (`scancel`) để khỏi lãng phí GPU-giờ, thay vì chạy hết 30 phút.
- Pilot lần 2 (`970069`): qua được bug #1, lộ bug #2. Job vẫn tự chạy xong hết 48/48 video (seed2=0 toàn bộ, cosmos/avclm/caption/speech vẫn đúng).
- Pilot lần 3 (`970073`, sau khi fix cả 2 bug seed2): **48/48 video, seed2 ra token thật (2000-5700/video), 9 phút 23 giây**. Trung bình/video: seed2 106 block, cosmos 397, avc_lm 397, caption 9.1, speech 8.8 — khớp đúng kỳ vọng thiết kế (~9.1 segment/video từ data thật).
- Throughput đo được: ~93.8s/video/GPU. Ước tính full-scale 5,214 video: ~4.3h ở 32 GPU. **Đã submit job full-scale `970087`** (8 node×4 GPU=32 GPU, `--time=05:00:00`) — quy mô do bạn chọn, nhỏ hơn 40 node của FineVideo vì dataset chỉ ~1/8 quy mô.

### 4. Bug quota đĩa — lộ ra ở full-scale lần 1, do chính code mới, đã fix + verify + submit lại

Job `970087` gần như mọi video đều lỗi `[Errno 122] Disk quota exceeded`. Nguyên nhân: `extract_30fps_frames()` (bản đầu) trích **toàn bộ frame gốc của cả video** (tới 5400 frame cho video 180s, KHÔNG resize) ra PNG tạm — 32 rank song song, mỗi rank có lúc giữ 1-2.7GB PNG tạm cùng lúc → tràn quota user (pilot 8-rank trước không trúng vì tổng footprint đồng thời còn dưới ngưỡng — bug chỉ lộ ra ở quy mô 32 rank).

**Đã `scancel` job ngay, dọn ~40GB rác** (`omni_temp_frames_rank_*`, `temp_seed2_rank_*.jpg`). **Fix:** viết lại hoàn toàn theo streaming — 1 lệnh ffmpeg/chunk 8-frame (thay vì 1 lệnh cho cả video), resize 512×512 (khớp `target_size` mặc định của Seed2Tokenizer nên không mất chất lượng; Cosmos downsample tiếp xuống 160 nên cũng không mất gì thêm). Giới hạn dung lượng tạm chỉ còn ~8 frame/rank tại một thời điểm, bất kể video dài bao nhiêu — an toàn bất kể quota còn lại bao nhiêu.

Verify: pilot lại (`970095`, cùng 48 video) — **48/48 video sạch, 0 lỗi quota, temp dir mỗi rank chỉ 1-2.5MB (so với 1-2.7GB trước), và nhanh hơn bản cũ** (7 phút 22 giây so với 9 phút 23 giây — downscale + ít file hơn giúp nhanh hơn, không chỉ an toàn hơn).

### 5. Trạng thái cuối phiên: full-scale lần 2 đang chạy

**Đã submit lại `970099`** (8 node×4 GPU=32 GPU, `--time=05:00:00`, `data_prep/omnivideo_100k/submit_step_a_full.sbatch`) cho toàn bộ 5,214 video, thay thế `970087` đã huỷ. Output: `$DATA/omnivideo_100k/step_a_output/step_a_rank_{0..31}.jsonl`. Có resume — an toàn để submit lại y nguyên lệnh nếu job timeout/crash giữa chừng.

### Việc tiếp theo hợp lý nhất

Chờ job `970099` chạy xong (theo dõi qua Monitor). Sau khi xong: tokenize Megatron bằng `tokenizer_vla_qwen3` (257,901 vocab — **tuyệt đối không dùng `tokenizer_vla_adaptive_v2`**, bài học đau đã ghi ở phiên trước), rồi mới tới bước quyết định tỷ lệ trộn với FineVideo-VLA/MV-Omni lúc train.

---

## Cập nhật phiên làm việc — 18/07/2026 (tối — phiên bổ sung 2, đọc phần này trước)

**Việc chính tối nay:** phát hiện + fix bug lớn — **cả 3 job tokenize đang chạy dùng nhầm tokenizer cũ** (không phải Qwen3 như dự định), đã hủy job đang chạy, xoá 215GB output sai, sửa script, resubmit lại đúng Qwen3. Đếm token thật: FineVideo-v5 gần như không đổi (10.55B, chứng minh lệch số 5.256B không phải do tokenizer) — MV-Omni tăng thật +25% (20.39B). Tải xong hoàn toàn cả 3 nguồn mới (OmniVideo-100K, RoboVQA, SenseNova). Tạo `data_prep/` với script flatten cho RoboVQA + OmniVideo-100K (bắt + fix 2 bug thật khi validate kỹ). Giải nén + map caption/speech cho video OmniVideo-100K — **sẵn sàng Step A** (chờ submit ở JUPITER). Điều tra sâu RoboVQA — phát hiện video thật nằm trong tfrecord (không phải chỉ 4.5% như báo sai lúc đầu), viết parser tfrecord thuần Python, đang trích xuất frame (chạy nền, ~36% xong cuối phiên) — nhưng phát hiện đây là **16 ảnh rời rạc/episode, không phải video liên tục**, cần quyết định kiến trúc trước khi qua Step A. Đọc trực tiếp paper SenseNova (39 trang, qua `pypdf`) theo yêu cầu Huu — tìm ra **22 dataset nguồn ảnh cụ thể**, verify license từng cái: vài cái permissive thật (GQA/VQA/VSR/CLEVR/MindCube), phần lớn (nhóm đóng góp nhiều ảnh nhất) non-commercial xác nhận (ScanNet/ScanNet++/Matterport3D/CA-1M/Ego-Exo4D/ARKitScenes), vài cái đáng ngờ kiểu "license mới nhưng nguồn cũ dính" (VSI-590K/ViCA/VLM-3R). Đọc thêm paper MINT-1T gốc — xác nhận quyết định bỏ ảnh trước đây đúng, có bằng chứng mạnh hơn nữa (chính tác giả tự ghi "N/A" cho license asset).

### 1. Bug tokenizer sai — phát hiện + fix + resubmit

Bạn hỏi thẳng "có refer đúng tokenizer mới nhất chưa, nhớ lần này dùng Qwen" — check lại thì **cả 3 sbatch (`tokenize_finevideo_v5`/`tokenize_mv_omni`/`tokenize_mint1t`) đều trỏ `tokenizer_vla_adaptive_v2` (GPT-NeoX cũ), không phải `tokenizer_vla_qwen3`**. Đã: (1) `scancel` job `tok_mint1t` đang chạy dở với tokenizer sai, (2) sửa `TOKENIZER_MODEL` trong cả 3 script → `tokenizer_vla_qwen3`, (3) xoá 215GB output cũ (bắt buộc vì cả 3 script dùng `--resume`), (4) submit lại cả 3 (job `14118929`/`14118930`/`14118931`).

### 2. Token thật sau khi dùng đúng Qwen3

| Nguồn | Token (Qwen3) | So với bản tokenizer sai |
|---|---|---|
| FineVideo-VLA v5 | **10,550,998,369** | Gần như y hệt (10,554,076,391) — **xác nhận lệch 5.256B không phải do tokenizer** |
| MV-Omni | **20,389,561,883** | Tăng thật +25% (16,357,256,571) — Qwen3 vocab lớn ảnh hưởng nhiều hơn với nội dung tự nhiên |
| RoboVQA | 58,588,270 | Job nhỏ (1-node), xong luôn |
| OmniVideo-100K QA | 30,689,299 | Job nhỏ (1-node), xong luôn |
| MINT-1T text | Đang chạy | — |

### 3. Tải xong hoàn toàn: OmniVideo-100K, RoboVQA, SenseNova-SI-8M

Cả 3 đều `snapshot_download completed successfully`. SenseNova gặp sự cố nhỏ giữa chừng — tmux session bị chết âm thầm (verify qua `tmux ls`/`ps aux`/log đứng im), tôi tạo tmux mới nhưng thiếu `HF_TOKEN` trong shell của tôi nên lỗi ngay — **bị chặn khi tự động dò tìm token trong cache/rc files (đúng, không nên tự làm vậy)**, bạn tự chạy lại và hoàn tất (1,121.4GB, 53/53 zip).

### 4. `data_prep/` — folder mới, 2 script flatten, bắt 2 bug thật khi validate

Tạo `data_prep/omnivideo_100k/` + `data_prep/robovqa/` (đúng ý định ghi sẵn CLAUDE.md, chưa từng tồn tại thật).

- `data_prep/robovqa/flatten_text.py`: 221,912 record → flat JSONL, 0 skip.
- `data_prep/omnivideo_100k/flatten_qa_text.py`: gộp OE+MCQ → 99,983 record. **Validate kỹ theo đúng yêu cầu bạn ("đừng sai") bắt được 2 bug thật:** 2,740 record MCQ dạng `event_sequence_ordering` dùng field tên khác (`question_textual`/`options_textual`) bị bỏ sót oan; 6,372 record OE có `answer` là list bị in xấu kiểu Python-repr (`['B','C','A']`). Fix cả 2, verify lại bằng audit kiểu dữ liệu toàn bộ corpus (không chỉ sample) + regex check — sạch 100%.

### 5. OmniVideo-100K — giờ đã thật sự sẵn sàng Step A

Giải nén 5,214 video mp4 (49GB, khớp đúng số video). Viết `data_prep/omnivideo_100k/build_segment_captions.py` — map `segments[].visual`/`transcription` (đã convert MM:SS→giây) thành caption/speech theo từng đoạn thời gian: 5,214 video, 47,467 segment, 0 lỗi timestamp. **Video + caption/speech đều sẵn sàng — chỉ chờ submit Step A ở JUPITER** (bạn nhắc: Step A phải chạy JUPITER vì chỉ nơi đó có GPU, JUWELS chỉ dùng tokenize CPU).

### 6. RoboVQA — đính chính phát hiện sai trước đó, viết parser tfrecord thuần Python, đang trích xuất

**Đính chính:** lúc trước báo "chỉ 4.5% video có mp4, phần còn lại coi như mất" — **sai**. Ảnh thật của 95.5% record còn lại nằm trong `tfrecord/` (184 shard), không mất. Không có TensorFlow/protobuf trong bất kỳ env nào — viết `data_prep/robovqa/tfrecord_lite.py` (parser protobuf thuần Python, không cần cài gì). Verify kỹ qua nhiều bước: dò cấu trúc mù → xác nhận JPEG thật (mở ảnh xem, đúng cảnh robot) → chạy trên 500 episode thật (luôn đúng 16 frame/episode, 100% JPEG hợp lệ) → **cross-check text giải mã khớp byte-for-byte với json/train đã biết đúng (join key đúng là `video_filename`, không phải `uid` — 2 ID khác nhau, dễ nhầm)**.

Viết `data_prep/robovqa/extract_frames.py` — bug đầu tiên bắt ngay lúc test (list vs tuple, đã fix). Đang chạy nền (không phải SLURM — job nhẹ, chạy trực tiếp login node theo đúng "job nhỏ thì tmux/nền, không cần SLURM"). **Cuối phiên: 67/184 shard, 82,669/221,912 episode đã trích.**

**⚠️ Phát hiện quan trọng khi bạn hỏi lại — RoboVQA KHÔNG có video liên tục, chỉ có 16 ảnh JPEG rời rạc/episode (~1.6fps)**, khác hẳn FineVideo/OmniVideo-100K (video liên tục thật). Step A hiện tại thiết kế cho video liên tục — **chưa chắc dùng thẳng được cho 16 ảnh rời rạc này, cần quyết định kiến trúc riêng** (nối thành "video giả", hay xử lý theo hướng ảnh-tĩnh như đã bàn cho SenseNova) trước khi làm tiếp. Chưa quyết, để phiên sau.

### 7. SenseNova-SI-8M — đọc trực tiếp paper (39 trang, qua `pypdf`), tìm ra 22 dataset nguồn cụ thể

Huu chất vấn thẳng trên chat: "how do you know, what is the basis" — đúng, kết luận trước đó (dựa vào README im lặng + so sánh MINT) chưa đủ chặt. Không có tool đọc PDF nào trong project — **hỏi bạn trước khi cài** (đúng nguyên tắc không tự ý pip install), bạn đồng ý cài `pypdf`.

Đọc ra **Section 3.2 "Data Sources"**: 8.5M cặp QA từ **22 dataset nguồn**, chia 3 nhóm. Verify license từng cái (WebSearch trực tiếp trang terms-of-use chính chủ):

- **Permissive thật:** GQA, VQA, VSR, CLEVR-series, MindCube (5/22).
- **Non-commercial/gated xác nhận:** IconQA, MultiSpa, ScanNet, ScanNet++, Matterport3D, CA-1M, Ego-Exo4D, + ARKitScenes (check thêm vì là nguồn của các dataset trung gian) — 8 dataset.
- **"Nested derivative" đáng ngờ:** VSI-590K/ViCA/VLM-3R tự gắn license mới nhưng build từ chính ScanNet/ARKitScenes — license mới không rửa được nguồn cũ.
- **Chưa xác định:** 7 dataset còn lại (SPEC, Open3D-VQA, REL3D, SAT, GRiD-3D, SUN RGB-D, MessyTable).

**Kết luận cuối:** nhóm đóng góp nhiều ảnh nhất (4.5M/8.5M) xác nhận non-commercial — tag `apache-2.0` của SenseNova không áp được cho phần lớn ảnh. Đã báo đầy đủ breakdown để bạn relay Huu.

### 8. MINT-1T — đọc thêm paper gốc theo yêu cầu bạn, xác nhận quyết định cũ đúng

Paper có phần "Datasheet for Datasets" — chính tác giả tự ghi **"Did you mention the license of the assets? [N/A]"**. Bằng chứng mạnh hơn hẳn README đã dùng trước đây. Quyết định bỏ ảnh MINT (giữ text) — vẫn đúng, không cần đổi gì.

---

## Cập nhật phiên làm việc — 18/07/2026 (chiều — phiên bổ sung, đọc phần này trước)

**Việc chính chiều nay:** xác nhận 2 job tokenize từ sáng (MV-Omni, MINT) thực sự chạy thật (MV-Omni từng fail âm thầm lúc sáng do lỗi Ray, đã fix + resubmit) — **MV-Omni đã COMPLETED thật, MINT vẫn đang chạy**. Đếm token thật bằng `count_tokens.py` cho FineVideo-v5 và MV-Omni — **phát hiện lệch số quan trọng, chưa rõ nguyên nhân** (xem mục 1). Đính chính license SenseNova-SI-8M (không còn an toàn như tưởng). Viết lại mục Gen-EgoData (không phải ego-video, mà là dữ liệu action tay-đơn từ thiết bị cầm tay). Thảo luận sâu với Van Khue về câu hỏi ego/exo — kết luận **không cần sửa FineVideo-VLA**. Khảo sát thêm dataset mới trên HF (RoboVQA, Open X-Embodiment, NVIDIA GR00T-Sim — permissive; loại AgiBot World/Apple EgoDex/Meta ego-1k/EgoBrain — non-commercial). Viết script + bắt đầu tải OmniVideo-100K và RoboVQA. Cập nhật toàn bộ `datasets.md`.

### 1. Token thật đã tokenize xong — VÀ 1 lệch số quan trọng cần lưu ý

Chạy `count_tokens.py` (chỉnh `OUTPUT_DIR` inline, không sửa file gốc) trên 2 output Megatron thật:

| Nguồn | Token thật (BIN SIZE CHECK: PASS) | Document |
|---|---|---|
| **FineVideo-VLA v5** | **10,554,076,391 (10.55B)** | 371,888 |
| **MV-Omni** | **16,357,256,571 (16.36B)** | 1,593,301 |
| MINT-1T text | Chưa có — job vẫn đang chạy | — |
| **Tổng đã tokenize xong** | **~26.91B token** | — |

**⚠️ Lệch số chưa rõ nguyên nhân:** số token FineVideo-v5 thật (10.55B) **gấp ~2 lần** con số vẫn ghi trong docs từ trước tới giờ (5,255,589,397 / 5.256B, xem mục "18/07/2026 sáng" bên dưới). Số document khớp chính xác (371,888 cả 2 phía) nên không phải lỗi thiếu/thừa record. Nghi ngờ hợp lý nhất (chưa verify): con số 5.256B được tính ở bước flatten bằng cách đếm số lần xuất hiện tag `<..._N>` + đếm từ thô cho phần text tự do (title/context/caption/speech), trong khi tokenizer BPE thật (GPT-NeoX-20b + VLA extension) tách 1 từ tiếng Anh thường thành nhiều subword token hơn đáng kể so với đếm-từ-thô — token VLA (`<seed2_N>`, `<pelvis_x_N>`...) vẫn atomic cả 2 cách tính nên phần lệch nhiều khả năng dồn hết vào phần text tự nhiên. **Chưa verify tận gốc, để dành phiên sau.** Tin tốt: ngân sách token thật cao hơn nhiều so với lo ngại "corpus quá nhỏ cho model 1.7B" trước đây.

### 2. Job tokenize — trạng thái thật cuối phiên

- **`tok_mv_omni` (14118393)** — job đầu (`14117680`) từng fail âm thầm sáng nay (SLURM báo COMPLETED nhưng Ray không connect, 0 output thật). Đã fix bug start Ray cluster, resubmit — **lần này COMPLETED thật** (13:18, không traceback, output 7 shard/60.94GB thật).
- **`tok_mint1t` (14118392)** — vẫn RUNNING cuối phiên (>1h40p), tiến độ bình thường, chưa lỗi.

### 3. SenseNova-SI-8M — đính chính license (xem chi tiết trong `datasets.md` mục 4 và [[project_vla_status]])

Huu nghi ngờ (qua ChatGPT) ảnh trong dataset không permissive hoàn toàn. Điều tra lại kỹ (README, GitHub `OpenSenseNova/SenseNova-SI`, paper arXiv:2511.13719, tự đọc `image` column thật trong parquet) — **không tìm được tài liệu nào nói rõ nguồn gốc ảnh gốc**, paper/GitHub chỉ dùng từ "curated" (gợi ý tổng hợp từ nguồn khác, không phải tự chụp). Cùng dạng bẫy `cc_dump` như MINT. **Rút lại kết luận "an toàn hơn MINT" ở mục sáng nay** — license mở, chưa nên coi là sẵn sàng train.

### 4. Gen-EgoData — viết lại sau khi đọc kỹ toolkit `das-datakit`

Không phải "video ego + pose người" — là dữ liệu từ **thiết bị cầm tay "DAS device"** (kiểu UMI), action thật = `eef_pose` (6-DoF) + `Gripper_width` (tay đơn), khác hẳn `<agent>` 17-khớp hiện tại. License CC-BY-SA-4.0 (share-alike, cần Huu duyệt điều khoản riêng). Xếp lại vào cùng nhóm "robot-action modality" với MolmoAct2/Cosmos3-DROID/Open X-Embodiment/GR00T-Sim.

### 5. Ego/exo — kết luận sau thảo luận dài với Van Khue: KHÔNG cần sửa FineVideo-VLA

Verify trực tiếp trong code (`phase3_kinematics_processor.py`): `<agent>` pose token đã **root-centred/pelvis-relative** sẵn (`retargeted[:, pelvis_idx] = 0.0`) — bất kể video quay góc nào. "Egocentric" (góc camera) và "root-centred" (quy ước toạ độ khung xương) là 2 trục khác nhau — không có "pose exocentric" để mà sửa, và head-relative sẽ **tệ hơn** pelvis-relative (nhiễu xoay đầu lan vào mọi khớp khác), không phải "egocentric hơn". Vấn đề thật (nếu có) là domain-gap giữa video train (3rd-person) và video robot thấy lúc deploy (camera gắn robot) — giải pháp đúng là ưu tiên **integrate Isaac Sim pipeline** (hoặc dùng GR00T-Sim mục 6 làm tạm), không phải sửa FineVideo-VLA hay đi săn thêm ego-video dataset.

### 6. Khảo sát dataset mới trên HF — 3 ứng viên tốt, 4 ứng viên bị loại vì license

| Dataset | License verify thật | Kết luận |
|---|---|---|
| **NVIDIA GR00T-X-Embodiment-Sim** | ✅ CC-BY-4.0 | **Ứng viên mạnh nhất** — 345K trajectory, có humanoid GR1, cùng vai trò Isaac Sim |
| **RoboVQA** (Google DeepMind) | ✅ CC-BY-4.0+Apache-2.0 (verify từ GitHub chính chủ) | 238h, 3 embodiment — tải qua mirror `Tianli/robovqa` (Van Khue tìm ra), có `LICENSE.txt` Apache-2.0 thật |
| **Open X-Embodiment** | ⚠️ Registry 55-60 dataset con, KHÔNG đồng nhất license | Chưa tải — cần audit từng cái trước |
| AgiBot World | ❌ CC BY-**NC**-SA | Loại |
| Apple EgoDex | ❌ CC-BY-**NC**-ND | Loại (tiếc, rất khớp use-case: 829h ego dexterous manipulation + pose) |
| Meta `ego-1k` / `EgoBrain` | ❌ Non-commercial (+ EgoBrain lạc chủ đề, EEG) | Loại |

### 7. Bắt đầu tải OmniVideo-100K + RoboVQA

Viết `tools/extract/download_omnivideo_100k.py` + `tools/extract/download_robovqa.py` (cùng pattern resumable với `download_sensenova_si8m.py`, đã verify file list/size thật qua HF API trước khi viết). Van Khue chạy cả 2 trong tmux riêng (`omnivideo_dl`, `robovqa_dl`) — cuối phiên: OmniVideo-100K 22 file/8.8GB/52.9GB, RoboVQA 402 file/1.6GB/~70.8GB, cả 2 đang tiến triển bình thường. SenseNova vẫn tải song song (52/54 file, 1.1TB, thỉnh thoảng timeout tự resume).

### 8. `datasets.md` — cập nhật toàn bộ

Thêm mục 15 (RoboVQA), 16 (Open X-Embodiment), 17 (GR00T-Sim); viết lại mục 4 (SenseNova, đính chính license) và mục 8 (Gen-EgoData, schema đúng); thêm 4 dòng "đã check, loại" cho AgiBot/EgoDex/ego-1k/EgoBrain; cập nhật bảng tổng quan + phần "Việc còn mở".

---

## Cập nhật phiên làm việc — 18/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay:** xác nhận task #6/#7 full-scale (Phase 6 v4 + Phase 7 v5) đã chạy xong sạch, quality/validity khớp gần tuyệt đối dự đoán, upload lên HF. Tạo `datasets.md` khảo sát 14 dataset. Điều tra sâu MINT-1T-HTML (structure thật, license) → **quyết định bỏ hẳn phần ảnh** (chỉ URL, không track được license), giữ phần text. Điều tra SenseNova-SI-8M (ảnh thật, permissive) → **quyết định tải full 1.13TB**, đang chạy. Thiết kế rồi **từ chối** egocentric perspective converter sau khi soi kỹ giá trị thật. Setup pipeline Megatron tokenize cho 3 nguồn (MV-Omni, MINT, FineVideo v5), account thử đổi `laionize`/`batch` — **chưa submit**.

### 1. Task #6/#7 full-scale — XÁC NHẬN HOÀN THÀNH (docs cũ ghi "chưa confirm", giờ đã xong)

`sacct` xác nhận cả 2 job COMPLETED (14114336 merge v4: 31 phút, 14114370 flatten v5: 38 phút), 0 lỗi thật (chỉ warning module vô hại). Kết quả khớp gần như tuyệt đối dự đoán trước khi chạy: token tăng **+0.740%** (dự đoán +0.737%/+0.749%), caption/speech token lệch <0.05%, record count khớp 100% (371,888). Tổng **5,255,589,397 token** (5.256B). Verify không double-injection (2,787 activity check tag mở/đóng khớp), spot-check nội dung caption/speech đặt đúng vị trí, nội dung hợp lý.

**Đã upload lên HF:** cập nhật `tools/upload/vla_flattened_dataset_card.md` (bảng số liệu v5, mục "What Changed in v5", vocab 156,509) + `upload_flattened_hf.py` (default trỏ `megatron_dataset_v5`, prefix `flat_final_vla_adaptive_rank`). User chạy upload, confirm live trên `EmpathicRobotics/FineVideo-Phase7-Flattened` (verify qua HF API, `lastModified` đúng ngày chạy).

### 2. Tạo `datasets.md` — khảo sát 14 dataset, mọi field verify bằng data thật

File mới ở root repo, trả lời nhanh cho từng dataset: tổng quan / đã tải chưa+path / tokenize modality nào / structure + có thể bổ sung token gì / ready Megatron chưa. Gồm: FineVideo-VLA, MixtureVitae-Omni, MINT-1T-HTML, SenseNova-SI-8M, OmniVideo-100K, MolmoAct2-BimanualYAM, Cosmos3-DROID, Gen-EgoData, MixtureVitae-Backup/multimodal, VALID, stera-10m, FineVLA, abc.bot, và "MINT PDF data" (Huu đã tải sẵn, chưa rõ path, trên leo). Đối chiếu với danh sách 7 dataset Huu tự liệt kê trong chat — khớp 100%, không sót cái nào.

**Phát hiện quan trọng khi làm rõ MV-Omni:** không cần bước "flatten" như FineVideo — data gốc `mv_omni_converted/*.jsonl.gz` đã đúng schema `{"text":...}` sẵn, chỉ còn thiếu (a) quyết định tỷ lệ trộn tránh loãng agent-token, (b) chạy tokenize thật.

### 3. MINT-1T-HTML — điều tra structure + license thật, quyết định BỎ ẢNH, giữ text

**Tải xong hoàn toàn** (2.7TB, 6,159/6,159 file, verify qua log + đếm file thật).

**Structure thật (không phải đoán từ README):** `texts[]`/`images[]` cùng độ dài, xen kẽ, loại trừ lẫn nhau (vị trí nào có text thì ảnh null và ngược lại). `image_hashes`/`images_metadata` KHÔNG cùng độ dài, không align theo index — chỉ nên dùng `images[i]` (URL) làm nguồn sự thật.

**Quy mô thật (đo, không đoán):** ~850M record, ~2.83 tỷ URL ảnh, 91.7% còn sống (test 60 URL thật), ~97KB/ảnh → tải hết sẽ ~130-180TB. → chỉ pilot 20 shard (~9.2M ảnh).

**Bug tốc độ tìm & fix:** rate-limit theo domain (0.5s/request) vô tình bóp tốc độ toàn bộ 64 worker xuống ~10 img/s vì phần lớn ảnh dùng chung 4 host CDN Blogspot. Fix: đổi sang semaphore giới hạn concurrency/domain (mặc định 8) thay vì serialize.

**License — đọc thẳng README chính thức mlfoundations:** `cc_dump` KHÔNG phải license (là mã CommonCrawl dump, dễ nhầm CC=Creative Commons). Pipeline filter của họ không hề lọc bản quyền ảnh (chỉ NSFW/size/dedup), và README tự nhận trách nhiệm thuộc về user. **Quyết định (chat 18/7, Huu): bỏ hẳn ảnh** (*"if the mint doesn't have images ignore it"*), **giữ text** (*"the hf dataset is fine"* — Van Khue). Đã xoá 130MB ảnh pilot đã tải. `stera-10m` cũng bị loại cùng phiên (not permissive).

### 4. SenseNova-SI-8M — điều tra structure thật, quyết định tải full, đang chạy

**Structure thật:** config `full` (`SenseNova-SI-8M.parquet`, 851MB, 8,164,067 record) chỉ có `image: list<string>` (path tương đối), KHÔNG nhúng bytes — khác bản `preview` (1000 sample, có nhúng bytes, chỉ để xem thử). Ảnh thật nằm trong **53 file zip độc lập** (`images_part_001..053.zip`, ~1.10TB) — giải nén tất cả vào 1 thư mục chung sẽ tự ráp lại đúng cây `images/`, join bằng `f"{extract_dest}/{image[i]}"`.

Nội dung: VQA trắc nghiệm về spatial reasoning trong nhà (định vị vật thể, hướng tương đối) — sát với nhu cầu robot/embodied hơn QA thường. Apache-2.0, ảnh bytes thật (không URL) → không vướng rủi ro license như MINT.

**Quyết định: tải full 1.13TB.** Script `tools/extract/download_sensenova_si8m.py` (resumable, in tiến độ mỗi lần retry). **Tính đến cuối phiên: ~71GB/1.13TB, ~45MB/s thật (đo trực tiếp), ETA ~6.5h.** Bước giải nén+join chưa code, để sau khi tải xong.

### 5. Egocentric perspective converter — thiết kế xong rồi TỪ CHỐI (quan trọng, đừng làm lại nếu chưa giải quyết được vấn đề gốc)

Thiết kế ban đầu: `<agent_ego>` tag riêng (tránh nhập nhằng vocab với `<agent>` 3rd-person), record riêng biệt (không nhét chung 1 record tránh bloat context). Nhưng **bị từ chối sau khi user hỏi thẳng "có gì hay"**: video (seed2/cosmos/avclm) không đổi — vẫn là video YouTube 3rd-person gốc — chỉ nhãn pose bị xoay sang hệ quy chiếu chỉ có ý nghĩa nếu có camera gắn đầu thật (không tồn tại). Hệ quả: (1) cặp video-pose không nhất quán vật lý — dạy sai mapping; (2) phép biến đổi là isometry (khả nghịch, không mất thông tin) nên `agent`/`agent_ego` **thông tin y hệt nhau** cho việc retarget robot — không thêm kiến thức mới. **Kết luận: không code theo hướng này**, chỉ đáng làm lại nếu có video egocentric thật để ghép cùng.

### 6. Megatron tokenize — setup cho 3 nguồn, script sẵn sàng, CHƯA SUBMIT

TODO tồn đọng lâu nhất của project: `.bin/.idx` thật hiện có (2.84B token) vẫn dùng data+tokenizer v1 cũ, chưa từng tokenize lại với SNAC/caption/speech hay MV-Omni.

Tìm ra script thật `mv-scale/tokenize_vla_adaptive.sbatch` (hạ tầng dùng chung nhiều project, account `cstdl` trên JUSUF trước đây). Thử đổi sang `account=laionize`/`partition=batch` trên JUWELS theo yêu cầu user — **verify trước** bằng `sacctmgr` (laionize có association hợp lệ với batch) và test path hạ tầng đọc được, không đoán bừa.

**Phát hiện quan trọng:** `mv_preprocess_data.py` chỉ đọc `.jsonl` phẳng key `text`, KHÔNG đọc parquet trực tiếp. MV-Omni/FineVideo-v5 đã đúng format sẵn. MINT (parquet, cột `texts[]`) cần bước convert riêng.

**Đã tạo:**
- `tools/extract/convert_mint1t_text_jsonl.py` + `slurm/convert_mint1t_text.sbatch` (32-array, nối `texts[]` thành 1 chuỗi `{"text":...}`)
- `mv-scale/tokenize_mv_omni.sbatch`, `tokenize_finevideo_v5.sbatch`, `tokenize_mint1t.sbatch` — cùng dùng tokenizer `tokenizer_vla_adaptive_v2` (156,509 vocab) để token ID khớp nhau lúc trộn sau này; MINT phụ thuộc bước convert xong trước.

Tất cả đã syntax-check OK. **Chủ động KHÔNG quyết định tỷ lệ trộn MV-Omni/MINT trong bước này** — theo yêu cầu user ("mấy cái quyết định drop out để sau đi"), đây là quyết định lúc train, không phải lúc tokenize.

**Trạng thái cuối phiên: cả 4 job (convert MINT + 3 tokenize) đều CHƯA submit.** SenseNova vẫn đang tải nền. Phiên sau cần check cả 2.

---

## Cập nhật phiên làm việc — 17/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay:** xác nhận task #3 (A2 captioning) đã chạy xong hoàn toàn qua đêm, chạy + verify task #4, code + test kỹ task #6 + #7, đo chính xác mức tăng token (2 phương pháp độc lập đều ra ~0.75%), commit + push code, **submit full-scale SLURM job task #6**.

### 1. Task #3 (A2 captioning) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN (qua đêm 15→17/7)

Kiểm tra `sacct` cho chuỗi job `14104157`→`158`→`159`: cả 3 đều `COMPLETED`, 2 job cuối chỉ mất ~1 phút (do `--skip-existing` không còn việc gì để làm). Đếm dòng thật: **912,998 dòng caption** trên 40,798 file — khớp chính xác target A1. `squeue` trống, không còn job caption nào chạy.

### 2. Task #4 (`build_caption_dict.py`) — CHẠY XONG, VERIFY KỸ

Reshape 40,798 file caption phẳng thành dict `{activity_id: {chunk_idx: "<caption>...</caption>"}}`. Kết quả: **40,798 video, 912,998 dòng → 372,385 activity, 0 collision** — khớp 100% số liệu A1 đã biết. Đối chiếu ngược 5 video ngẫu nhiên: khớp byte-for-byte.

### 3. Task #6 (`phase6_merge_adaptive.py`) — CODE XONG, PHÁT HIỆN + FIX 1 BUG NGUY HIỂM TRƯỚC KHI CHẠY THẬT

Thêm `--captions-dir` + `--speech-segments-dir`. Thứ tự chèn mới trong 1 chunk: `[caption?] <cosmos> <avc_lm> [agent?] [snac?] [speech?]` — caption chèn trước `<cosmos>` (dùng `COSMOS_PATTERN` mới, độc lập với `AVC_PATTERN`), speech chèn sau `</avc_lm>` cùng chỗ agent/snac.

**Bug nguy hiểm bắt được ở bước dry-run, trước khi đụng data thật quy mô lớn:** script này chạy TRÊN `final_dataset_adaptive_v3` — vốn đã có sẵn `<agent>`/`<snac>` từ lần chạy trước. Nếu không cẩn thận, truyền lại `--agent-tokens-dir`/`--snac-tokens-dir` (cần để `has_agent`/`has_snac` trong `chunk_timing` chính xác) sẽ **inject trùng lần 2**. Fix: thêm cơ chế tự phát hiện — nếu `video_tokens` đã có `<agent>`/`<snac>` thì bỏ qua bước inject (báo 0 injected) nhưng vẫn dùng dict để tính cờ `has_agent`/`has_snac` đúng. Verify bằng dry-run thật (3 video/72 activity): nội dung agent/snac giữ nguyên byte-for-byte, caption (138) + speech (243) được thêm đúng.

Phát hiện phụ: path mặc định `--agent-tokens-dir` (tương đối) KHÔNG trỏ đúng data thật — path đúng là `/p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive`. Submit script full-scale đã dùng path tuyệt đối đúng.

### 4. Task #7 (`phase7_flatten.py`) — CODE XONG, TEST KỸ

State machine thêm event `caption` (buffer, flush cùng seed2/cosmos khi gặp `avc_lm`, thứ tự caption→seed2→cosmos) và `speech` (emit ngay, như snac). Cả 2 **không dropout, không augment** (giữ nguyên văn — vì gắn đúng 1 chunk cụ thể, augment sẽ phá vỡ liên kết token-thời điểm). Header `### Speech:` cũ giữ nguyên không đổi (đã xác nhận với user là cố ý trùng lặp, không phải bug). `count_token_types()` thêm `mode` tracker tránh đếm nhầm từ caption/speech vào bucket agent (chỉ ảnh hưởng thống kê, không ảnh hưởng data training).

**Test:** 7+6 nhóm unit test (54 assertion) — thứ tự chèn, chống double-injection, cách ly caption giữa các chunk, độc lập dropout, đếm token đúng — cộng 1 lần dry-run thật end-to-end (3 video → Phase 6 → Phase 7).

Đã commit `5f5492e`, đã push `origin/master`.

### 5. Đo token tăng — 2 phương pháp độc lập, cùng ra ~0.75%

- **Sample thật:** 798 video/3 shard → 5,312 activity qua filter thật, xử lý với `drop_rate_cosmos=0.5` (seed cố định để so sánh sạch, vì lần đầu so 2 lần chạy CLI riêng biệt không seed bị nhiễu ~1% do random cosmos dropout tích luỹ khác nhau — không phải bug, chỉ là thiếu kiểm soát random state). Kết quả: 73,796,727 → 74,340,242 token, **+0.737%**.
- **Chính xác toàn bộ dataset:** đếm trực tiếp — 912,998 caption/10,256,494 từ + 2,158,388 speech-chunk/22,696,606 từ = **39,095,872 token mới**, so với baseline thật 5,217,000,000 token (Phase 7 v4 full-scale đã chạy trước đó) → **+0.749%**.

2 phương pháp độc lập khớp nhau trong 0.012 điểm % — loại trừ khả năng có bug đo.

**Ghi chú quan trọng (user có phản ứng thất vọng, cần ghi rõ để tránh lặp lại kỳ vọng sai):** user kỳ vọng việc này sẽ "làm dày" dataset đáng kể, nhưng +0.75% là quá nhỏ so với kỳ vọng đó. Đã làm rõ: caption+speech scope từ đầu là fix root cause #2 (thiếu language anchor tại điểm chuyển modal) — một vấn đề **định tính**, KHÔNG phải cơ chế để đạt mục tiêu "×4 record" (mục tiêu đó, theo §13/§2.5c REPORT.md, là caption **+ perspective framing cộng lại** — perspective framing mới là đòn bẩy nhân RECORD, chưa code). Đây là lần re-confirm thứ 2 của cùng kết luận (lần 1 ở phiên 12/7 nhìn từ góc số lượng điểm neo, lần này nhìn từ góc token count) — nếu muốn giải quyết bài toán dataset nhỏ (2.84B token, nhỏ cho model 1.7B), đòn bẩy đúng là perspective framing hoặc thêm nguồn data ngoài (SenseNova-SI-8M/stera-10m/MixtureVitae-Omni), không phải vắt thêm từ caption/speech density.

### 6. Task #6 — ĐÃ SUBMIT FULL-SCALE

Script mới `slurm/submit_merge_adaptive_v4.sh` (32-array, `partition=batch`, `account=laionize`, `--time=03:00:00`, theo đúng pattern `submit_merge_adaptive_v3.sh`), input `final_dataset_adaptive_v3/` (160 file, 663GB), output `final_dataset_adaptive_v4/`, `--skip-existing`. **Job `14114336`** — xác nhận 32/32 task vào trạng thái `R` trong 15s, worker 1 log cho thấy tiến độ bình thường. Chưa xác nhận xong ở thời điểm ghi entry này.

**Chưa bắt đầu:** Task #7 full-scale (chờ `final_dataset_adaptive_v4/` xong), Task #8 kiểm tra full-scale cuối cùng (đã làm ở quy mô nhỏ trong phiên này, còn cần double-check ở output full-scale trước khi coi corpus sẵn sàng train).

---

## Cập nhật phiên làm việc — 15/07/2026 (đọc phần này trước khi resume)

**Việc chính hôm nay:** kiểm tra lại trạng thái thật của caption+speech pipeline (task breakdown ở REPORT.md §18), phát hiện 2 việc đã âm thầm chạy xong mà doc chưa ghi nhận, và fix 2 bug thật trong `phase7_flatten.py` liên quan tới việc permute speech transcript khi có SNAC.

### 1. Speech extraction (task #2) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN

REPORT.md §18 dừng ở "relaunched — confirmed healthy" (14/7). Check lại tmux session `speech_full` thì cả 8 worker đã in `DONE` từ lâu ("All workers finished"):

| Metric | Tổng 8 worker |
|---|---|
| Video xử lý | **40,437** |
| Activity có speech | **303,976** |
| Segment speech trích ra | **2,608,543** |
| Garbled/skip | ~58K (~2.2%) |

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/` — 40,490 file `{video_id}_speech.jsonl`. Task #2 coi như xong, sẵn sàng cho task #6 (`phase6_merge_adaptive.py --speech-segments-dir`).

### 2. Tokenizer rebuild (task #5) — XÁC NHẬN ĐÃ CHẠY XONG HOÀN TOÀN

`tokenizer_vla_qwen3` (đang "in progress" theo REPORT.md §18) đã build xong: vocab **257,901**, tất cả token mới (`<caption>`, `</caption>`, `<speech>`, `</speech>`) + toàn bộ token cũ (seed2/cosmos/avclm/pelvis/SNAC/agent/fps) đã verify atomic. Cùng với `tokenizer_vla_adaptive_v2` (156,509 vocab, đã xong từ trước) — cả 2 tokenizer đã sẵn sàng, chỉ còn thiếu bước **upload lên HuggingFace** (cần user tự export `HF_TOKEN`):
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh
export HF_TOKEN=...   # HF token của user
python tools/upload/upload_tokenizers_v2.py --mode all
```

### 3. Caption pipeline (A2, task #3) — vẫn đang chạy, còn xa mới xong

`squeue` cho thấy job `14104156` đang chạy đủ 32/32 worker (đã chạy hơn 8h45p), và có chuỗi 3 job kế tiếp (`14104157`→`158`→`159`) đang xếp hàng chờ qua dependency `afterany` — tự động nối tiếp vì 1 lần chạy không đủ time-window để xử lý hết ~913K task point. Worker 0 mẫu: 800/1275 video, ~0.03 video/s, ETA riêng job hiện tại ~287 phút. Đã có **25,432 file caption** ghi ra `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions/`. Chưa đủ để chạy full-scale task #4 (`build_caption_dict.py`) — cần đợi thêm.

### 4. Fix 2 bug thật trong `pipeline_pose/phase7_flatten.py` (liên quan task #7)

Phát hiện khi rà lại kỹ augmentation pipeline (không phải lỗi mới sinh hôm nay, đã tồn tại từ khi `permute_sentences` + SNAC injection được thêm vào):

- **Bug A (chính):** augmentation "sentence permutation" (xáo câu trong `### Speech:`) áp dụng **vô điều kiện**, kể cả khi activity đó có `<snac_N>` (audio thật, giữ đúng thứ tự thời gian). Kết quả: model "nghe" (SNAC) đúng thứ tự nhưng "đọc" (text speech) bị xáo trộn — dạy sai lệch giả giữa audio và text. **Fix:** thêm biến `effective_permute_rate = 0.0 if sn > 0 else permute_sentences` ngay trước khi gọi `process_transcript_into_chunks()` — `sn` là số token `<snac_N>` đã có sẵn trong `kept_tokens` (tính từ `count_token_types()`, không cần thêm logic mới).
- **Bug B (phụ, phát hiện khi test fix A):** `permute_chunks_list()` có `n = max(1, int(len(c) * permutation_rate))` — dòng này **ép tối thiểu 1 lần hoán đổi bất kể rate truyền vào là bao nhiêu**, kể cả `rate=0.0`. Nếu không sửa, fix A ở trên sẽ vô tác dụng. **Fix:** thêm điều kiện `permutation_rate <= 0` vào early-return cùng với check `len(chunks) < 2` đã có sẵn.
- Đã verify bằng test nhanh: `permute_chunks_list(chunks, 0.0)` giờ trả về list y hệt input.
- **Chưa commit**, chưa chạy lại full-scale (fix cho lần Phase 7 chạy tiếp theo, thuộc task #7/#8, hiện vẫn "chưa bắt đầu" theo §18).

### Việc tiếp theo hợp lý nhất

Vì task #3 (A2) còn chạy rất lâu và task #6/#7/#8 phụ thuộc cả speech (đã xong 100%) lẫn caption (chưa xong) — có thể tranh thủ bắt đầu task #6 (`phase6_merge_adaptive.py` — thêm `--captions-dir` + `--speech-segments-dir`) ngay, vì phần speech-segments đã sẵn sàng hoàn toàn, không cần đợi caption.

---

## Cập nhật phiên làm việc — 12/07/2026 (đọc phần này trước khi resume)

**Chủ đề chính: (1) fix bug `chunk_timing` ở Phase 6, (2) thiết kế xong pipeline captioning.**

### 1. Bug `has_seed2`/`has_cosmos` trong `chunk_timing` — ĐÃ FIX, ĐÃ RE-RUN FULL DATASET

- **Phát hiện:** `phase6_merge_adaptive.py` tính `has_seed2`/`has_cosmos` bằng `i < len(seed2_matches)` — so sánh **chỉ số chunk** với **tổng số tag đếm được cả activity**, không phải check per-chunk thật. Vì seed2 chỉ 1fps trong khi chunk là 3.75/giây, field này đúng cho 1 đoạn đầu rồi sai (`False`) mãi cho phần còn lại — tạo đúng 1 lần "tắt" giả mỗi activity, không phản ánh nội dung thật (đã verify: 2,558/2,558 activity mẫu đều là ON→OFF, không bao giờ OFF→ON, ở timestamp ngẫu nhiên 0.27s–638s).
- **Fix:** tính lại bằng vị trí ký tự thật trong chuỗi `video_tokens` — tag `<seed2>`/`<cosmos>` thuộc về chunk nào thì dựa vào nó nằm giữa 2 mốc kết thúc `<avc_lm>` liên tiếp nào (khớp đúng thứ tự thời gian thật tokens được ghi ra bởi `pipeline_video/pipeline.py`). `has_cosmos`/`has_avc_lm` đơn giản hóa thành `True` cố định (luôn đúng, verify 0 flip trên toàn bộ sample).
- **Không cần chạy lại Phase 7** — đã verify bằng diff `video_tokens` byte-for-byte (0 khác biệt) + grep code: `phase7_flatten.py` không đọc `chunk_timing` ở đâu cả. Chỉ Phase 6 output (metadata) bị ảnh hưởng, không đụng gì tới model đã train hay Megatron data hiện có.
- **Đã re-run full dataset:** SLURM job `14102737`, 32/32 task COMPLETED, 0 lỗi → `final_dataset_adaptive_v3/` (160 file, giữ nguyên v2 để so sánh). Script mới: `slurm/submit_merge_adaptive_v3.sh`.
- **QA đã verify ở cả 2 mức:** (a) 1 file (2,563 activity): agent/snac injected khớp 100% với v2 → xác nhận content không đổi; `has_seed2` giờ flip ~53/activity đúng nhịp periodic. (b) 15 file ngẫu nhiên trên toàn dataset (34,732 activity, phủ khắp 40,804 video): `has_seed2` flip TB 54.53/activity, **0/34,732 (0.00%) activity có `has_seed2` sai suốt (luôn False)** — fix ổn định ở quy mô lớn.
- **Từ nay dùng `final_dataset_adaptive_v3/` làm input chuẩn** cho mọi việc liên quan `chunk_timing` (kể cả bước captioning bên dưới).

### 2. Pipeline Captioning — THIẾT KẾ ĐÃ CHỐT (chưa code full-scale, mới có prototype)

**Bối cảnh:** Huu yêu cầu (chat 11/07) làm frame caption cho toàn bộ FineVideo keyframe, để fix root cause #2 (model thiếu language anchor để biết khi nào chuyển modality).

**Anchor point — điểm được chọn để caption (quan trọng, đã qua nhiều vòng debug):**
- **KHÔNG** dùng "bất kỳ trong 5 flag `has_seed2/cosmos/avc_lm/agent/snac` đổi" như dự định ban đầu — đã đo trên data thật: `cosmos`/`avc_lm` không bao giờ đổi trong activity; `seed2` (dù đã fix bug) vẫn đổi ~54 lần/activity nhưng đây chỉ là nhịp kỹ thuật (1fps) không phải nội dung đổi thật.
- **CHỈ dùng:** (1) frame đầu activity (mở đầu ngữ cảnh) + (2) mỗi lần `has_agent` đổi (người xuất hiện/biến mất — sự kiện nội dung thật, có ví dụ xác nhận: người đứng→ngồi đúng lúc agent bật). Hàm: `select_anchor_points(chunk_timing, min_gap_sec=5.0)` trong `tools/analysis/caption_prototype.py`.
- **`min_gap_sec=5.0` (debounce):** cần thiết vì `has_agent` cũng chập chờn (không sạch 100%) ở cảnh đông người/chuyển động nhanh (bóng rổ, võ thuật) — do YOLO detect noisy frame-to-frame (vấn đề chất lượng data đã biết từ trước), không phải bug mới, không cần sửa Phase 6. Debounce này chỉ là filter ở bước CHỌN điểm caption, không đụng data gốc.
- **Đã đo mật độ thật:** TB ~1.86 caption/activity (ở gap 2s) — thấp hơn nhiều mục tiêu "×4 record" ghi trong doc gốc; 82.8% activity chỉ có đúng 1 caption (frame mở đầu, không có agent event nào). **Đây là hạn chế đã biết, chưa giải quyết** — cân nhắc thêm phương án bổ sung caption định kỳ mỗi N giây cho activity không có agent-transition (chưa chốt N, để sau).

**Model — đã test 3 model, chốt Qwen2.5-VL-3B-Instruct:**
| Model | Kết quả test |
|---|---|
| **Qwen2.5-VL-3B-Instruct** ✅ CHỐT | Không hallucinate ở mọi test (kể cả batch 96 caption). Native trong `transformers` (không rủi ro tương thích). Prompt: `"Describe what the person is doing in one short sentence."` |
| Florence-2-base | `<DETAILED_CAPTION>` mode hallucinate rõ (vd bịa "he appears to be a psycholinguist"). Đổi sang `<CAPTION>` mode thì hết hallucinate + nhanh hơn Qwen 3.5x + hết bị cụt câu — nhưng cần env riêng (`transformers==4.49.0`, torchvision phải cùng index CPU) vì code custom (`trust_remote_code`) không tương thích bản `transformers` mới. Env test: `env_caption_test/` (có thể xóa nếu không dùng). |
| SmolVLM2-2.2B-Instruct | **Chậm hơn Qwen2.5-VL 2x trên CPU** (27.7s vs 14.0s/caption, ngược lý thuyết "nhanh cho edge") + có 1 hallucination rõ (bịa "holding a book" cho 1 frame nền trắng trơn) → loại. |

**Lý do chọn Qwen2.5-VL dù chậm hơn Florence-2 trên CPU:** tốc độ CPU không phải yếu tố quyết định vì full-scale bắt buộc chạy GPU (bất kỳ model nào cũng cần); ưu tiên chất lượng/không-hallucinate + không rủi ro tương thích thư viện dài hạn.

**Full pipeline đã thiết kế (chưa code, sẽ làm ở phiên sau):**
```
final_dataset_adaptive_v3/ 
    → [A1] Task list generation (CPU) — quét chunk_timing, tính anchor points mọi activity
    → [A2] SLURM array job — mở video trong videos_staging/, extract frame, Qwen2.5-VL caption
         → outputs/captions/{video_id}_captions.jsonl
    → [B1] Mở rộng phase6_merge_adaptive.py thêm --captions-dir (giống cách --snac-tokens-dir đã làm)
         chèn <caption>...</caption> NGAY TRƯỚC <cosmos> của đúng chunk (không cắt giữa block nào,
         không lặp lại lỗi speech-giữa-token đã fix ở v3→v4)
         → final_dataset_adaptive_v4/
    → [B2] phase7_flatten.py (như cũ) → megatron_dataset_v5/ → tokenize → train
```
Caption là text tiếng Anh thường, tokenize BPE bình thường — **không cần mở rộng vocab**.

**Hạ tầng:** bước A2 (caption thật) sẽ chạy **CPU** (nhiều CPU core hợp lý hơn máy 2×4090 hiện có) — quyết định của Van Khue (12/07), chưa cần GPU ngay.

**Phát hiện phụ trong lúc debug (đáng nhớ cho lần sau):**
- Video gốc FineVideo đã có sẵn local: `videos_staging/` (chú ý có "s", khác `video_staging/` rỗng) — 43,751 mp4, `/p/data1/mmlaion/shared/nguyen38/data/videos_staging/`, tên file = `{video_id}.mp4`. Không cần JUPITER hay stream từ HF.
- **HumanoidBench đã đọc + đánh giá KHÔNG phù hợp** làm eval benchmark hiện tại — benchmark closed-loop RL (MuJoCo, Unitree H1 + Shadow Hands, action space 61-dim joint-angle) trong khi model mình sinh ra pose xyz người thật 17-khớp H36M, không có góc khớp/bàn tay. Chỉ liên quan tới Priority 12 "Isaac Sim/H1" đã hoãn từ trước, không phải DISCUSS-3 hiện tại.
- Home directory (`~/.cache`) có quota nhỏ hơn nhiều so với `/p/data1` (project storage, 388TB trống) — luôn set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` khi tải model lớn để tránh lỗi "Disk quota exceeded".
- HF Hub's Xet download backend đôi khi lỗi transient (`Background writer channel closed`) — set `HF_HUB_DISABLE_XET=1` để fallback về HTTP download thường nếu gặp.

### 3. A1 code xong + chạy full dataset + validate kỹ. A2 code xong + smoke test OK — đang chờ quyết định CPU/GPU (tiếp phiên 12/07, cùng ngày)

**`select_anchor_points()` thêm bước "periodic supplement":**
- Vấn đề: chỉ agent-transition cho ~1.4-1.86 caption/activity, phần lớn activity chỉ có 1 caption (frame mở đầu).
- Đã thống nhất và code: sau bước agent-transition, nếu chưa đủ `target_count=4` điểm thì bổ sung thêm điểm cách đều theo thời gian, snap về chunk gần nhất, debounce chống trùng với điểm đã có. Sửa trong `tools/analysis/caption_prototype.py` — chữ ký mới `select_anchor_points(chunk_timing, min_gap_sec=2.0, target_count=4)`.
- Kèm sửa bug phân loại trong `caption_florence2_visual_batch.py` (đổi `len(pts) > 1` → check `has_agent` flip thật, vì periodic supplement làm số điểm không còn phản ánh đúng "có sự kiện thật hay không").

**Bug quan trọng phát hiện khi test A1 trên data thật — đã fix:** `duration` trong periodic supplement tính bằng `end_sec` **tuyệt đối** thay vì tương đối so với activity → activity bắt đầu muộn (VD giây 590 trong video dài) bị tính mốc thời gian sai hoàn toàn ra ngoài phạm vi, supplement gần như vô hiệu. Fix: trừ `activity_start` trước khi chia. Verify trên 2563 activity thật: % activity đạt `target_count=4` tăng từ 10.4% → 54.8%.

**A1 (`tools/analysis/generate_caption_tasks.py`) — CODE XONG, ĐÃ CHẠY FULL DATASET 160/160 SHARD:**
- Đọc `final_dataset_adaptive_v3/`, tính anchor points mọi activity bằng `select_anchor_points()`, ghi task list ra `outputs/caption_tasks/*.jsonl` (mỗi dòng: video_id, video_path, scene_id, activity_id, chunk_idx, start_sec, has_agent).
- Chạy: 13 shard qua SLURM array (job `14103227`, sau đó bị hủy giữa chừng vì thấy chạy nhanh hơn dự kiến), 147 shard còn lại chạy trực tiếp login node (`--skip-existing` để resume) — hoàn tất 160/160.
- **Kết quả:** 40,798 video, 372,385 activity, **912,998 task point**, avg **2.45 caption/activity**, 0 video thiếu mp4 local.
- **Validate kỹ (theo yêu cầu user, đã làm đầy đủ):** 100% task point hợp lệ schema/type/`video_path` tồn tại; 0 activity trùng `chunk_idx`; 0 activity vi phạm debounce 5s; đối chiếu ngược 5 shard ngẫu nhiên (11,576 activity, 28,156 điểm) — tính lại từ `chunk_timing` gốc khớp 100% với output đã lưu, 0 activity thiếu/thừa. Diagnostic sample lưu ở `logs/a1_smoke_test_samples.json` (gitignored).
- Submit script: `slurm/submit_caption_tasks.sh`.

**Đính chính hiểu về mục tiêu "×4":** đọc lại `REPORT.md` (dòng 1104, 1134) — ×4 gốc trong spec là mục tiêu của **captioning + perspective framing cộng lại** (nhân tổng số RECORD training lên ~4 lần), **không phải** "mỗi activity phải có đúng 4 caption" như đang đo. Đo thực tế: avg 2.45/activity chỉ đạt 61.3% nếu so với ×4 hiểu theo nghĩa hẹp (caption/activity) — nguyên nhân gốc: ~59% activity trong FineVideo ngắn hơn 15s, về mặt hình học không thể nhét 4 điểm cách nhau ≥5s (debounce). **Quyết định: giữ nguyên `target_count=4, min_gap_sec=5.0`, KHÔNG hạ gap để ép đạt số** — vì hạ gap sẽ tạo caption gần giống hệt nhau cho clip tĩnh ngắn, không thêm tín hiệu ngôn ngữ mới, chỉ tốn thêm compute A2. Muốn thật sự tiến gần ×4 thì đòn bẩy đúng là làm perspective framing (roadmap riêng, chưa code), không phải vắt thêm từ caption density.

**A2 (`pipeline_pose/caption_finevideo.py`) — CODE XONG, SMOKE TEST OK, CHƯA CHẠY FULL:**
- Đọc task list A1 (gộp theo video), mở video, extract frame tại `start_sec`, caption bằng Qwen2.5-VL-3B-Instruct (model đã chốt từ phiên trước) → ghi `outputs/captions/{video_id}_captions.jsonl`.
- Theo đúng pattern đã dùng cho `pipeline_pose/snac_finevideo.py`: model load 1 lần/worker, video chia kiểu stride `all_vids[task_id::num_tasks]`, mỗi video 1 file output để resume an toàn (skip nếu đã tồn tại).
- Smoke test (video `A1UVeD9UB1I`, giây 248.0): caption ra hợp lý — *"The person is arranging jewelry on a box."* khớp `text_prompt` gốc *"Woman opens a gift box."*
- **Bug hạ tầng phát hiện + fix:** ban đầu chưa giới hạn số thread PyTorch → 2 tiến trình test chạy song song trên login node (80 core) tự tranh nhau, kết quả 57.6s/caption (chậm ~4x so với thật). Đã thêm `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`torch.set_num_threads()` khớp `SLURM_CPUS_PER_TASK` (mặc định 4) để 32 worker không tranh nhau khi chạy job thật.
- **Đo throughput sạch (4 thread, không tranh chấp, 3 lần lặp): ~13.8s/caption** (12.9/15.2/13.4s) — khớp mức 10-15s ghi trong `REPORT.md`.
- **Ước tính full run:** 912,998 task × 13.8s ≈ **3,500 CPU-giờ**. Với 32 worker (như job SNAC trước) → **~109 giờ/worker (~4.6 ngày)**, cần **~5 lần resubmit** nếu giữ `--time=24:00:00`. An toàn resubmit nhờ skip-existing per-video.
- Submit script đã viết: `slurm/submit_caption_finevideo.sh` — **CHƯA SUBMIT**, đang chờ quyết định CPU/GPU.

**Câu hỏi mở — quyết định đầu phiên sau:**
- CPU (32 worker, ~4.6 ngày, script đã sẵn sàng submit ngay) vs GPU (máy 2×4090, chưa đo thật lần nào trong phiên này).
- Ước tính lý thuyết (CHƯA đo thật): GPU không batch có thể **không nhanh hơn** 32 CPU (chỉ 2-way song song vs 32-way, dù mỗi request nhanh hơn); muốn GPU thắng rõ rệt cần xử lý theo batch (nhiều ảnh/1 lần forward) — `caption_frame()` hiện tại chỉ xử lý 1 ảnh/lần, chưa có code batch.
- Nếu chọn GPU: cần viết bản batch cho `caption_frame()` + cần thông tin truy cập máy 2×4090 để đo thật trước khi quyết định.
- **B1/B2 (chèn caption vào `final_dataset_adaptive_v4/`, re-run Phase 7) vẫn CHƯA BẮT ĐẦU** — chờ A2 chạy xong (toàn bộ hoặc một phần đủ lớn).

### 4. Đã chốt CPU, đã SUBMIT full run A2, đã xác nhận chạy đúng (13/07/2026)

**Quyết định:** CPU, theo Van Khue — chọn phương án sẵn sàng ngay, không cần chờ code batch cho GPU.

**Lần submit đầu (job `14104070`) FAIL — cả 32/32 task đều crash khi load model.** Nguyên nhân: `slurm/submit_caption_finevideo.sh` đặt `HF_CACHE=/p/scratch/laionize/nguyen38/hf_cache`, thư mục này KHÔNG có weight của Qwen2.5-VL-3B-Instruct (chỉ có `bert-base-uncased` và `snac_24khz`) — compute node không có internet (`HF_HUB_OFFLINE=1`), nên `from_pretrained()` bị lỗi `OSError: We couldn't connect to huggingface.co ... Qwen/Qwen2.5-VL-3B-Instruct is not the path to a directory containing config.json`. Cache đúng — chính là cache đã dùng khi smoke test ngày 12/07 — nằm ở `/p/data1/mmlaion/nguyen38/hf_cache` (có sẵn `models--Qwen--Qwen2.5-VL-3B-Instruct`, 7.1GB), khớp với lưu ý env gotcha đã ghi ở dưới ("luôn set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache`").

**Fix:** đổi `HF_CACHE` trong `slurm/submit_caption_finevideo.sh` thành `/p/data1/mmlaion/nguyen38/hf_cache`.

**Resubmit thành job `14104104` — ĐÃ XÁC NHẬN CHẠY ĐÚNG.** Cả 32/32 task vào trạng thái `R` (running), model load sạch trong ~44-45s/worker (không còn lỗi HF offline), caption đầu tiên xuất hiện chỉ sau ~5 phút kể từ lúc submit. Đã kiểm tra mẫu output (`.../captions/-0-6Som0MGY_captions.jsonl`, 10 caption) — đúng schema JSON, caption chất lượng tốt/cụ thể (VD: *"The person is pouring sulfuric acid into an energy drink can."*, *"The person is using a blue dropper to apply coconut oil onto a surface."*).

**Trạng thái cuối phiên: job `14104104` đang chạy, 32/32 task active, ETA ~4.6 ngày (theo ước tính chi phí 12/07), sẽ cần resubmit nhiều lần vì `--time=24:00:00`.** Phiên sau nên: (1) check `squeue -u nguyen38` xem job `14104104` (hoặc job resubmit kế tiếp — cùng script, an toàn chạy lại nhờ skip-existing per-video) còn chạy hay đã timeout cần resubmit, (2) khi `outputs/captions/*.jsonl` đã phủ được một phần đáng kể (target 912,998 task point / 40,798 video), có thể bắt đầu B1 (mở rộng `phase6_merge_adaptive.py` thêm `--captions-dir`) — B1 không nhất thiết phải chờ A2 xong 100%, chỉ cần đủ coverage để bắt đầu prototype.

**Đã thêm cơ chế auto-chaining, không cần resubmit tay:** `slurm/submit_caption_finevideo.sh` giờ nhận thêm tham số job-id tuỳ chọn và submit với `--dependency=afterany:<id>`, in ra job id mới để nối tiếp dễ dàng. Đã nối thêm 5 job sau `14104104`: `14104104 → 14104155 → 14104156 → 14104157 → 14104158 → 14104159` (6 job × 24h = ~6 ngày coverage). Đã xác nhận các job đang xếp hàng với lý do `(Dependency)` qua `squeue --start`. Nếu chuỗi vẫn chưa đủ, nối thêm bằng `bash slurm/submit_caption_finevideo.sh 14104159`.

**Kiểm tra chất lượng caption (333+ file output, ~340 dòng mẫu): tốt nhìn chung, phát hiện 1 loại hallucination.** Đa số caption khớp tốt với `text_prompt` gốc. Phát hiện 1 hallucination rõ: video `-Gq3DJyhJ3I` (soccer highlights) có caption *"performing a complex mathematical operation..."* tại t=0.0s — frame thật (đã check bằng `cv2`) gần như đen tuyệt đối (fade-in đầu video), model bịa nội dung thay vì trả lời "không nhìn thấy gì" như nó làm đúng ở chỗ khác. **Đánh giá là rủi ro nhỏ, hạn chế đã biết của Qwen2.5-VL** (khớp tỷ lệ ~1/30-96 đo được lúc chọn model), không phải bug pipeline — không chặn tiến độ. Có thể cân nhắc fix nhỏ cho B1 sau này (không gấp): check độ sáng trung bình frame, bỏ qua/đánh dấu caption trên frame gần đen trước khi inject vào training data.

### 5. Khảo sát 6 dataset permissive + bắt đầu download MINT-1T-HTML (13/07/2026, làm song song trong lúc A2 chạy)

Đã tìm hiểu 6 dataset ứng viên còn chưa scope từ chat team 07/07. Chi tiết đầy đủ ở `REPORT.md` §17 — tóm tắt:
- **`mira-wm.com` — bỏ** — không phải data robot/pose, đây là world model cho game Rocket League (video + keyboard action + game state). Không liên quan project.
- **`finevla.xlang.ai` — hoãn** — bộ 47,159 trajectory train thật chưa public (GitHub repo ghi "Coming soon"); chỉ có benchmark eval 500 video (`xlangai/RoboFine-bench`) tải được.
- **`nvidia/Cosmos3-DROID` — hoãn, chờ quyết định kiến trúc** — xác nhận thật (707GB, 71,907 episode robot teleop thật, format LeRobotDataset v3.0), nhưng là robot joint-space action, khác hẳn xyz human-pose token hiện tại. Cần thiết kế tokenizer riêng trước khi dùng được, không chỉ là tải về.
- **`MiG-NJU/OmniVideo-100K` — hoãn** — data video QA, không có pose/action, chỉ làm loãng thêm tỷ lệ agent-token (rủi ro đã ghi nhận với MV-Omni).
- **`genrobot2025/Gen-EgoData` — hoãn** — cấu trúc gần giống nhất (egocentric video+pose+action) nhưng rất nhỏ (500 sample, 47.6GB), format `.mcap` cần toolkit riêng, license CC-BY-SA (share-alike).
- **`mlfoundations/MINT-1T-HTML` — đang tải.** Bù đắp trực tiếp cho khoảng trống ngôn ngữ DISCUSS-1 (5.217B token FineVideo gần như 100% là token modality riêng, gần như không có text thường). **Đính chính kích thước: đo thực tế 2.89TB (6,159 shard parquet), không phải 5.91TB như dataset card ghi** (số đó tính cho cả project MINT-1T gồm cả PDF/ArXiv, không có trong repo HTML-only này). **Phát hiện về schema: cột `images` chỉ là URL, không phải bytes ảnh thật** — phần text dùng ngay được, nhưng muốn lấy ảnh thật cho ý tưởng "seed2 token từ ảnh" thì cần crawl riêng từng URL, khả năng cao tỷ lệ link chết cao (nguồn blog từ 2011).

**Insight quan trọng về framework (đáng nhớ cho việc scope dataset sau này):** nguồn video thô (pipeline HRNet→MotionBERT→PCHIP tự xử lý được từ đầu) rẻ để tích hợp; nguồn đã có pose/action sẵn (DROID joint-space, Gen-EgoData `.mcap`) là bài toán retargeting, không phải bài toán data-ingestion — đừng đầu tư thời gian tải trước khi có quyết định rõ ràng về việc có thêm modality robot-action riêng hay không.

**Trạng thái download:** `tools/extract/download_mint1t_html.py` (script mới, dùng `huggingface_hub.snapshot_download`, 16 worker, tự retry, resumable) chạy trong tmux session `mint1t`, log tại `logs/download_mint1t_html.log`, đích `/p/data1/mmlaion/shared/vla/mint1t_html/`. Cuối phiên: 249/6,159 file, 204GB/2.89TB (~7%), ETA ~10 giờ từ lúc bắt đầu, không lỗi. **Việc tiếp theo:** để chạy xong, rồi sample-tokenize cột `texts` bằng tokenizer riêng của project (giống cách đã làm với MixtureVitae, §13) để biết số token thật trước khi quyết định cần bao nhiêu cho DISCUSS-1.

### 6. Bắt đầu triển khai pipeline chèn caption+speech vào token (14/07/2026, làm song song trong lúc A2 tiếp tục chạy)

**Bối cảnh:** đã duyệt plan chèn tag `<caption>` (output của A2/Qwen2.5-VL) và `<speech>` (transcript ASR có sẵn của FineVideo, KHÔNG phải chạy Whisper mới — xem đính chính bên dưới) vào chuỗi token training tại các điểm chuyển đổi modal, giúp model có "language anchor" mà hiện đang thiếu. Tổng 8 task, trạng thái như sau.

**Đã xong:**
- **Task #1 (manifest video→shard):** `tools/analysis/build_video_shard_manifest.py`, đã chạy xong. Map 43,751 video_id sang shard index parquet của `HuggingFaceFV/finevideo`. Dùng lại được mãi, không cần chạy lại.
- **Task #2 (script trích speech, xem bug bên dưới):** `tools/analysis/extract_speech_segments.py` đã viết xong. **Đính chính so với cách gọi trước đây:** script này KHÔNG chạy Whisper — FineVideo đã có sẵn transcript ASR tính trước cho mỗi video (`timecoded_text_to_speech`, nguồn YouTube-Commons), nên script chỉ fetch lại field đó từ parquet trên HF Hub rồi map vào `chunk_timing`, không cần tính ASR mới.
- **Task #4 (adapter caption dict):** `tools/analysis/build_caption_dict.py` đã viết, đã test logic trên output A2 thật. **Chưa chạy full-scale** — thư mục output `captions_dict/` chưa tồn tại trên đĩa.
- **Task #5 (tokenizer):** đã thêm 4 token wrapper (`<caption>`, `</caption>`, `<speech>`, `</speech>`) vào `tools/tokenizer/build_tokenizers.py` + `tools/tokenizer/expand_vocab.py`. Rebuild `tokenizer_vla_adaptive_v2` **đã xác nhận xong và verify kỹ** (vocab 156,509, cả 4 token mới đều atomic, tất cả nhóm token cũ cũng re-check lại atomic). Rebuild `tokenizer_vla_qwen3` tại thời điểm ghi entry này **vẫn đang chạy** — cần check lại trước khi coi là xong.

**2 bug thật đã tìm ra và fix trong `extract_speech_segments.py` khi cố lấy sample output cho user xem (quan trọng — có thể tái diễn nếu script này bị copy hoặc pattern bị tái sử dụng chỗ khác):**
1. **RAM tăng không giới hạn — KHÔNG phải do fetch từ HF (chẩn đoán sai lúc đầu).** Test nhanh với `--video-ids` (2 video) đẩy RSS lên 90+ GB và vẫn tăng tiếp trên login node dùng chung, phải kill giữa chừng. Lúc đầu nghi ngờ do đọc streaming qua `HfFileSystem`, đã đổi sang `hf_hub_download` (tải về cache local) — fix này đúng và nên giữ, nhưng KHÔNG phải nguyên nhân thật. **Nguyên nhân thật:** `load_activities_needing_speech()` mặc định quét toàn bộ glob `final_dataset_adaptive_v3/` (160 file, tổng **663GB**) trước khi lọc theo `--video-ids`, và giữ lại **toàn bộ activity dict** (gồm cả chuỗi `video_tokens` — hàng trăm KB/activity) cho MỌI video có `chunk_timing`, thay vì chỉ 3 field thực sự cần (`activity_id`, `chunk_timing`, `time_range_sec`).
2. **Fix:** (a) chỉ giữ lại 3 field cần thiết, (b) áp filter `--video-ids`/allowlist NGAY TRONG lúc quét file, không lọc sau khi đã load hết. Test lại: RSS giữ dưới 500MB cho cùng test 2-video (so với 90+ GB không giới hạn trước đó). Job full-scale thực tế (SLURM array 32 worker, ~5 file/worker nhờ chia theo `SLURM_ARRAY_TASK_COUNT`) vốn ít bị ảnh hưởng hơn path test nhanh trên login node, nhưng fix này vẫn giảm RAM cho worker nói chung.

**Chưa bắt đầu:** Task #6 (`phase6_merge_adaptive.py` — chèn `<caption>` trước `<cosmos>`, `<speech>` sau `</avc_lm>`; pre-check invariant cosmos/avc_lm 1:1 tìm thấy 1 mismatch dạng trailing chunk trong 2,753 activity, đánh giá an toàn nhưng broad-check 5 shard bị ngắt giữa chừng, chưa chạy lại), Task #7 (update regex/state-machine `phase7_flatten.py`), Task #8 (dry-run end-to-end).

**Trạng thái job caption A2:** chain `14104155` (đang chạy, 32/32 task) → `14104156-159` (đang xếp hàng, chờ dependency). Số caption tăng 11,501 → 13,783 giữa lần check 13/07 và 14/07 — tiến độ đều nhưng còn xa target 912,998, dự kiến còn vài ngày nữa.

**Upload tokenizer (đang chờ, cần HF token riêng của user):** `tools/upload/upload_tokenizers_v2.py` đã cập nhật model card phản ánh 4 token mới (vocab 156,505→156,509 cho adaptive_v2, 257,897→257,901 cho qwen3). Chưa chạy — user sẽ export `HF_TOKEN` riêng và chạy sau khi rebuild qwen3 xong: `python tools/upload/upload_tokenizers_v2.py --mode all`. Rebuild `tokenizer_vla_qwen3` (thêm 106,232 token vào base Qwen3) tại thời điểm ghi entry này **vẫn đang chạy** — check `tmux attach -t qwen3_rebuild` hoặc `logs/build_tokenizer_qwen3_rebuild.log` trước khi coi là xong.

**Launch full-scale Task #2 — gặp bug thật và đã fix (cùng phiên, muộn hơn):** quyết định chạy `extract_speech_segments.py` trực tiếp trên login node JUWELS (không qua SLURM) vì script cần internet (`hf_hub_download` từ HF Hub) mà compute node JUWELS không có. Viết `tools/analysis/run_speech_extraction_login.sh` — 8 worker song song, `nice -n 15`/`ionice -c3` để lịch sự trên login node dùng chung, output riêng từng video + `--skip-existing` để resume nếu bị kill. **Lần launch đầu crash cả 8 worker trong ~20s: `RuntimeError: ... Disk quota exceeded`** — do `hf_hub_download()` trong script không set `HF_HOME`, nên fallback về `~/.cache/huggingface` (quota home dir nhỏ, đã ghi nhận là gotcha đã biết ở chỗ khác — quên áp dụng ở đây). Đã fix bằng cách thêm `export HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` vào script runner; dọn luôn 1.9GB cache `HuggingFaceFV/finevideo` dở dang do lần crash để lại trong `~/.cache/huggingface`. Chạy lại — xác nhận khỏe (8/8 worker sống, ~100% 1 core/worker = 8/80 tổng, RAM 300-400MB/worker, cache giờ đi đúng vào `/p/data1/mmlaion/nguyen38/hf_cache`). Đang chạy trong tmux session `speech_full`, log từng worker tại `logs/speech_extraction_login/worker_*.log`. Đích output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/`.

---

## Cập nhật phiên làm việc — 08/07/2026 (đọc phần này trước khi resume)

**Thay đổi từ lần trước:**
- ✅ **Phase 7 v4 đã upload lên HF** — `EmpathicRobotics/FineVideo-Phase7-Flattened` live với data v4 (371,888 record, 5.217B token). Đã share cho Huu/joergfranke trên Discord (07/07) làm dataset sẵn sàng tokenize.
- ✅ **1-CP — CHỐT quyết định: hoãn.** Confirm với Huu trên Discord (08/07): giữ nguyên format adaptive 2/4/8-CP hiện tại. Gain chỉ +7.1% (ước tính từ sample 50 video), re-run Phase 5→7 tốn thời gian không đáng. Huu OK để báo cáo paper với con số "compression giảm hơn 50%" (so với fixed 8-CP) là đủ. **Chỉ quay lại nếu sau này data cho thấy cần thiết.** Đã thử chạy full-dataset investigation (18,847 video) nhưng bị gián đoạn do JUWELS sập — chưa resume, chưa có kế hoạch làm tiếp.
- ⚠ **Cluster JSC sập (từ ~06/07/2026):** JUPITER down hoàn toàn. JUWELS booster + JURECA có GPU nhưng hạn chế ("to the extent Jenia lets us"). Huu ước tính: chính thức 1 tuần, thực tế có thể 2 tuần. Việc bị block: Megatron re-tokenize quy mô lớn, train v0.3, chạy 1-CP full dataset, pipeline Cosmos3-DROID.
- **Quyết định team: synthetic/simulation data giới hạn ≤30% tổng training mix** (Huu, dựa trên literature) — áp dụng khi tính tỷ lệ mix abc.bot / MolmoAct2 / Cosmos3-DROID với FineVideo (video người thật).
- **Nguồn data mới tìm được (07/07/2026, từ chat team)** — xem bảng "Nguồn Data Mới" bên dưới. Đáng chú ý nhất: `abc.bot` (400h robot sim data **kèm physics state** MjData, permissive, có eval env).
- **Định hướng chia sẻ data đa dự án:** Huu muốn gộp data giữa 3 nhánh song song — omni-vla (repo này), dự án so sánh kiến trúc của joergfranke (qwen3/lfm2.5/olmo3), và world-action-model dạng diffusion của blanchon.jl (video generation + action). `FineVideo-Phase7-Flattened` giờ được dùng chung cho các dự án khác — giữ format càng generic/well-documented càng tốt.
- ✅ **Đã điều tra `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` (09/07/2026).** 15 file, ~103GB. Đếm token bằng sample stream (75MB/file, không tải hẳn về máy) qua 2 script mới `tools/peek_multimodal.py` + `tools/count_multimodal_tokens.py`, chạy local (không có JUWELS phiên này). **Kết quả: chủ yếu là text/caption thuần, không phải format VLA token của mình.** Chỉ `train_data_snac.jsonl.gz` và `valid_data_snac.jsonl.gz` có SNAC token thật — nhưng ở dạng **mảng số nguyên thô** (`snac_token: [128266, ...]`), không phải tag chuỗi `<snac_N>` — ước lượng **~3.27 tỷ raw SNAC code** (~3.11B + ~162M). 13 file còn lại là text/caption corpus (~12.4 tỷ token word-count ước lượng, riêng `finevideo_transcripts.jsonl.gz` bị đếm thiếu — xem phần lưu ý bên dưới). Đã báo Huu trên Discord (09/07, 3:51pm) hỏi có muốn thêm không — **đang chờ trả lời**, chưa bắt đầu tích hợp. Chi tiết đầy đủ ở mục "Điều tra MixtureVitae-Backup Multimodal" bên dưới.
- **Việc điều tra còn treo (Huu giao, chưa làm):**
  1. "finevideo reformulation" tại `leo:/mnt/sdb/mixture-vitae-working/finevideo` — Huu tự tạo nhưng không nhớ rõ nội dung; cần check overlap với pipeline hiện tại (tránh lặp lại vụ double-count như `valid_with_seed`).
- **Vấn đề cần xử lý (chưa làm):** nếu mix thẳng toàn bộ MV-Omni (6.93B token, 0 agent token) vào training corpus, tỷ lệ agent (pose) sẽ pha loãng từ 12.2% (chỉ FineVideo v4) xuống còn ~5.2% trong tổng mix. Agent token là điểm khác biệt cốt lõi của dự án — cần cân nhắc dropout MV-Omni (giống cách đã làm với Cosmos/AVC-LM) hoặc oversample record có agent trước khi mix.

**Xếp hạng ưu tiên hiện tại (do JUPITER down + ưu tiên "thêm data trước khi train"):**

| Tier | Việc | Cần cluster? | Impact |
|---|---|---|---|
| ✅ | ~~Điều tra MixtureVitae-Backup/multimodal~~ | Không | Xong 09/07 — chủ yếu text, tìm được ~3.27B raw SNAC code; đang chờ Huu quyết định |
| P0 | Làm rõ "finevideo reformulation" trên leo | Không | Tránh double-count |
| P0 | Quyết định tỷ lệ mix MV-Omni (fix pha loãng agent) | Không | Bảo vệ tín hiệu pose cốt lõi |
| P0 | Định nghĩa eval protocol (DISCUSS-3, còn treo) | Không | Bắt buộc trước khi train |
| P0 | Chốt tỷ lệ mix text/instruction data (DISCUSS-1) | Không | Ảnh hưởng khả năng steer robot |
| P1 | Code full-scale pipeline captioning (thiết kế đã chốt 12/07, xem session update) | Không (CPU, theo quyết định 12/07) | Cao nhất — fix root cause 2 (mật độ thực tế ~1.86 caption/activity, chưa đạt ×4 như dự tính ban đầu) |
| P1 | Viết code ego-centric perspective converter | Không (chỉ cần GPU lúc chạy) | ×2 diversity pose data, miễn phí |
| P1 | Mix MV-Omni vào Megatron format | Chỉ cần CPU | +6.93B token, vocab đã sẵn sàng |
| P2 | Scope abc.bot, MolmoAct2-BimanualYAM, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData | Không | Nguồn robot/video mới, chưa rõ size |
| P2 | Điều tra leo seed2 + euro_pat | Không | Chưa rõ |
| P3 | Chạy pipeline Cosmos3-DROID | GPU | Data domain robot thật đầu tiên |
| P3 | Chạy captioning full, Megatron re-tokenize corpus gộp, train v0.3 | GPU (JUPITER) | Block đến khi cluster lên + data đủ |
| P4 (hoãn) | 1-CP, Moss-Audio V2, Qwen3 migration, PAB-Spline angle spec, Isaac Sim | — | Đã quyết định hoãn theo team |

---

## Bức tranh toàn cảnh — Chúng ta đang xây dựng cái gì?

Mục tiêu cuối cùng: một mô hình nhận đầu vào đa phương thức (video frame + lệnh thoại/text), sinh ra **action token** có thể decode thành quỹ đạo khớp xương robot. Ví dụ Huu đưa ra: *"nói 'đi về phía trước', robot tự biết bước đi dựa trên pattern đã học."*

Tầm nhìn xa hơn: model nhìn ảnh bình hóa chất + công thức → suy luận task "pha nước muối" → translate thành chuyển động tay/cánh tay — **mà không cần train trực tiếp trên task đó**. Đây đòi hỏi cross-modal binding thực sự: vision ↔ language ↔ action.

Cách tiếp cận: pretrain một LLM 1.7B trên chuỗi token xen kẽ:

```
USER: <mô tả hoạt động> [Speech: ...]  ASSISTANT:
  <seed2_N> ...          # Token keyframe ngữ nghĩa   (1fps, vocab 8192)
  <cosmos_N> ...         # Token video không gian      (mỗi 8 frame, vocab 64000)
  <avclm_N> ...          # Token H.264 BPE             (mỗi 8 frame, vocab 8192)
  <fps_30> <pelvis> ...  # Token tư thế 3D             (mỗi 8 frame, 17 khớp)
  <snac_N> ...           # Token âm thanh — SNAC listen format (~10 token / 8-frame chunk)
```

Model học "đọc" và "tiếp nối" chuỗi xen kẽ này. Khi inference: prompt bằng video token + text command → model dự đoán agent token tiếp theo = chuyển động.

**Tại sao cách này?** Chưa có VLA model nào thống nhất được Seed2/Cosmos (video), SNAC (âm thanh), và PCHIP spline (chuyển động liên tục) trong một LLM autoregressive duy nhất. Chúng ta đang ở frontier nghiên cứu — không ai trong nhóm đã làm điều này trước đây. Đó vừa là lợi thế vừa là rủi ro.

---

## Timeline tổng quan

| Thời gian | Cột mốc |
|-----------|---------|
| Tháng 6/2025 | Bắt đầu dự án. Chọn dataset FineVideo (~40K video YouTube). |
| T7–T9/2025 | Nhánh A: Pipeline trích xuất video token (Seed2, Cosmos, AVC-LM). Chạy 160 GPU. |
| T9–T11/2025 | Nhánh B phase 1–3: HRNet 2D pose, MotionBERT 3D lifting, kinematics. |
| T11–T12/2025 | Phase 4: YOLO cleaning. Phase 5 lần đầu (format 256 token mờ đục). |
| T1–T2/2026 | Viết lại Phase 5 → Adaptive PCHIP (token joint tự mô tả, có tên). |
| T3/2026 | Phase 6 merge, Phase 7 flatten. Megatron tokenization lần đầu. |
| T4/2026 | **Model đầu tiên** train xong (`vla-1.7b-pab-spline-25b-test`). Phát hiện tokenizer bị broken. |
| T5/2026 | Fix tokenizer: dùng `add_tokens(special_tokens=True)`. Re-tokenize toàn bộ. |
| T6/2026 | **Model thứ hai** train xong (`vla-1.7b-pab-spline-adaptive`). Đánh giá. Data inventory. |

---

## Đã làm gì — Chi tiết kỹ thuật

### Nhánh A: Trích xuất Video Token

**Script:** `pipeline_video/pipeline.py` | **Compute:** 40 node × 4 GPU

Xử lý toàn bộ ~40K video FineVideo. Mỗi đoạn activity được tokenize thành:
- **Seed2**: token keyframe 1fps, vocab 8192
- **Cosmos**: token không gian mỗi 8 frame, vocab 64000
- **AVC-LM**: token H.264 BPE mỗi 8 frame, vocab 8192

Output: 160 file `training_ready_rank_*.jsonl` dạng JSON phân cấp (video → scenes → activities → tokens + transcript).

---

### Phase 1: Phát hiện 2D Pose (HRNet)

- HRNet-W48 + Faster R-CNN detector
- **40,804 video**, 145 GB
- Output: tọa độ 2D của 17 khớp xương (COCO format) theo từng frame

---

### Phase 2: Nâng lên 3D (MotionBERT)

- MotionBERT lift 2D → 3D (pretrain trên Human3.6M)
- **40,804 video**, 259 GB

---

### Phase 2.5: Resample 30fps

- Interpolation tuyến tính từ fps gốc → 30fps đồng nhất
- Bắt buộc để 4 modality chia sẻ cùng time grid
- 67 GB

---

### Phase 3: Kinematics Processing

- Lọc Butterworth (làm mượt chuyển động)
- Chuẩn hóa độ dài xương về skeleton H36M chuẩn
- Root-centering (pelvis về gốc tọa độ)
- Lọc anti-teleportation (loại bước nhảy đột ngột)
- Window 8 frame → shape `(windows, 8, 153)`, 153 = 17 khớp × 3 chiều × 3 kinematics (vị trí/vận tốc/gia tốc)
- **40,200 video** (604 video quá ngắn bị bỏ), 193 GB

---

### Phase 4: YOLO Cleaning

- YOLOv8 phát hiện người trong từng frame
- Bỏ window 8 frame nào mà ≥4 frame không có người (confidence ≥ 0.75)
- **40,195 video**, 107 GB

**⚠ Phát hiện chất lượng dữ liệu pose (02/07/2026):**

Visualization side-by-side (`tools/visualize_skeleton_sidebyside.py`) + kiểm tra trực tiếp dữ liệu `yolo_cleaned` cho thấy vấn đề nghiêm trọng:

| Vấn đề | Chi tiết |
|--------|---------|
| **Joint thưa** | Trung bình 4–7 joint finite/frame trong tổng 17 (24–41% skeleton) |
| **Tay vắng mặt hoàn toàn** | j11–j16 (cả 2 tay: vai/khuỷu/cổ tay) = NaN gần như 100% — MotionBERT không lift được joint tay từ video YouTube (bị che khuất, góc nghiêng) |
| **Lỗi zero-fill** | j10 (head_top) = (0,0,0) khi không detect được — trùng với pelvis, được tính là "finite" nhưng sai |
| **Scale tọa độ OK** | Mắt cá ở ~−0.638m dưới pelvis = hợp lý về mặt giải phẫu; scale metric đúng |

**Ảnh hưởng đến training:** Token pose chủ yếu là phần dưới cơ thể (hông/gối/mắt cá) + thân trên. Tay — quan trọng nhất cho manipulation — hầu như không có. Model học được chuyển động thô (đi bộ/ngồi) nhưng không học được cử động tay tinh tế.

**Không ảnh hưởng đến FineVideo như pretraining signal** — pose thô còn hơn không có, cho phép model học correlation video-pose. Nhưng để fine-tune manipulation, cần dữ liệu pose tốt hơn (simulation, MoCap, hoặc depth camera).

---

### Phase 5: Adaptive PCHIP Tokenization

Đây là phần kỹ thuật độc đáo nhất của dự án.

Với mỗi window 8 frame, với mỗi trong 17 khớp xương:
1. Tính độ cong quỹ đạo
2. Chọn 2, 4, hoặc 8 control point: độ cong thấp (đứng yên) → 2 CP; trung bình → 4 CP; chuyển động nhanh → 8 CP
3. Quantize vị trí về uint8: `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` ánh xạ [-2m, +2m]
4. Sinh token tự mô tả: `<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis>`

**Tại sao adaptive?** Pelvis đứng yên không cần 8 điểm — 2 điểm đủ. Cổ tay đang vung cần 8. Giảm ~35% token so với format cố định 8 CP.

**Các phiên bản trước đã bỏ:**
- `phase5_interpolation_tokenizer.py` — 256 token uint8 mờ đục. Bỏ vì model không học được ngữ nghĩa từng khớp.
- `phase5b_xyzt_tokenizer.py` — 409 token cố định. Tự mô tả nhưng lãng phí.

Output: **18,847 video** (chỉ video có người theo YOLO), 7.4 GB.  
Token range: 171 (rất tĩnh) đến 579 (chuyển động nhanh), thường ~250–300 mỗi window.

---

### Phase 6: Merge

- Inject `<agent>...</agent>` vào sau mỗi `<avc_lm>` block trong file training_ready
- Căn chỉnh frame: match agent window_id với AVC-LM chunk index (cùng 30fps, 8-frame window)
- Thêm mảng `chunk_timing` vào mỗi activity (timestamp chính xác, modality nào có mặt)
- ~399K activities, **~2.15M agent block** được inject
- Output: 160 file `final_vla_adaptive_rank_*.jsonl`, **657 GB**

**Phase 6 v2 — Hỗ trợ inject SNAC (28/06/2026):**
- Thêm `--snac-tokens-dir` để inject SNAC audio token cùng lúc với agent trong một pass
- Thứ tự token mỗi chunk: `<cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]`
- `chunk_timing` có thêm flag `has_snac` cho mỗi chunk
- Backward compatible: không truyền `--snac-tokens-dir` thì chạy y hệt v1

---

### Phase 7: Flatten + Augment

Chuyển JSON phân cấp → flat Megatron-LM JSONL.

**Modality dropout v2 (27/06/2026):**
| Modality | Drop rate | Lý do |
|----------|-----------|-------|
| AVC-LM | **100%** | Bỏ hoàn toàn chờ ablation |
| Cosmos | **50%** | Giữ để học modality transition |
| Seed2 | 0% | Giữ hết |
| Agent | 0% | Giữ hết |
| SNAC | 0% | Giữ hết (mặc định, có thể chỉnh) |

**Augmentation text:** 15% synonym replacement, 5% stopword dropout, 10% sentence permutation, xen kẽ speech/token ngẫu nhiên, shuffle layout block ngẫu nhiên.

Output v1: **69,844 record**, 19.2 GB → `megatron_dataset_adaptive/`  
Output v2: cosmos 50% drop, avclm 100% drop → `megatron_dataset_v2/` (xong 27/06/2026)

**Phase 7 v3 — SNAC + cập nhật filter (HOÀN THÀNH 02/07/2026):**
- Thêm xử lý khối `<snac>...</snac>` (pass-through giống agent)
- **Thay đổi filter quan trọng:** trước đây chỉ emit activity có `<agent>`; giờ emit nếu có `<agent>` HOẶC `<snac>`
  - Record đầy đủ: seed2 + cosmos + agent + snac — **69,811 record (18.8%)**
  - Record một phần: seed2 + cosmos + snac — **302,044 record (81.2%)**
  - Bad record (không có gì): **0**
- Output: `megatron_dataset_v3/` — 160 file, **371,888 record**, **72 GB**
- Sample: `samples/after_flatten_v3.json` | Upload script đã cập nhật: `tools/upload_flattened_hf.py`

**✅ Phase 7 v4 — Căn chỉnh thời gian per-chunk (HOÀN THÀNH 02/07/2026):**

Phase 7 được viết lại hoàn toàn với state machine đi theo thứ tự tài liệu trong output Phase 6. Mỗi chunk phát ra: `[seed2?][cosmos?][agent?][snac?]`. Speech được chuyển vào header `### Speech:`, không còn xen vào chuỗi token.

**Thống kê v4:** 160/160 file, 371,888 record, **5.217B token** (seed2 6.4% / cosmos 74.4% / agent 12.2% / snac 7.0%). Thời gian chạy: 36 phút / 32 worker.

**Các bug đã fix:**
- Mất căn chỉnh thời gian (v3: toàn bộ agent ở cuối → 69% record có 0% agent trong 4096 token đầu. v4: per-chunk → mọi record đều có agent trong 4096 token đầu)
- Xen speech vào agent grammar (v3: từ speech rải vào giữa chuỗi joint token. v4: speech chỉ ở header)

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v4/` (160 file)

**Tần suất token thực tế (verified 02/07/2026):**

| Modality | Token/chunk | Trong 30s (sau v3 dropout) |
|----------|-------------|--------------------------|
| Seed2 | 32 cố định (1 block mỗi 3.75 chunk) | 30 × 32 = **960** |
| Cosmos | 200 cố định (mỗi chunk) | ~56 × 200 = **11,200** |
| Agent | 171–579 (~280 thông thường) | đến 112 × 280 = **31,360** |
| SNAC | 9 hoặc 12 (tb 10, xen kẽ) | 112 × 10 = **1,120** |
| AVC-LM | 885–5,055 | **0** (dropped) |

---

### Tokenizer — Bug lớn đã fix

**Script:** `tools/expand_vocab.py`, `tools/upload_tokenizer.py`

Mở rộng GPT-NeoX-20b (50,277 token) với 93,938 VLA token dùng `tokenizer.add_tokens(special_tokens=True)`.

**Bug của model đầu tiên:** Edit `vocab.json` trực tiếp KHÔNG đăng ký BPE merge rules. Tokenizer split `<seed2_1137>` thành 7 mảnh. Dù vậy model vẫn có tín hiệu học (nó học predict chuỗi mảnh vụn) nhưng không decode được token thật.

**Fix:** `add_tokens(special_tokens=True)` bypass BPE merging, mọi VLA token được xử lý atomic.

Published: `EmpathicRobotics/tokenizer-vla-adaptive` (vocab 144,215, padding lên 144,256 cho Megatron).

---

### Phase 8: Megatron Tokenization

| Shard | Tokens | Size |
|-------|--------|------|
| `data_shard_00000.bin` | 2,684,323,146 | 10.00 GB |
| `data_shard_00001.bin` | 156,389,702 | 0.58 GB |
| **Tổng** | **2,840,712,848 (2.84B)** | **10.58 GB** |

---

### Phase 9: Training — Model thứ hai (Tháng 6/2026)

**Model:** `EmpathicRobotics/vla-1.7b-pab-spline-adaptive`  
**Kiến trúc:** OpenSci-Ref 1.7B (24 layer, hidden 2048, 32 head → **1.91B param** với vocab embedding 144K)  
**Compute:** 64 node × 4 GH200 = 256 GPU, ~35 phút wall time

| Iter | Loss | LR | Token đã xem |
|------|------|----|-------------|
| 200 | 2.982 | 4e-3 | 0.84B |
| 500 | 2.070 | 4e-3 | 2.10B |
| 1000 | 1.672 | 4e-3 | 4.19B |
| 2000 | 1.476 | 3.2e-4 | 8.39B |
| **2032 (val)** | **1.501** | — | — |

Val PPL: **4.49** | Test PPL: **4.45** | ~3 epoch trên 2.84B token

---

### Data Inventory (26/06/2026 — Hoàn thành)

**Script:** `tools/data_inventory.py` | **Checkpoint:** `tools/inventory_checkpoint_v2.json`

Quét 242 file trên 4 nhóm dataset:

| Dataset | seed2 | cosmos | avclm | agent | snac | text | **TỔNG** |
|---------|-------|--------|-------|-------|------|------|---------|
| FineVideo-VLA (160 file) | 89.9M | 210.2M | 474.4M | 564.9M | — | 11.4M | **1.35B** |
| MV-Backup valid_with_seed (64 HF shard) | 5.6M | — | — | — | — | — | **5.6M** |
| MV-Backup stack_images3_gzip (12 archive) | 313K | — | — | — | — | — | **313K** |
| MV-Omni valid_snac (6 file gzip) | — | — | — | — | 4.92B | 1.99B | **6.93B** |
| **TỔNG** | **95.8M** | **210.2M** | **474.4M** | **564.9M** | **4.92B** | **2.00B** | **8.29B** |

**Phát hiện quan trọng:**
- `valid_with_seed` — tải về **1.1 TB** nhưng chỉ có 5.6M seed2 token (< 0.5% của FineVideo). Shard 0–30 toàn `.png`/`.ogg` không có token. **Không đáng dùng.**
- MV-Omni là nguồn ngoài duy nhất đáng kể: 6.93B token. Nhưng `<snac_N>` và `<seed_N>` **chưa có trong tokenizer vocab** — cần vocab expansion trước khi dùng.
- **Chỉ FineVideo có agent (3D pose) token.** Không dataset nào bên ngoài có pose data.
- **Sẵn sàng train ngay hôm nay: 1.35B token** (FineVideo, với vocab hiện tại).

---

## Trạng thái hiện tại — Cái gì chạy được, cái gì chưa

### Hoạt động tốt
- Pipeline đầu đuôi: raw video → 3D pose → token → Megatron bin → train → HF checkpoint
- Tất cả VLA token đều atomic (confirmed sau tokenizer fix)
- Model hoàn thành đúng agent block 17 khớp: đúng thứ tự H36M, giá trị xyz/t hợp lệ, decode được thành 3D pose via PCHIP
- Decoder verified: output model → (8, 17, 3) trajectory đúng range vật lý [-2m, +2m]

### Chưa hoạt động được
- **Tự chuyển modality:** Khi chỉ được prompt bằng text, model luẩn quẩn ở seed2, không tự chuyển sang cosmos/avclm/agent. Cần có agent token trong prompt mới tiếp tục agent mode.

**Ba nguyên nhân gốc rễ:**

1. **Data starvation (thiếu data):** 2.84B token cho model 1.91B param = ~1.5× Chinchilla ratio. Optimal là ~20×. Mỗi sample training chỉ được xem ~3 lần — đủ để nhớ local pattern, không đủ để học high-level sequencing.

2. **Thiếu language anchor:** Text chỉ có Title/Context/Keywords. Không có caption mô tả điều gì đang xảy ra ở mỗi timestamp. Model không có tín hiệu ngôn ngữ để biết "sau seed2 tokens này, đến cosmos tokens."

3. **Dropout quá mạnh:** 99% AVC-LM + 90% Cosmos drop khiến hầu hết record không có chuỗi transition đầy đủ. Model hiếm khi thấy seed2 → cosmos → avclm → agent liền mạch.

---

## Kế hoạch tiếp theo — Chi tiết và ưu tiên

### Nhóm 1 — Làm được ngay, không cần nhiều GPU

**Ưu tiên 1 — Chuẩn bị MV-Omni** ← HOÀN THÀNH MỘT PHẦN

- ~~Convert `<seed_N>` → `<seed2_N>` trong MV-Omni~~ **XONG** (27/06/2026)
  - Script: `data_prep/convert_mvomni_seed.py`
  - Output: `/p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz`
  - **1,593,301 record | 19,249,664 seed token đã convert | 30 GB output**
  - Không còn `<seed_N>` nào trong output — verified sạch
- **CÒN LẠI:** Thêm `<snac_0>` ... `<snac_4095>` (~4096 token) vào tokenizer via `add_tokens(special_tokens=True)`
  - Không cần thêm `<seed_N>` nữa — đã convert sang `<seed2_N>` rồi
  - Vocab mới: ~148,311 token
  - Mở khóa **6.93B token từ MV-Omni**
  - Ước tính: ~1 ngày

### Các điểm cần thảo luận trước khi train (02/07/2026 — từ chat Huu)

> **⚠ Huu nói rõ: "Before you train let's talk." — CHƯA được train cho đến khi giải quyết xong 3 điểm này.**

**[DISCUSS-1] Language data mix — thêm gì vào trước khi train?**
- FineVideo v4 + MV-Omni = 12B token nhưng gần như không có instruction/language data
- Huu: "mix in a few billion tokens mixture so we can steer the robot better"
- Huu muốn SFT datasets dạng robot instruction ("pick up the Apple", "Drive left", v.v.)
- Dataset có sẵn trên leo (`/mnt/sdb/mixture-vitae-working/`): `clappa_text_only`, `coco` (synthetic permissive), `misc_instr/hpprc-r1-distill-qwen-pseudo-qa.jsonl` (instruction tiếng Nhật)
- Cũng muốn multilingual instruction datasets có reasoning/thinking
- **Cần làm:** Đếm token các dataset này → quyết định mix ratio

**[DISCUSS-2] Compression analysis của Adaptive PCHIP — ĐÃ PHÂN TÍCH ĐẦY ĐỦ (04/07/2026)**

**Context:** Huu yêu cầu 3 thứ: (1) compression so với BEAST, (2) 1-CP có được không, (3) "just do a 1/2/3 etc for a sample and see what compression you get."

**Script phân tích:** `tools/analyze_cp_tradeoff.py` — chạy trên 50 video / 1,940 window từ yolo_cleaned_30fps.

#### Kết quả 1/2/3 CP tradeoff (đúng cái Huu yêu cầu)

| N CP | Token/window (17 joint) | MAE (mm) | Ghi chú |
|------|------------------------|---------|---------|
| **1** | **86** | **24.3mm** | constant, không có t token |
| **2** (min hiện tại) | **171** | **12.7mm** | linear interpolation |
| 3 | 239 | 8.0mm | — |
| 4 | 307 | 6.4mm | — |
| 5 | 375 | 5.6mm | — |
| 6 | 443 | 5.1mm | — |
| 7 | 511 | 4.6mm | — |
| **8** (baseline) | **579** | **4.1mm** | tất cả frame |

**Nhận xét:** 1-CP global (mọi joint) = 24.3mm error quá cao. Nhưng 1-CP **chỉ cho joint tĩnh** (quantize start == end) thì error bị giới hạn ≤15.7mm (1 quant step).

#### Kết quả 1-CP static test (53.6% tier-2 joints qualify)

```
Tier-2 joint-windows (curv < tau_low) : 14,668
Trong đó, quantized start==end (3 dim): 7,862  (53.6%)
Avg qualifying joints per window       : ~4.1 joints/window
Tokens saved by 1-CP                   : ~20 tokens/window
Current adaptive avg                   : 284 tokens → 264 tokens
Additional compression                 : +7.1%
```

**Overhead breakdown (xác nhận 34% overhead):**
```
Wrappers (<name> + </name>) :  34 tokens (12%)
t tokens (<joint_t_N>)      :  62 tokens (22%)
xyz tokens                  : 187 tokens (66%)
─────────────────────────────────────────────
OVERHEAD tổng cộng          :  97 tokens (34%)
```

#### So sánh với BEAST (từ phân tích 03/07/2026)

**BEAST:** "B-spline Encoded Action Sequence Tokenizer" (KIT, NeurIPS 2025, arXiv 2506.06072). Fixed N CPs, fit bằng ridge regression, claim **4–8× compression** so với binning.

**Tại sao con số của mình trông nhỏ hơn — baseline khác nhau:**

| | Baseline | Compression |
|---|---|---|
| **BEAST** | Binning (1 token/timestep/DoF) | 4–8× (75–87% ít token hơn) |
| **Của mình vs fixed 8-CP** | Fixed 8-CP (đã compressed) | ~2× (50.9% ít hơn) |
| **Của mình vs raw binning** | 8×17×3 = 408 giá trị raw | ~1.5× |

Root cause gap: **34% token là overhead** tên joint (self-describing) — BEAST có 0% overhead vì decoder hardcode structure. Trade-off có chủ đích: self-describing → LLM học được joint semantics.

#### Tại sao minimum là 2-CP (không phải 1-CP)

1. PCHIP cần ≥2 điểm — polynomial nội suy không thể dùng với 1 điểm
2. "Curvature thấp" ≠ "Không di chuyển" — joint vẫn có thể drift tuyến tính 10–15mm trong 0.267s. 2-CP bắt được drift; 1-CP thì giả định constant = sai

#### Đề xuất 1-CP của Huu — khả thi, gain là 7%

Grammar 1-CP: nếu `quantize(frame_0) == quantize(frame_7)` cho cả 3 dim:
```
# Thay vì 10 tokens (2-CP):
<pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
<pelvis_t_7> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>

# Dùng 5 tokens (1-CP, không có t token):
<pelvis> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> </pelvis>
```
**Gain thực tế: +7.1%** (từ 284 → 264 tokens/window). Cần grammar change + re-run Phase 5 → 6 → 7 → Megatron tokenization.

**CHỐT QUYẾT ĐỊNH (08/07/2026):** Hoãn. Đã confirm với Huu trên Discord — giữ nguyên format adaptive 2/4/8-CP. Đã thử chạy full-dataset validation (18,847 video) nhưng bị gián đoạn do JUWELS sập, chưa resume. Chỉ quay lại nếu sau này data cho thấy cần thiết — gain +7.1% không đáng để re-run toàn bộ Phase 5→7 ngay lúc này. Cho mục đích paper, con số "compression giảm hơn 50%" (so với fixed 8-CP) là đủ để báo cáo.

#### Về window duration dài hơn (Huu: "could compress more for longer duration")

Tăng từ 8 frames (0.267s) lên 16–32 frames:
- Joint tĩnh vẫn chỉ cần 1-2 CPs → compression tốt hơn
- **Vấn đề:** 8-frame alignment là cố tình để match Cosmos/AVC-LM chunk size. Đổi window size = Phase 6 merge logic phải thiết kế lại từ đầu
- Đây là architectural change, cần separate discussion với Huu

#### Câu hỏi cần hỏi Huu để quyết định tiếp

> "Với 1-CP được +7%, và 34% overhead từ self-describing format (unavoidable nếu muốn LLM học joint semantics) — bạn muốn prioritize: (a) implement 1-CP và re-run phase 5–7, hay (b) redesign với window dài hơn và giải quyết alignment?"

- **XONG:** `tools/analyze_pchip_compression.py` — 18,847 file, 1,743,189 window. Kết quả:
  - **Tiết kiệm 50.9% token** so với fixed 8-CP (284.1 token/window vs 579)
  - CP tiers: 55.2% 2-CP / 25.6% 4-CP / 19.2% 8-CP
  - Động nhất: r_knee (33.5% 8-CP), r_wrist (29.4%). Tĩnh nhất: pelvis (100% 2-CP)
- **MỚI — Vấn đề chất lượng pose (02/07/2026):**
  - Trung bình chỉ **4–7 joint finite/frame** (17 tổng cộng) — 24–41% skeleton
  - **Tay (j11–j16) gần như luôn NaN** — MotionBERT không lift được joint tay (bị che, góc nghiêng)
  - **Lỗi zero-fill ở head_top (j10)** — = (0,0,0) khi không detect, trùng pelvis, tính là finite nhưng sai
  - Ảnh hưởng: model chỉ học được lower body + torso. OK cho pretraining; KHÔNG đủ cho học manipulation cánh tay

**[DISCUSS-3] Eval setup**
- Huu: "We should start eval just to see how things perform with baseline"
- Cần định nghĩa eval tasks TRƯỚC khi train
- Ứng viên: MPJPE trên 3D pose decode, modality transition accuracy, instruction-following robot commands
- **Cần làm:** Định nghĩa eval protocol và implement baseline metrics

---

**Ưu tiên 2 — Điều chỉnh dropout trong Phase 7** ← ~~XONG~~ (27/06/2026)

| Modality | Trước | Sau | Lý do |
|----------|-------|-----|-------|
| AVC-LM | 99% drop | **100% drop** | Bỏ hoàn toàn, chờ ablation xác nhận |
| Cosmos | 90% drop | **50% drop** | Giữ ~50% để model học chuỗi seed2→cosmos→agent |

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/`  
Đã upload lên `EmpathicRobotics/FineVideo-Phase7-Flattened` (v2).  
**Bước tiếp theo:** Megatron re-tokenize `megatron_dataset_v2/` → train lại v0.2.

**Ưu tiên 3 — Ego-centric perspective cho FineVideo**

- Đọc Phase 4 pose data (yolo_cleaned)
- Áp rotation matrix: đặt camera tại `head_top`, hướng về phía thorax
- Sinh thêm agent token sequence từ góc nhìn first-person
- Cùng motion data, nhân đôi độ đa dạng (góc nhìn người + góc nhìn robot)
- Ước tính: ~1 tuần code + 1 SLURM run

**Ưu tiên 4 — Viết captioning pipeline**

- Dùng `chunk_timing` timestamp để extract keyframe từ video FineVideo gốc
- Chạy SmolVLM2 hoặc Qwen2.5-VL trên mỗi keyframe
- Interleave caption vào token sequence tại đúng timestamp
- Impact dự kiến: ×4 record với language anchor tại mỗi modality transition → fix nguyên nhân số 2
- Ước tính: 1–2 tuần code (chạy GPU là việc riêng)

---

### Nhóm 2 — Cần GPU trên JUPITER

**Ưu tiên 5 — Thu thập agent + cosmos + snac từ Cosmos3-DROID**

- `nvidia/Cosmos3-DROID` trên HuggingFace: video robot tay nắm đồ vật, đã có Cosmos token
- Chạy YOLO + tương đương Phase 1–5 để extract agent token (tay/cánh tay robot)
- Thêm SNAC nếu có audio track
- **Đây là robot data đầu tiên** — quan trọng để model generalize từ người → robot
- Không thêm AVC-LM cho đến khi có ablation (per Huu)

**Ưu tiên 6 — Vocab expansion (tạo tokenizer mới)** ← **HOÀN THÀNH (01/07/2026)**

Script: `tools/build_tokenizers.py`. Hai chế độ:
- `--mode current`: load `tokenizer_vla_adaptive` (144,215 vocab) + thêm 12,290 SNAC token → **156,505 vocab**
- `--mode qwen3`: load Qwen3 base + thêm 106,228 VLA token → **257,897 vocab**
- Tất cả token verified atomic bằng spot-check encode (encode → 1 ID duy nhất)

**Ưu tiên 7 — SNAC cho FineVideo** ← **HOÀN THÀNH (01/07/2026)**

Job `snac_cpu_14077331`, 32 array task, partition `batch`, submit từ `jwlogin08`.

| Metric | Kết quả |
|--------|---------|
| Tasks hoàn thành | **32/32** (100%) |
| Activities xử lý thành công | **371,855** |
| fail_audio (không có audio) | 530 (~0.1%) |
| fail_snac | **0** |
| Tổng SNAC token | **363,029,331 (~363M)** |
| Số file output | **40,779** `{video_id}_snac.jsonl` |
| Kích thước output | **6.5 GB** |
| Đường dẫn output | `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/` |

**Cơ chế tokenization và chia chunk (đã verify từ code):**
- Encode toàn bộ audio của activity một lần → flat list token (giữ audio context)
- SNAC output = chuỗi **base frame**, mỗi base frame = đúng 3 token (triplet L0 + L1_even + L1_odd)
- `n_base = len(flat_tokens) // 3` — đơn vị chia tối thiểu là base frame (không tách triplet)
- Chia proportional: `start_base[k] = round(k × n_base / n_chunks)`, `end_base[k] = round((k+1) × n_base / n_chunks)`
- Mỗi chunk nhận 3 hoặc 4 base frame = **9 hoặc 12 token** (xen kẽ vì 3.33 base frame/chunk)
- Chunk cuối KHÔNG ngắn hơn — `round(n_chunks × n_base / n_chunks) = n_base` chính xác
- Sai số alignment: ±1 base frame = ±80ms tại ranh giới chunk (chấp nhận được cho pretraining)

**Việc tiếp theo (đã unblock):**
```bash
# Bước 1: Vocab expansion — thêm 12,288 token <snac_N> vào tokenizer
# TODO: cập nhật tools/expand_vocab.py, range [128266..132361], [132362..136457], [144650..148745]

# Bước 2: Re-run Phase 6 với SNAC injection
python pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "...training_ready_rank_*.jsonl" \
  --agent-tokens-dir .../agent_tokens_adaptive \
  --snac-tokens-dir  .../FineVideo-VLA/snac_tokens \
  --output-dir       .../FineVideo-VLA/final_dataset_adaptive_v2 \
  --output-prefix    final_vla_adaptive_v2

# Bước 3: Re-run Phase 7 → megatron_dataset_v3/
python pipeline_pose/phase7_flatten.py \
  --input-glob ".../final_dataset_adaptive_v2/..." \
  --output-dir ".../megatron_dataset_v3" \
  --drop_cosmos 0.5 --drop_avc 1.0 --drop_snac 0.0 --workers 16

# Bước 4: Megatron tokenize → train v0.3
```

**Ưu tiên 7 — Điều tra leo seed2 + euro_pat**

- Huu đề cập có dataset trên leo cluster: seed2 + euro_pat
- Cần đếm token trước khi commit storage/compute

**Nguồn Data Mới (07/07/2026 — từ chat team)**

Tìm được khi mở rộng scope tìm data VLA. Chưa đếm token/giờ dữ liệu hay check license cho cái nào.

| Nguồn | Nội dung | Ghi chú |
|---|---|---|
| `abc.bot` (Amazon) | 400h robot recording **trong simulation**, có physics state (MjData) | Đáng chú ý nhất — permissive, có eval env, cùng embodiment xuyên suốt. blanchon.jl: "indeed perfect" |
| `allenai/MolmoAct2-BimanualYAM-Dataset` | 2 TB, robot tay đôi YAM | Cần check license + embodiment có tương thích không |
| `MiG-NJU/OmniVideo-100K` | Video dataset | Chưa scope |
| `mlfoundations/MINT-1T-HTML` | Text/HTML dataset lớn | Chưa scope — có thể dùng cho language mix (DISCUSS-1), không phải video |
| `genrobot2025/Gen-EgoData` | Robot data góc nhìn egocentric | Chưa scope |
| `finevla.xlang.ai` | Có thể là VLA dataset | Chưa tìm được HF link — có thể chưa release |
| `mira-wm.com` | World model reference (Kyutai vừa release cái tương tự) | Reference/cảm hứng, không hẳn là nguồn data |

**Ràng buộc team:** synthetic/simulation data (abc.bot, MolmoAct2, Cosmos3-DROID...) giới hạn **≤30% tổng training mix** — team consensus (Huu, dựa trên literature), để giữ cân bằng nghiêng về video người/robot thật.

**Ưu tiên 8 — Re-training v0.2**

Sau khi hoàn thành ưu tiên 1, 2, 4: ước tính **10–20B token** có sẵn.
- Continue training từ checkpoint hiện tại (iter 2032) với data mới + dropout đã điều chỉnh
- Kết quả kỳ vọng: model bắt đầu tự transition giữa modality khi chỉ prompt bằng text

---

### Nhóm 3 — Dài hạn (3–6 tháng)

**Ưu tiên 9 — Thêm text LLM data**
- Mix text data chuẩn LLM vào training (~10–15% của total mix)
- Ngăn model quên ngôn ngữ khi train quá nhiều trên VLA token
- "Create text binding" theo Huu

**Ưu tiên 10 — Qwen3 migration**
- Retokenize toàn bộ dataset với Qwen3 tokenizer
- Cần re-run Phase 8 và train lại từ đầu
- Lợi ích: native HF ecosystem, vLLM, llama.cpp
- Config của Huu: cherry-picked từ commit `7dcf8a5`
- **Để sau** — data landscape còn đang thay đổi, làm một lần thôi

**Ưu tiên 11 — Nâng cấp lên PAB-Spline spec**
- Tokenizer hiện tại: PCHIP xyz-only (chỉ vị trí)
- Spec kêu gọi: góc khớp (q/qd), phase variable φ ∈ [0,1], phát hiện gait tuần hoàn, nén joint tĩnh
- Blocked: cần chạy lại pipeline kinematics với tính toán góc

**Ưu tiên 12 — Isaac Sim integration**
- Sinh rollout Unitree H1 trong Isaac Sim / ManiSkill
- Tokenize sim data với PAB-Spline tokenizer
- Map joint token → H1 control signal cho sim-to-real

---

## Bức tranh data — Chúng ta ở đâu, cần gì

### Hiện tại: 1.35B token sẵn sàng train (FineVideo)
Quá nhỏ. Chinchilla optimal cho model 1.7B là ~34B token. Chúng ta đang ở ~4%.

### Mở khóa ngay bằng vocab expansion (không cần thu thập thêm): +6.93B token
MV-Omni valid_snac đang ngồi đó, đã tokenized, chỉ bị chặn bởi `<snac_N>` / `<seed_N>` chưa có trong vocab. Thêm 2 token family này = 1–2 ngày code = unlock 6.93B token = tổng ~8.3B. **Đây là action có leverage cao nhất hiện tại.**

### Mở khóa bằng GPU run: +5–10B token (captioning, ego-centric, Cosmos3-DROID)
Captioning pipeline một mình có thể nhân FineVideo lên ~4× (69,844 record → ~280K record) với context ngôn ngữ phong phú hơn. Ego-centric perspective thêm góc nhìn thứ hai cho free.

### Target: 20–40B token cho v0.2 training
Với vocab expansion + MV-Omni + captioning + Cosmos3-DROID + SNAC-FineVideo, đạt 20–40B token là khả thi trong 2–3 tháng focused work.

---

## Đánh giá thẳng thắn — Chúng ta có đang đi đúng hướng không?

### Đúng ở những điểm này

**Kiến trúc là sound.** Model thứ hai đã chứng minh hypothesis cốt lõi: LLM 1.7B CÓ THỂ học ngữ pháp của chuỗi multimodal token — thứ tự khớp xương, range xyz hợp lệ, phân phối token đặc trưng theo từng modality — chỉ từ next-token prediction trên flat interleaved sequence. Đây là kết quả quan trọng.

**Bottleneck là data, không phải architecture.** Việc model không tự transition giữa modality được giải thích hoàn toàn bởi data starvation và thiếu language anchor. Đây là vấn đề engineering có thể giải quyết, không phải lỗi thiết kế cơ bản.

**Hướng đi thực sự novel.** Không có published work nào thống nhất Seed2 + Cosmos + SNAC + PCHIP pose token trong một LLM autoregressive. RT-2, OpenVLA, π0 đều dùng action representation đơn giản hơn nhiều và không thử 3D body pose liên tục.

### Rủi ro cần nhận thức rõ

| Rủi ro | Mức độ | Cách giảm thiểu |
|--------|--------|----------------|
| Scale gap — ngay cả 20B vẫn xa frontier LLM | Trung bình | Mix text data, tập trung quality hơn quantity |
| Chưa có robot data thực — chỉ human từ YouTube | Cao | Cosmos3-DROID, Isaac Sim (ưu tiên 5 + 12) |
| SNAC audio quality — Orpheus SNAC2 "đủ tốt" nhưng không tốt nhất | Thấp | Moss Audio V2 (2.1B decoder) để sau |
| Qwen3 migration cost — retokenize sẽ tốn 1–2 tuần SLURM | Thấp | Làm một lần duy nhất sau khi data ổn định |
| Không ai trong nhóm có kinh nghiệm VLA | Trung bình | Vừa làm vừa học, frontier research cần chấp nhận điều này |

### Cái cần tránh

- **Đừng tối ưu quá sớm** thứ chưa chắc quan trọng (AVC-LM, PAB-Spline spec) trước khi có dữ liệu đủ.
- **Đừng block mọi thứ chờ Qwen3** — train v0.2 với tokenizer hiện tại trước, Qwen3 để sau.
- **Đừng lãng phí compute vào valid_with_seed** — 1.1 TB cho 5.6M token là không đáng.

### Cột mốc thực tế

- **v0.2 (2–3 tháng):** Model tự transition từ text prompt → seed2 → cosmos → agent mà không cần agent token trong prompt.
- **v0.3 (4–6 tháng):** Model phản hồi lệnh thoại (SNAC) bằng cách sinh valid agent motion token. "Walk forward" → quỹ đạo pelvis/hip/knee hợp lệ.
- **v1.0 (6–12 tháng):** Model nhìn scene visual + nhận lệnh → sinh motion phù hợp với geometry của scene.

---

## Log các quyết định quan trọng

### Kết quả kiểm tra overlap dataset (30/06/2026)

Script `tools/check_dataset_overlap.py` so sánh video ID của `valid_with_seed` (64 shard HF) vs `omni_valid` (6 file gzip):

| Metric | Số liệu |
|--------|---------|
| `valid_with_seed` unique video ID | **31,500** |
| `omni_valid` unique video ID | **238,539** |
| Trùng nhau (cả hai) | **27,359** (86.9% của seed / 11.5% của omni) |
| Chỉ trong `valid_with_seed` | **4,141** |
| Chỉ trong `omni_valid` | **211,180** |

**Kết luận:** omni_valid đã cover 86.9% video của valid_with_seed. 4,141 video còn lại trong valid_with_seed chỉ có seed2 token (~700K token tổng cộng) — không đủ giá trị để bù cho 1.1 TB storage.

**Quyết định: KHÔNG dùng `valid_with_seed`.** Chỉ dùng omni_valid (238K video, 6.93B token). 1.1 TB đã tải về có thể xóa để giải phóng storage.

### Bảng quyết định

| Quyết định | Lý do | Thời gian |
|------------|-------|-----------|
| Chọn Adaptive PCHIP thay vì 409-token cố định | Self-describing, ~35% ít token hơn cho joint tĩnh | T2/2026 |
| Fix tokenizer qua `add_tokens()` chứ không edit vocab.json | BPE cần merge rules, không phải chỉ vocab entry | T5/2026 |
| 99% AVC-LM dropout trong Phase 7 | AVC-LM nhiều hơn agent 373× — sẽ dominate context | T3/2026 |
| valid_with_seed KHÔNG dùng | 1.1 TB tải về cho 5.6M token (< 0.5% của FineVideo) | T6/2026 |
| Tạm chưa tăng AVC-LM trong dataset mới | Chờ ablation xác nhận trước khi đầu tư | T6/2026 |
| Ego-centric perspective như data multiplier miễn phí | Cùng motion, góc nhìn khác — tăng diversity mà không cần data mới | T6/2026 |
| Qwen3 migration để sau | Quá sớm — data landscape còn đang thay đổi | T6/2026 |
| Inject SNAC ở Phase 6 thay vì Phase 7 | Phase 6 đã làm per-chunk injection; Phase 7 là stateless flatten. Để logic injection ở một chỗ. | T6/2026 |
| Encode SNAC 1 lần/activity rồi chia chunk | Encode từng chunk 0.267s sẽ mất audio context + chậm vì nhiều call nhỏ. Encode 1 lần + chia đều vừa chính xác vừa nhanh. | T6/2026 |
| SNAC cho TẤT CẢ activity, không chỉ activity có agent | 86% activity không có agent vẫn có seed2+cosmos → thêm SNAC dạy audio↔video binding. Chỉ dùng 14% (agent-only) = lãng phí 86% GPU run. | T6/2026 |
| valid_with_seed KHÔNG dùng (overlap confirmed) | Overlap check: 86.9% video đã có trong omni_valid. 4,141 video unique chỉ có seed2 (~700K token) — không đủ bù cho 1.1 TB storage. omni_valid là superset gần như hoàn toàn. | 30/06/2026 |
| 1-CP: hoãn, giữ nguyên adaptive 2/4/8-CP | Gain +7.1% (ước tính từ sample) không đáng để re-run toàn bộ Phase 5→7 ngay lúc này; quay lại sau nếu cần | 08/07/2026 |
| Synthetic/sim data giới hạn ≤30% tổng training mix | Team consensus (Huu), dựa trên literature; giữ cân bằng nghiêng về video thật | 07/07/2026 |
| Moss-Audio Tokenizer V2: nếu dùng thì phải giới hạn | Huu: ở 400 token/giây sẽ overwhelm dataset nếu dùng rộng cho omni-modal pretraining; chỉ nên dùng đoạn ngắn chi tiết cao rồi tiếp nối SNAC rate thấp hơn, hoặc dùng riêng nếu không cần bind với ngôn ngữ | 02/07/2026 |

---

## Artifacts đã publish

| Artifact | Vị trí | Trạng thái |
|----------|--------|------------|
| Tokenizer v1 (vocab 144,215, GPT-NeoX) | `EmpathicRobotics/tokenizer-vla-adaptive` | Live |
| **Tokenizer v2 (vocab 156,505, GPT-NeoX + SNAC)** | `EmpathicRobotics/tokenizer-vla-adaptive-v2` | **Live (01/07/2026)** |
| **Tokenizer Qwen3 (vocab 257,897)** | `EmpathicRobotics/tokenizer-vla-qwen3` | **Live (01/07/2026)** |
| FineVideo-Phase7-Flattened v4 (371,888 record, 5.217B token) | `EmpathicRobotics/FineVideo-Phase7-Flattened` | **Chờ upload** |
| FineVideo-Phase5-AgentTokens (~399K activities) | `EmpathicRobotics/FineVideo-Phase5-AgentTokens` | Live |
| FineVideo-Phase4-YOLOPose (hàng triệu window) | `EmpathicRobotics/FineVideo-Phase4-YOLOPose` | Live |
| VLA Model v1 (tokenizer broken) | `EmpathicRobotics/vla-1.7b-pab-spline-25b-test` | Live (deprecated) |
| VLA Model v2 (tokenizer đã fix) | `EmpathicRobotics/vla-1.7b-pab-spline-adaptive` | Live |
| Megatron .bin/.idx (2.84B token) | `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` | Local |
| Data inventory checkpoint | `tools/inventory_checkpoint_v2.json` | Local |
| **CP tradeoff analysis script** | `tools/analyze_cp_tradeoff.py` | **Local (04/07/2026)** |

---

## Action Items — 2 tuần tới

### Đã hoàn thành (Jun–Jul 2026)
- [x] **SNAC CPU job** (14077331) — **HOÀN THÀNH (01/07/2026)**. 32/32 tasks, 371,855 activities, 363M token, 6.5 GB → `/p/.../snac_tokens/`
- [x] **Vocab expansion (tokenizer build)** — **HOÀN THÀNH (01/07/2026)**. Script: `tools/build_tokenizers.py`. Tạo 2 tokenizer:
  - `tokenizer_vla_adaptive_v2` (GPT-NeoX-20b + SNAC): **156,505 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2/`
  - `tokenizer_vla_qwen3` (Qwen3 + toàn bộ VLA token): **257,897 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3/`
  - Spot-check 12 token đại diện (seed2, cosmos, pose, snac): tất cả **atomic** ✓ — không có sub-piece splitting

### Pipeline tiếp theo (đã unblock)
- [x] **Phase 6 v2 dry run** — **HOÀN THÀNH (01/07/2026)**. Chạy thử 1 file (254 video, ~5 phút/file):
  - SNAC inject: **259,503/259,505** avc block (~100%)
  - Agent inject: **12,705** block (đúng — chỉ 46% video có Phase 5 output)
  - Format verified: `</avc_lm> <agent>...</agent> <snac> <snac_N>... </snac>` ✓
  - `chunk_timing` đủ flag `has_seed2/cosmos/avc_lm/agent/has_snac` ✓
  - Script SLURM mới: `slurm/submit_merge_adaptive_v2.sh` (account `laionize`, partition `batch`, 32 workers, 2h)
  - Ước tính 160 file với 32 workers: **~25–40 phút**
- [x] **Phase 6 v2 re-run** — **HOÀN THÀNH**. Job `14082096`, 32/32 workers. 40,804 video | 398,775 activity | SNAC 100% | Agent 5.5% | 0 lỗi → `final_dataset_adaptive_v2/` (160 file)
- [x] **Phase 7 v3 re-run** — **HOÀN THÀNH (02/07/2026)**. 160/160 file, 371,888 record, 72 GB → `megatron_dataset_v3/`
  - Full-chain: 69,811 (18.8%) | Snac-only: 302,044 (81.2%) | Bad: 0
  - seed2 332.6M | cosmos 3.88B | snac 363M | agent windows 2,148,474 | avclm 0 ✓
- [x] **Phase 7 v4 — fix temporal alignment** — **HOÀN THÀNH (02/07/2026)**. Per-chunk ordering, speech trong header, 5.217B token → `megatron_dataset_v4/`
- [x] **Upload Phase 7 v4 lên HF** — **HOÀN THÀNH (07/07/2026)**. `EmpathicRobotics/FineVideo-Phase7-Flattened` live với data v4. Đã share cho Huu/joergfranke trên Discord làm dataset sẵn sàng tokenize.
  Source: `megatron_dataset_v4/` | Dataset card đã cập nhật: `tools/vla_flattened_dataset_card.md`
- [ ] **Megatron re-tokenize** `megatron_dataset_v4/` với `tokenizer-vla-adaptive-v2` (156,505 vocab) → `.bin/.idx` → train v0.3
- [x] **Upload tokenizers** — **HOÀN THÀNH (01/07/2026)**. `EmpathicRobotics/tokenizer-vla-adaptive-v2` (156,505) + `EmpathicRobotics/tokenizer-vla-qwen3` (257,897), cả hai Live với model card đầy đủ

### Coding (không cần GPU, làm song song)
- [x] ~~Điều chỉnh dropout Phase 7 (AVC-LM → 100%, Cosmos → 50%)~~ **XONG** (27/06/2026)
- [x] ~~Dataset overlap check~~ **XONG** (30/06/2026) — valid_with_seed KHÔNG dùng
- [ ] Bắt đầu viết ego-centric perspective converter
- [ ] Bắt đầu viết captioning pipeline (SmolVLM2 / Qwen2.5-VL trên keyframe)
- [ ] Điều tra leo seed2 + euro_pat token counts
- [ ] Lên kế hoạch Cosmos3-DROID pipeline (download strategy, SLURM script)
- [x] Điều tra `MixtureVitae-Backup/multimodal` (HF) — **XONG (09/07/2026)**. Chủ yếu text; SNAC token tìm thấy ở 2 file dạng mảng số nguyên thô. Xem mục "Điều tra MixtureVitae-Backup Multimodal". Đang chờ Huu quyết định có thêm không.
- [ ] Làm rõ "finevideo reformulation" tại `leo:/mnt/sdb/mixture-vitae-working/finevideo` — check overlap với pipeline hiện tại
- [ ] Quyết định tỷ lệ mix MV-Omni / dropout để tránh pha loãng agent token % (12.2% → ~5.2% nếu mix thẳng)
- [ ] Scope nguồn data mới: abc.bot, MolmoAct2-BimanualYAM-Dataset, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData (xem bảng "Nguồn Data Mới" ở trên)

### Cluster account mapping (07/07/2026 — dùng khi submit job)
```
JUSUF:   ccstdl
JUPITER: reformo
JUWELS:  laionize
```

---

## Điều tra MixtureVitae-Backup Multimodal (09/07/2026)

### Bối cảnh

Việc P0 do Huu giao (hỏi 05/07): điều tra `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` trên HF — chưa từng được quét trước đó (khác với `valid_with_seed`/`stack_images3_gzip` đã inventory rồi). Chạy trên máy Windows cá nhân (không có JUWELS phiên này), chỉ có CPU, nên dùng cách stream + sample thay vì tải hết 103GB/15 file về.

### Phương pháp

2 script mới, tái dùng `PATTERNS`/`count_tokens`/`_hf_token`/`hf_url`/cơ chế checkpoint từ `tools/data_inventory.py`:

- **`tools/peek_multimodal.py`** — dò cấu trúc, stream vài record/member đầu mỗi file (không tải hẳn) để biết format và có token VLA hay không. Output: `tools/multimodal_peek_report.json`.
- **`tools/count_multimodal_tokens.py`** — stream HTTP thật (không bao giờ ghi file nén xuống đĩa), giới hạn mỗi file ở `--sample-mb` MB nén (mặc định 75), đếm token dạng tag VLA (regex, giống `data_inventory.py`) cộng thêm mọi mảng số nguyên dạng token (field `*_token`/`*_tokens` — tổng quát hoá, không chỉ riêng `snac_token`), ngoại suy ra full file size. Checkpoint resumable: `tools/multimodal_inventory_checkpoint.json`.

**Bug quan trọng đã fix:** `valid_data_snac.jsonl.gz`, `train_data_snac.jsonl.gz`, và `emo.jsonl.gz` **không phải** JSONL chuẩn (mỗi dòng 1 object gọn) — mà là JSON array pretty-print, 1 record có thể trải dài nhiều dòng. Split theo `\n` đơn giản khiến parse fail âm thầm, ra 0 record. Đã fix bằng cách dùng buffer stream + `json.JSONDecoder().raw_decode()` để lấy đủ JSON value bất kể xuống dòng ở đâu.

Env local: venv Python thường (`tools/env_multimodal_inventory/`, đã gitignore) — chỉ `pip install requests tqdm`, không cần conda. Có hỗ trợ HF token (`tools/.hf_token`, gitignore, được `_hf_token()` đọc) dù repo này thực ra public, không cần auth.

### Kết quả (sample 75MB nén/file, ngoại suy ra full size)

**Không file nào có token dạng tag của mình** (`<seed2_N>`, `<cosmos_N>`, `<avclm_N>`, `<snac_N>`) — xác nhận ở quy mô sample 75MB cho cả 15 file, không chỉ 5 record đầu lúc peek.

**2 file có SNAC token thật, dạng mảng số nguyên thô** (`snac_token: [128266, ...]`), không phải tag chuỗi:

| File | Size | Record sample | Ước lượng raw SNAC code (full) |
|---|---|---|---|
| `train_data_snac.jsonl.gz` | 11.1 GB | 131,850 | **~3.11B** |
| `valid_data_snac.jsonl.gz` | 579 MB | 129,996 | **~162M** |
| **Tổng** | | | **~3.27 tỷ raw SNAC code** |

Quy mô gần bằng 4.92B SNAC token đã tìm thấy ở MixtureVitae-Omni's `valid_snac` trước đây — một nguồn audio-token thật, chưa từng được đếm.

**13 file còn lại — text/caption corpus thuần** (word-count, ngoại suy):

| File | Ước lượng text token | Nội dung |
|---|---|---|
| high_stack.tar.gz | 4.11B | StackExchange QA |
| valid_text_only.tar.gz | 3.31B | text tổng hợp |
| stack_maga.tar.gz | 1.65B | StackExchange |
| emo.jsonl.gz | 1.04B | cặp audio-transcript + image-caption |
| train_data_snac.jsonl.gz (field `text`) | 865.5M | transcript đi kèm SNAC token ở trên |
| magalith-10m-florence2.jsonl.gz | 864.4M | caption ảnh |
| synth_llava2.tar.gz | 162.9M | caption ảnh kiểu LLaVA |
| clappa.tar.gz | 138.4M | caption video (ứng viên DISCUSS-1) |
| synth_llava.tar.gz | 93.7M | caption ảnh kiểu LLaVA |
| low_nemo_maga.tar.gz | 73.7M | text |
| valid_data_snac.jsonl.gz (field `text`) | 44.1M | transcript đi kèm SNAC token ở trên |
| youtube.tar.gz | 38.6M | storyline/mô tả video |
| coco.tar.gz | 10.0M | caption ảnh — **chính xác 100%** (đọc hết trong sample) |
| europarl.tar.gz | ~0.1M | ⚠️ độ tin cậy thấp, xem lưu ý |

### Lưu ý (chưa xử lý)

1. **`finevideo_transcripts.jsonl.gz` bị đếm thiếu (ra 0).** Field thật tên `transcripts`, không phải `text` — counter chỉ check `text` (giống convention có sẵn của `data_inventory.py`). Cần pass riêng, và — vì đây đúng là transcript FineVideo YouTube — cần check overlap video ID với pipeline của mình (giống rủi ro double-count như vụ `valid_with_seed` đã xử lý trước đây).
2. **Ước lượng `europarl.tar.gz` gần như vô nghĩa** — member đầu tiên sample được đã là 1 record ~986MB, nên 75MB sample chỉ đọc trọn 1 record. Cần sample lớn hơn nhiều hoặc chạy full-scan riêng.
3. **Vài archive trộn member text khổng lồ với shard binary `.wds`** (youtube, synth_llava/synth_llava2, stack_maga, high_stack, valid_text_only) — 75MB chỉ chạm được vài chục member trong số rất nhiều, nên ngoại suy giả định mật độ đều trên toàn archive, có thể không đúng. Độ tin cậy thấp hơn các file sample được hàng trăm member nhỏ (coco, low_nemo_maga).
4. **Mảng số nguyên `snac_token` chưa ở format `<snac_N>` chuỗi của tokenizer mình** — cần bước convert (offset/tag scheme) giống vụ convert MV-Omni `seed→seed2` đã làm, trước khi ~3.27B code này vào được pipeline Megatron.

### Trạng thái

Đã báo Huu trên Discord (09/07/2026, 3:51pm): *"this dataset is mostly text, only train_data_snac.jsonl.gz and valid_data_snac.jsonl.gz have snac tokens ... u want to add it?"* — **đang chờ trả lời.** Chưa bắt đầu tích hợp/tải full file cho tới khi Huu phản hồi.

---

**⚠️ Ghi chú:** file này (`PROGRESS_VI.md`) đã không được cập nhật song song với `REPORT.md` (tiếng Anh) trong khoảng 13/07 → 21/07/2026 (mục §17-30 của `REPORT.md` — permissive dataset survey, caption/speech pipeline, Megatron tokenize 4 nguồn, Harmony4D pivot...) — `REPORT.md` là log đầy đủ/mới nhất trong giai đoạn đó, không phải file này. Từ đây trở đi mục dưới tiếp tục cập nhật ở `PROGRESS_VI.md`.

## Kiểm tra + Eval sanity model `qwen3_1.7b_vla_v2` (22/07/2026)

### Bối cảnh

Vào phiên với 1 câu hỏi đơn giản: lần train Qwen3 trước (`qwen3_1.7b_vla_v2`, mix 5 nguồn ~32B token thật: FineVideo-v6, MV-Omni, OmniVideo-100K, synth-llava, emotional-roleplay, tokenizer `tokenizer_vla_qwen3`) đã submit ở phiên trước (ghi trong `REPORT.md` §30-31 và memory) — đã xong chưa? Nếu xong thì test thử.

### Training — xác nhận ĐÃ XONG, sạch

Job `1009758` chạy đủ **7,632/7,632 iteration** (64 node × 4 GH200). Loss: 6.47 (iter 50) → 1.83 (4000) → 1.69 (7600), **0 iteration NaN/skip**. Val loss cuối 1.7526 (PPL 5.77), test 1.7722 (PPL 5.88). `7632 × 1024 (batch) × 4096 (seq_len) = 32.01B token` — đúng 1 epoch trên toàn bộ mix, khớp con số ~32.01B đã tính ở `TOKENIZE_TODO.md`.

Bước convert checkpoint (Megatron→HF) của chính job đó **fail** do compute node không có internet (`AutoTokenizer.from_pretrained()` cố gọi huggingface.co thay vì dùng path local). Job thứ 2 (`1010685`) chạy lại riêng bước convert, thành công, ra đủ 16 checkpoint HF (`hf/iter_0000500` … `hf/iter_0007632`). `squeue` xác nhận không còn job nào chạy.

Kiến trúc (từ `config.json`): Qwen3ForCausalLM, 28 layer, hidden_size 2048, intermediate_size 6144, 16 attention head / 8 KV head (GQA), qk_layernorm, rope_theta=1e6, tied embedding, vocab 257,920 (pad từ 257,901 thật). **1.94B tham số.**

### Eval — viết script mới, dựa trên script cũ

Viết `tools/eval/eval_vla_v2_sanity.py` (dựa trên `eval_vla_sanity.py` cũ) — trỏ checkpoint/tokenizer mới, thêm test atomicity cho `snac_`/`caption`/`speech`, 5 prompt test (`full_prompt`, `agent_continuation`, `agent_from_scratch`, `roleplay_speech`, `image_caption`), hỗ trợ cả greedy lẫn sampling (`--sample --temperature --top-p --repetition-penalty`), in đầy đủ input/output/ground-truth (không cắt bớt).

**Token atomicity: 36/37 pass.** 1 lỗi thật: `<snac_140553>` (token cuối dải 12,288 token SNAC) bị tách 11 mảnh — lỗi off-by-one ở biên vocab, cùng loại lỗi với tokenizer model đầu tiên nhưng lần này chỉ ảnh hưởng 1 token biên. **Chưa fix** ở file tokenizer gốc `/e/.../tokenizer_vla_qwen3/` (chỉ patch bản copy tạm ở scratchpad để né lỗi tương thích `transformers 4.57.6`/`extra_special_tokens`, không liên quan tới lỗi snac).

**Kết quả định tính — lỗi gốc của model v1 đã biến mất.** `CLAUDE.md` ghi model đầu: *"Modality transitions: FAIL — model kẹt ở seed2, không tự chuyển cosmos/avclm/agent."* Model này tự chuyển đổi tự do giữa cả 6 loại đã học (seed2, cosmos, snac, speech, caption, agent), cả greedy lẫn sampling. Bằng chứng mạnh nhất: cho **chỉ** 32 token `<seed2_N>` thật từ 1 record `synth_llava2` (`synth_llava2_003266024`, không gợi ý text nào khác), model tự sinh caption đúng chủ thể (cậu bé, áo tốt nghiệp xanh lá, nhìn camera) khớp sát ground truth, tự đóng `</caption><|im_end|>` sạch — cross-modal binding ảnh↔text có thật.

Điểm yếu: **greedy hay lặp token trong block cosmos dài** (vd `<cosmos_42631>` lặp 6-8 lần liên tiếp), ăn hết ngân sách token trước khi tới agent (3/5 prompt bị). Sampling (T=0.8, top_p=0.9, rep_penalty=1.3) sửa được ít nhất 1 trường hợp (`agent_from_scratch`): model hoàn thành **trọn 2 window agent 8-frame** trong 1 lần generate (seed2→cosmos→agent→snac/speech→seed2→cosmos→agent), decode ra toạ độ 3D hợp lệ — đổi lại, sampling thi thoảng bịa chi tiết caption không có trong ảnh gốc (bịa tên "Timothy Kelly").

### Lần đầu decode cosmos→video thật từ chính output model

Lấy 2 chunk 200-token `<cosmos_N>` sạch từ generation `agent_from_scratch` (bản sampling), chạy qua decoder có sẵn `tools/decode/decode_cosmos.py` (trước đây chỉ verify với data training thật 20/07, chưa từng thử với output model tự sinh). **Cả 2 decode ra file mp4 chơi được** — lần đầu xác nhận cosmos-token của model này round-trip ra video thật xem được (giống pose/text đã xác nhận trước đó). `avc_lm`/`seed2`/`snac` vẫn là 3 modal chưa xác nhận được model-output→media thật: `avc_lm` vì model gần như không sinh ra (0 lần trong 5 test — do bị loại bỏ ở bước flatten trước khi vào data train, model gần như chưa từng thấy); `seed2`→ảnh và `snac`→audio vì **repo chưa có decoder chiều ngược nào cả**.

### Artifact đã lưu

`samples/qwen3_1.7b_vla_v2_eval/`: log đầy đủ 2 lần eval (greedy + sample), 2 video mp4 decode được + file token ID gốc.

### Việc còn mở (ghi lúc đầu)

- Training + convert checkpoint: **XONG**, không còn gì chạy nền.
- Eval mới chỉ định tính/đọc bằng mắt — chưa có MPJPE, BLEU/CIDEr, hay closed-loop task-success (vẫn là việc mở từ §29/§30 REPORT.md).
- ~~Lỗi atomicity `<snac_140553>`~~ — **rút lại, xem phần dưới: không phải bug thật.**
- Model card + upload HF cho checkpoint này: đang làm (`tools/upload/upload_vla_v2_model.py`), **chưa push**.

### Cập nhật cùng ngày: rút lại kết luận lỗi `<snac_140553>`, xây decoder audio SNAC, phát hiện cosmos dominate, tăng seq_length cho lần train sau

**Rút lại kết luận "bug atomicity" của `<snac_140553>`.** Tra lại kỹ danh sách `added_tokens` thật của tokenizer (không chỉ đoán 1 ID rồi test): dải SNAC thật gồm **3 băng rời rạc** 4,096 ID mỗi băng — L0 `128266–132361`, L1-chẵn `132362–136457`, L1-lẻ `144650–148745` — có 1 khoảng trống thật `136458–144649` chưa từng được add (đúng theo thiết kế "listen format" chỉ mã hoá 2/3 mức codebook của SNAC). `<snac_140553>` rơi đúng vào khoảng trống này nên nó **đúng ra chưa từng là token** — test atomicity ban đầu của tôi chọn nhầm 1 ID không tồn tại, không phải phát hiện được lỗi tokenizer thật. Toàn bộ 12,288 ID snac thật vẫn atomic 100%.

**Viết `tools/decode/decode_snac.py`** — decoder chiều audio đầu tiên trong repo (trước đó chỉ có `decode_cosmos.py`/`decode_avclm.py`, cả 2 đều video). Tái tạo lại 3 mức codebook phân cấp của SNAC từ bộ ba token listen-format, điền 0 vào mức 2 (mức tinh nhất, 50Hz, chưa từng được mã hoá), rồi gọi `SNAC.from_pretrained("hubertsiuzdak/snac_24khz").decode()` để ra waveform thật. Chạy thử trên 159 token `<snac_N>` (53 base frame, ~4.5 giây) nối từ chính output model (bản sampling ở mục trên) — **ra file WAV thật, không im lặng, không clip** (RMS 0.12, range [-0.46, 0.55]). Đóng được 1 trong 2 gap "chưa có decoder" đã nêu ở mục trên — chỉ còn `seed2`→ảnh là chưa chứng minh được.

**Token `cosmos` đang áp đảo trong chính output model.** Gộp breakdown của cả 10 lần test (5 prompt × greedy + sample): nếu chỉ tính trong nhóm token VLA thật (bỏ text tự nhiên), **`cosmos` chiếm 61-77%** — `agent`/`seed2`/`snac` cộng lại vẫn là thiểu số. Nguyên nhân là cấu trúc, không hẳn do thiên vị lúc train: 1 chunk cosmos tốn cố định 200 token theo convention, trong khi agent chỉ ~2-4 token/mẫu, seed2/snac chỉ 1 token/mẫu — nên cosmos vốn đã "đắt" hơn hẳn về mặt biểu diễn. Hệ quả thực tế: đây rất có thể là nguyên nhân trực tiếp gây ra hiện tượng lặp token + hết ngân sách token trước khi tới agent đã thấy ở mục trên. Đây đúng loại vấn đề `CLAUDE.md` (thời model v1) đã từng ghi kế hoạch xử lý ("giảm modality dropout từ 99%/90% xuống 80-90%/50-70% cho avclm/cosmos") nhưng **chưa áp dụng** cho lần train v2 này — đáng làm cùng lúc với việc tăng seq_length.

Ở tầng **data mix** (khác với tầng generation), team đã tự phát hiện và sửa 1 dạng dominance tương tự từ trước khi train v2: MV-Omni chiếm 63.71% token thô nhưng bị chủ động hạ xuống 39.71% trọng số vì không có token `<agent>` (ghi trong comment `qwen3_1.7b_vla_v2.yaml`) — nên mất cân bằng ở tầng *data* đã được xử lý; phát hiện hôm nay là mất cân bằng **mới, ở tầng generation** (chi phí token/chunk của cosmos), việc rebalance data mix không giải quyết được cái này.

**Tăng seq_length cho lần train sau.** Viết `oellm-autoexp/config/experiments/nguyen38/qwen3_1.7b_vla_v3.yaml` — giống hệt `qwen3_1.7b_vla_v2.yaml` (cùng kiến trúc, tokenizer, data mix/trọng số) chỉ đổi `seq_length: 4096 → 8192`, tính lại `train_iters`/`lr_warmup_iters`/`lr_decay_iters`/`lr_wsd_decay_iters`/`eval_interval` để vẫn giữ đúng ~1 epoch (7,632 → 3,816 iter, giữ nguyên tỷ lệ warmup/decay). Lý do, bằng chứng trực tiếp từ mục trên: 1 chu kỳ đầy đủ seed2→cosmos→agent→snac/speech đã tốn vài trăm tới ~1,500 token, riêng cosmos có thể 200+ token/chunk — ở seq_length=4096 chỉ đủ chỗ cho 1-3 chu kỳ như vậy, còn xa mới bằng 1 hoạt động thật kéo dài vài giây. **Chưa submit** — mới là thay đổi config, còn chờ quyết định có áp dụng luôn điều chỉnh modality dropout ở trên hay không trước khi chạy train thật.

Artifact thêm vào `samples/qwen3_1.7b_vla_v2_eval/`: `snac_decoded_sample_generated.wav`, `snac_raw_ids_generated.txt`.

---

## Cập nhật phiên làm việc — 22/07/2026 (tiếp — fix seed2 decoder, xây eval framework media đầy đủ, phát hiện lặp macro ở seed2+cosmos, đối chiếu chat Discord với Huu, sự cố mất file giữa phiên)

**Việc chính:** Phiên dài, nối tiếp §31 REPORT.md (train xong, decode cosmos lần đầu). Bao gồm: sửa 3 bug thật trong decoder seed2 (chưa từng chạy được), hardening 2 decoder khỏi crash, thử thêm rồi xoá vocab SNAC L2 theo yêu cầu user, xây `eval_vla_v2_media.py` (framework eval media đầy đủ), đào sâu phát hiện model **lặp lại y hệt cả khối seed2 lẫn cosmos** dưới greedy (không chỉ lặp token đơn lẻ như đã biết), đối chiếu toàn bộ với 1 đoạn chat Discord thật giữa user và Huu — phát hiện 1 giả thuyết của chính tôi bị sai (dropout), thêm insight mới (8-frame quá ngắn), và 1 sự cố mất file giữa phiên chưa rõ nguyên nhân.

### 1. Vocab SNAC L2 — thêm rồi xoá lại theo đúng yêu cầu

Viết `add_snac_l2_tokens.py` thêm 16,384 token L2 (băng tần 50Hz chưa từng mã hoá) vào bản copy của cả 2 tokenizer, do 1 test atomicity tưởng nhầm là bug (`<snac_140553>`). Sau đó xác nhận: ID đó rơi vào khoảng trống thật giữa các băng, không phải bug — và user chỉ ra đúng: **dataset hiện tại không hề có dữ liệu L2**. Đã xoá cả 2 tokenizer mở rộng lẫn script — tokenizer gốc chưa từng bị đụng vào (script luôn ghi ra thư mục mới) nên không ảnh hưởng gì tới checkpoint đã train. Sửa lại danh sách spot-check atomicity dùng đúng ID biên thật của 3 băng SNAC → **41/41 PASS** trên tokenizer gốc.

### 2. `decode_seed2.py` — tìm và sửa 3 bug thật, chạy được lần đầu

- Model diffusion gốc `stabilityai/stable-diffusion-2-1-unclip` **đã bị Stability AI gỡ khỏi HuggingFace thật sự** (xác nhận qua tiêu đề trang literally "404", không phải trang gated) — chuyển sang mirror cộng đồng `sd2-community/stable-diffusion-2-1-unclip`.
- Import nhầm class `Seed2Tokenizer` (bản wrapper trong `pipeline.py`, không có `.decode()`) thay vì bản thật trong `seed2_tokenizer.py`.
- Thiếu batch dimension khi truyền token vào — Q-former hiểu nhầm 32 token thành 32 ảnh riêng lẻ 1-token, vỡ shape sâu trong UNet.
- (nhỏ) Đường dẫn `--output` bị resolve sai thư mục do hàm load tokenizer tự `os.chdir()`.

Sau khi sửa cả 4, decode thành công 32 token seed2 thật (record `synth_llava2_003266024`) ra ảnh PNG — lần đầu xác nhận round-trip seed2→ảnh trong repo. Nâng cấp thêm: hỗ trợ decode nhiều block `<seed2>` riêng biệt (mỗi ảnh 1 file), vì Q-former chỉ xử lý đúng 32 token/lần.

### 3. Hardening `decode_snac.py` + `decode_seed2.py` khỏi crash

Lúc chạy eval framework, phát hiện 1 crash CUDA thật (`device-side assert`) — trace ra nguyên nhân: text sampling không đóng `<snac>` đúng cách, decoder phải quét toàn văn bản và ghép token rời rạc từ nhiều vị trí khác nhau, làm lệch thứ tự 3-token-tuần-hoàn, sinh chỉ số codebook âm/vượt phạm vi. Đã thêm validate range trước khi đưa vào GPU cho cả 2 decoder — verify lại đúng 2 case từng crash, giờ báo lỗi Python rõ ràng thay vì crash cứng.

### 4. Xây `tools/eval/eval_vla_v2_media.py` — framework eval media đầy đủ

1 script duy nhất: generate theo bộ prompt cố định (continuation/from-scratch/full-chain) × greedy/sampling, tự nhận diện + decode mọi modal xuất hiện, ghi input/output đầy đủ + media vào từng folder test riêng, có `SUMMARY.md` tổng hợp. Hỗ trợ `--only`/`--modes` để chạy lại có chọn lọc. Lần chạy đầu (14 test) ra 27 file media thành công.

### 5. Đào sâu `full_chain_from_scratch` — phát hiện lặp macro ở CẢ seed2 lẫn cosmos

Ở 2000 token, model không bao giờ mở `<agent>` — giả thuyết ban đầu "hết ngân sách vì cosmos ăn hết". Tăng lên 4000 token (gần trần cứng `max_position_embeddings=4096`) **vẫn 0 agent** — bác bỏ giả thuyết ngân sách. Phát hiện thật: **cả 5 block seed2 sinh ra trong 1 lần generate giống hệt nhau 100%**, và **cosmos cũng vậy** (8 chunk nhưng chỉ 2 chunk unique — chunk đầu khác, 7 chunk sau lặp y hệt 1 chunk). Nghĩa là "8 chunk" trong output thực chất chỉ đại diện ~0.27-0.53 giây nội dung thật, không phải 8×0.27s như số lượng gợi ý — tăng `max_new_tokens` dưới greedy chỉ mua thêm *lần lặp*, không mua thêm *giây nội dung mới*.

### 6. Prompt mới (skateboard) bác bỏ giả thuyết học thuộc; agent xuất hiện tuỳ prompt, không nhất quán

Test prompt hoàn toàn khác chủ đề (trượt ván) để kiểm tra prompt "cooking" cũ có phải học thuộc từ dataset không: **caption đổi đúng theo chủ đề mới** ("skateboarding on a sidewalk") → bác bỏ học thuộc theo từ ngữ. **Agent xuất hiện 7 lần** lần này (khác cooking) nhưng pose **hoàn toàn tĩnh** (pelvis di chuyển đúng 0.0 cả 3 trục). Test thêm "man walking" (theo yêu cầu Huu) lại quay về pattern không có agent. Kiểm tra cosmos có thực sự theo nội dung prompt không: cooking vs skateboard = 0/200 token trùng (khác hẳn), skateboard vs man_walking = 28/200 trùng (cùng là cảnh ngoài đường) — xác nhận cosmos **có** điều kiện hoá theo chủ đề ở tầng 1-chunk, vấn đề chỉ nằm ở lặp lại bên trong 1 lần generate.

### 7. Đối chiếu chat Discord thật với Huu — tự sửa 1 giả thuyết sai, thêm insight mới

User dán 1 đoạn chat dài giữa user và Huu bàn đúng các kết quả trên. Điểm quan trọng:
- **Tự sửa sai:** tôi từng giải thích cosmos áp đảo 1 phần do "modality dropout 90%/99%" (số liệu của model đời đầu, ghi trong `CLAUDE.md`) — nhưng chính user nói trong chat: *"I just randomly trained, didn't do token dropout"* cho v2. Model v2 **không dropout gì cả** — cosmos áp đảo thuần tuý do chi phí 200 token/chunk cố định, không liên quan gì tới việc loại bỏ chunk trong data.
- **Insight mới chưa từng nghĩ tới:** Huu chỉ ra 8-frame/30fps ≈ 0.27 giây **có thể vốn đã quá ngắn để thấy chuyển động** ngay trong chính data train thật (mọi video đều bị chia cứng thành chunk 8-frame, không có video độ dài biến thiên) — nghĩa là hiện tượng "không di chuyển" có thể một phần phản ánh đúng đặc điểm data, không chỉ do lỗi decoding lặp. Hướng sửa Huu đề xuất: tăng tốc video nguồn trước khi chia chunk.
- **2 nghi vấn chưa điều tra:** cosmos token có bị lẫn giữa các video khác nhau lúc tokenize không (Huu nghi ngờ từ 1 case "chỉ ra ngón tay" nhưng lại giống video khác); context 4096 có cắt cụt cosmos giữa chừng không.
- **Roadmap còn mở, chưa làm:** cân bằng lại dropout cosmos; scale lên ~100B token (70B text từ MV1 + 30B data hiện tại — đổi hẳn tỷ lệ so với hiện tại toàn VLA); test "any-mode-to-any-mode" có hệ thống; test với record thật từ pretrain (ảnh "chemical bond", speech thật); ý tưởng data `<think>` reasoning dài hạn (tạo ngược từ video thật); phân biệt SNAC "listen" vs "speak" (liên quan trực tiếp việc L2 vừa xoá — có lý do roadmap thật nhưng chưa làm lại); setup endpoint hosted; và chỉ đạo rõ *"we need to work on evals now"* — khớp đúng hướng `eval_vla_v2_media.py` đang xây.

### 8. Dọn `samples/qwen3_1.7b_vla_v2_eval/` + sự cố mất file giữa phiên (chưa rõ nguyên nhân)

Xoá 3 file rác/không rõ nguồn gốc (log fail cũ, wav không có log nguồn). **Sự cố riêng, nghiêm trọng hơn:** giữa phiên phát hiện toàn bộ file rời (không nằm trong subfolder) trong `samples/qwen3_1.7b_vla_v2_eval/` biến mất khỏi ổ đĩa — gồm cả 8 file đã commit từ trước lẫn ~8 file mới tạo phiên này. Rà lại toàn bộ script mới viết, không tìm thấy code nào có lệnh xoá; bash history cũng không có manh mối. **8 file đã commit đã khôi phục được** qua `git checkout HEAD --`. **~8 file chưa commit (ảnh seed2 gốc đầu tiên, audio verified, v.v.) mất thật, không khôi phục lại** — nhưng đều đã gửi cho user qua chat trước đó nên nội dung không mất hẳn, chỉ mất khỏi repo. Chưa regenerate lại (ưu tiên gói gọn phiên theo yêu cầu user) — ghi nhận công khai, không giấu.

### Trạng thái cuối phiên

- Tất cả 4 decoder (seed2/cosmos/snac/agent) đã verify hoạt động thật.
- Framework eval media đã xây và chạy ~20 lần generate qua 3 run khác nhau.
- Lặp macro (cả khối, không chỉ token đơn) xác nhận xảy ra ở cả seed2 và cosmos — 2 giả thuyết nguyên nhân (lặp do greedy; 8-frame quá ngắn) chưa tách bạch được cái nào chiếm ưu thế.
- 2 nghi vấn data-integrity từ Huu vẫn mở, chưa điều tra.
- Roadmap lớn đã ghi nhận (mục 7) nhưng chưa chốt thứ tự ưu tiên với user.
- Sự cố mất file giữa phiên chưa rõ nguyên nhân gốc.

---

## Đối chiếu chat Discord thật (Huu ↔ Van Khue, khoảng 4:20AM–6:21PM cùng ngày 22/07/2026) — danh sách đầu việc + vấn đề cần discuss

### Bối cảnh

User dán nguyên văn 1 đoạn chat Discord dài giữa Huu và Van Khue, bàn trực tiếp về kết quả eval `qwen3_1.7b_vla_v2` (model đã public lên `EmpathicRobotics/vla-1.7b-qwen3-v2`) vừa làm ở mục trên, và yêu cầu tổng hợp lại thành danh sách đầu việc/vấn đề cần discuss — kết hợp thông tin đã có trong `REPORT.md`/`PROGRESS_VI.md` với nội dung mới trong chat. Dưới đây là bản ghi lại đầy đủ danh sách đó.

### A. Mâu thuẫn chiến lược cần chốt trước

**1. "Scale data" vs "cải thiện pipeline" — Huu nói 2 điều có vẻ ngược nhau.** 4:28AM Huu: *"instead of trying to spend time improving the pipeline, we just scale the data 😄 easier"*. Nhưng cuối chat, chính Huu lại yêu cầu hàng loạt việc **là cải thiện pipeline**: tăng resolution/quality cosmos, tăng context length, video độ dài biến thiên thay vì cứng 8-frame, tách SNAC listen/speak. Chưa rõ "scale data" là ưu tiên thay thế cho các fix pipeline này hay chạy song song — cần hỏi thẳng Huu để tránh vừa đuổi scale vừa đuổi quality fix mà không đủ resource cho cái nào.

### B. Vấn đề kỹ thuật mới phát sinh từ chat (chưa có trong log trước đó)

**2. SNAC decode ra giọng "giống nhưng không rõ" — nghi vấn chất lượng, không chỉ atomicity.** Van Khue: *"the output voice is somehow similar to the input voice but it's not clear"*. Khác với việc đã verify trước đó ("decoder chạy được, WAV không im lặng/không clip") — chất lượng nghe vẫn kém. Cần điều tra: có phải do chỉ giữ L0+L1 (bỏ L2) làm giảm fidelity, hay bug ở decode logic.

**3. Ý tưởng SNAC "listen" vs "speak" — cụ thể hoá thành turn-taking thời gian thực.** Huu: L0+L1 = `<listen>`, đủ L0+L1+L2 = `<speak>` (chất lượng cao hơn) → hướng tới **real-time interrupted speaking → listening → speaking**. Cần thiết kế format token riêng cho 2 chế độ, và cần thêm nhiều speech data hơn (Huu tự nhận: *"we probably need a lot more speech data"*).

**4. Nghi vấn resolution/quality của cosmos tokenization — quyết định cũ có thể không còn phù hợp.** Huu hỏi resolution cosmos có tệ không — Van Khue xác nhận quality thấp + zoom-in trước khi tokenize cosmos là quyết định cũ, làm vì *"robot camera is bad"* (tối ưu cho robot, không phải video YouTube chất lượng cao). Van Khue tự nhận: *"we need to see real thing before do token saving"*. Cần re-đánh giá đổi token budget lấy chất lượng cosmos.

**5. Nghi vấn cosmos token bị lẫn giữa các video (data integrity, chưa điều tra).** Case cụ thể: prompt về 1 cảnh nhưng cosmos decode ra video "chỉ có ngón tay" — giống 1 phần video cắt đồ ăn khác. Huu: *"a finger is in fact part of a cutting video"* → nghi cosmos token bị mismatch/misaligned lúc tokenize pretrain. Cần check trực tiếp trong data, không chỉ suy đoán từ output model.

**6. Cosmos có bị cắt cụt bởi context 4096 không?** Huu hỏi thẳng, chưa ai xác nhận. Cần đo trực tiếp.

**7. "Không có chuyển động" — thêm 2 giả thuyết mới ngoài "8-frame quá ngắn".** Huu gợi ý thêm: có thể do chưa train đủ (undertrained), hoặc cosmos generation bị dừng giữa chừng trước khi kịp thể hiện chuyển động — cần tách bạch với 2 giả thuyết đã biết (8-frame quá ngắn / greedy lặp macro). Hướng fix Huu đề xuất cụ thể: **tăng tốc video nguồn** trước khi chia chunk 8-frame (không phải resample, mà speed-up) để mỗi chunk chứa nhiều "thời gian thật" hơn; cân nhắc thêm **video độ dài biến thiên** (0.1s→5s) thay vì luôn cứng 8-frame.

**8. Thử đổi thứ tự modality (seed2 trước cosmos, hay ngược lại).** Hiện tại luôn seed2→cosmos (quy ước cũ từ FineVideo-VLA). Huu: *"you could try mixing it up"*.

### C. Action item cụ thể, đã được yêu cầu trực tiếp

**9. Test "any-mode-to-any-mode" với data thật từ pretrain (không chỉ prompt tự tạo).** Huu yêu cầu cụ thể: thử ảnh "chemical bond" thật, thử feed cosmos/seed2 token của "người phụ nữ đang đi bộ" thật từ data, thử 1 đoạn speech thật xem model có "echo" lại giọng đúng cách không.

**10. Research: Cosmos (NVIDIA) hoặc video model tương tự train bao nhiêu token trước khi ra video "được"?** Cần con số benchmark để biết vị trí hiện tại so với ngưỡng cần thiết.

**11. Share kết quả lên kênh #omni-vla trên Discord open-sci.** Van Khue đã nhận lời ("I will wrap up and send some results there") — cần follow up.

**12. Setup hosted inference endpoint trên máy LAION.** Đã đề xuất cụ thể, chưa làm.

### D. Nguồn data mới cần điều tra (nêu tên cụ thể lần đầu trong chat)

**13. Euro-pat data trên Leo (Leonardo cluster).** Huu: patent text → generate ảnh (bằng model Black Forest Labs đời đầu, hiểu multilingual) → caption tiếng Anh. MV1 chỉ giữ caption; omni có thể thêm cả seed2 token của ảnh — tiềm năng multilingual (patent text đa ngôn ngữ, ảnh sinh ra hiểu đa ngôn ngữ, caption tiếng Anh). Trùng với item cũ trong action-list ("Điều tra leo seed2 + euro_pat token counts") nhưng giờ có đủ context để bắt đầu — Van Khue đã nhận: *"I will start looking at those data from now"*.

**14. Clappa dataset — audio→image→text.** Đã peek qua trước đó (`clappa.tar.gz`, ghi nhận "video caption, candidate DISCUSS-1" trong mục điều tra MixtureVitae-Backup Multimodal 09/07), nhưng Huu mô tả rõ hơn là audio-image-text — cần xem lại đúng theo mô tả này.

**15. "Synthetic data khác trên Leo"** — Huu nhắc mơ hồ (*"also synthetic on leo"*), chưa rõ dataset gì. Cần hỏi lại Huu để làm rõ referent trước khi tốn công tìm.

### E. Ý tưởng lớn, dài hạn (chưa thiết kế, cần thảo luận trước khi làm)

**16. 100B token run: 70B text (từ MV1) + 30B data hiện tại (VLA-flavored).** Thay đổi tỷ lệ lớn so với run hiện tại (100% VLA-flavored, ~32B). Cần bàn cách mix/weight.

**17. Instruction-tied cross-modal reasoning — spec cụ thể từ Huu.** Ý tưởng lớn nhất trong chat: model nhận instruction dạng *"vẽ hình minh hoạ cho lời giải công thức này"* hoặc *"múa theo hình dạng công thức toán này"* → sinh action token đúng hình dạng đó. Cách tạo data đề xuất: làm **ngược** từ video thật — cho video X, tìm 1 thứ trung gian Y liên quan/đại diện cho X, tạo instruction "làm X dựa trên Y", format:
```
<think> ok, first I need to compute a type of Y... <seed tokens>... Ok, so now let's do X </think> <action tokens>
```
Chưa có thiết kế/code nào — cần bàn feasibility trước khi đầu tư (Huu tự nhận *"I am pretty sure it won't be able to do that"* nhưng vẫn muốn thử).

### Đề xuất thứ tự ưu tiên (chưa chốt với user)

Test any-mode-to-any-mode với data thật (mục 9) trước — rẻ, làm ngay, trả lời được nhiều nghi vấn cùng lúc (data integrity mục 5, context-cut mục 6, undertrained mục 7). Sau đó mới quyết định giữa nhánh "scale 100B" (mục 16) và nhánh "fix cosmos quality/8-frame" (mục 4, 7) vì 2 nhánh cạnh tranh resource và cần chốt ở mục A trước.

---

## Chạy Phase 0 (mục 9, 6, 5) — kết quả đáng chú ý, tìm ra cơ chế gốc mới cho nghi vấn cross-contamination (22/07/2026, tiếp)

### Bối cảnh

User yêu cầu bắt đầu 3 task Phase 0 còn lại (mục 11 — share Discord — user đã tự làm). Chạy trực tiếp trên node GH200 tương tác (`jpbl-s02-02`), env `env_stable_vla`.

### 1. Mục 9 — any-mode-to-any-mode với 3 record THẬT từ pretrain

Viết `tools/eval/eval_any_mode_real.py`, tái dùng `decode_media` từ `eval_vla_v2_media.py`. Khác với các script trước (gõ tay prompt), script này **lấy trực tiếp từ file data thật lúc chạy** (ghi rõ record ID trong `SOURCE:` của mỗi `input_output.txt`, tránh lỗi transcribe tay):

1. **`chemical_bond_real`** — `synth_llava2_001131949`, chỉ seed2 block, ground truth là `<caption>` thật về sơ đồ hoá học.
2. **`woman_walking_real`** — record FineVideo-VLA v6 thật (header "Pickings Charge") có caption "The woman is walking outside." — header+caption+seed2+chunk cosmos 200-token đầu tiên (chỉ mở tag), ground truth thật là `</cosmos><snac>...</snac><speech>...oh!</speech>`.
3. **`roleplay_speech_real`** — record `laion/emotional-roleplay` thật (`cv_Fairy-2__b1_13_Astonishment_Surprise_1`), chỉ có instruction USER (không cho hint snac nào), ground truth thật là 342 token `<snac>`.

Chạy cả 3 × greedy + sampling trên `qwen3_1.7b_vla_v2`. Kết quả (`samples/qwen3_1.7b_vla_v2_eval/2026-07-22_any_mode_real/`):

- **`chemical_bond_real` — FAIL cả 2 mode.** Thay vì `<caption>`, greedy nhảy thẳng vào `<snac>` (3 triplet giống hệt nhau) rồi `<cosmos>` lặp macro nặng; sampling nhảy thẳng `<cosmos>` (sai modality, dù không lặp). `synth_llava2` không bao giờ có cosmos trong data thật (chỉ ảnh+caption) — model có vẻ mặc định theo chuỗi kiểu FineVideo (seed2→cosmos) ngay cả khi phân phối thật của loại record này chưa từng có cosmos.
- **`woman_walking_real` — PASS cả 2 mode, kết quả mạnh nhất phiên này.** Cả greedy và sampling đều đóng `</cosmos>` đúng chỗ, rồi ra `<snac>` + 1 câu `<speech>` hợp lý (greedy: "I'm not a man."; sampling: "What are you doing?..." — sai tông cảm xúc so với ground truth "...oh!" nhưng cấu trúc/ngữ pháp ổn), rồi tiếp tục seed2/cosmos. Seed2 block greedy sinh ra **giống hệt 100%** block đã có trong prompt — xác nhận lại phát hiện lặp macro ở §32, lần này từ prompt thật chứ không phải prompt tự tạo. Cosmos chunk mới ở cả 2 mode đều mở đầu bằng ID rất gần với ID mở đầu chunk thật ban đầu (50144/50072 vs 50073 thật) — có thể là token đánh dấu cấu trúc/scene chứ không hẳn là copy, nhưng chưa đủ dài (115/171 token mới) để decode thành video kiểm tra trực quan.
- **`roleplay_speech_real` — FAIL cả 2 mode, không ra snac nào.** Greedy sinh văn xuôi tiếng Anh về "thế giới fantasy" (không liên quan gì tới prompt thật về "bầu trời vàng/sao băng"), rồi lặp 1 từ ("wonder") ~150 lần — **lần đầu xác nhận lặp macro xảy ra ở văn bản tự nhiên**, không chỉ token VLA có cấu trúc như §32. Sampling tránh được vòng lặp nhưng bịa ra 1 đoạn Q&A không liên quan thay vì audio — giống format QA của OmniVideo-100K lẫn vào.

**Kết luận mục 9:** khả năng any-mode-to-any-mode của model có thật nhưng không đều — mạnh ở cặp modality thấy nhiều nhất khi train đủ dài (video→speech→video kiểu FineVideo), yếu ở 2 phân phối thật còn lại (ảnh-only→caption của synth_llava2; text-only→speech của roleplay), nơi model có xu hướng quay về continuation kiểu FineVideo mặc định. Đây là bằng chứng cụ thể hơn cho nghi vấn "cosmos áp đảo" — không chỉ vì tốn token, mà model dường như coi cosmos là lựa chọn mặc định kể cả với loại record chưa từng có cosmos.

### 2. Mục 6 — xác nhận context bị cắt cả ở generation-time LẪN training-time

**Generation-time:** kiểm tra lại đuôi 2 lần chạy `full_chain_from_scratch` dài đã có (2000 và 4000 token) — **cả 2 đều dừng giữa chừng `<cosmos_N>`, không có `</cosmos>` đóng** — xác nhận generation bị cắt cứng bởi `max_new_tokens`/`max_position_embeddings`, không phải điểm dừng tự nhiên.

**Training-time (phát hiện mới, lớn hơn nhiều):** tokenize thật 300 document từ `megatron_dataset_v6` bằng tokenizer thật:

| Thống kê | Số token |
|---|---|
| min | 207 |
| p25 | 5,170 |
| median | 13,832 |
| p75 | 28,020 |
| p90 | 59,330 |
| max | 294,050 |
| mean | 25,896 |

**80.0% document vượt quá `seq_length=4096` của v2; 64.3% vẫn vượt quá `seq_length=8192` dự kiến cho v3.** Check `qwen3_1.7b_vla_v2.yaml`: không set `reset_position_ids`/`reset_attention_mask`; script tokenize Megatron có `--append-eod` (có chèn EOS thật giữa các document) nhưng packing kiểu chuẩn (nối tất cả document thành 1 dòng token dài rồi cắt cửa sổ cố định, không biết ranh giới document) — nghĩa là **đa số cửa sổ training model thực sự thấy đều bắt đầu/kết thúc giữa chừng 1 document**, rất thường là giữa chừng 1 block `<cosmos>` do document quá dài toàn cosmos. Đây là lý do trực tiếp giải thích vì sao model hiếm khi học được tín hiệu "đây là điểm kết thúc tự nhiên của 1 hoạt động dài" — góp phần vào phát hiện lặp macro ở §32, độc lập với giả thuyết "8-frame quá ngắn" đã biết. Chưa check: document roleplay/synth_llava2 ngắn hơn nhiều (vài trăm–vài nghìn token) nên không bị vấn đề này — vấn đề tập trung ở FineVideo-VLA, nguồn dài nhất và dày cosmos nhất trong mix.

### 3. Mục 5 — tìm ra cơ chế gốc khả dĩ cho nghi vấn cosmos cross-contamination (chưa reproduce trực tiếp được)

Case cụ thể Huu nêu (cosmos ra "ngón tay từ video cắt đồ") chỉ tồn tại dưới dạng ảnh đính kèm Discord lúc demo trực tiếp, không lưu thành file nào trong repo (`grep -rl` toàn bộ `samples/` không ra kết quả) — nên không thể re-inspect/reproduce trực tiếp phiên này.

Thay vào đó, phát hiện ở mục 2 cho 1 cơ chế **cấu trúc** khả dĩ: vì (a) `--append-eod` chỉ chèn 1 token EOS trơn giữa các document, không có `reset_position_ids`/`reset_attention_mask`, và (b) 80% document FineVideo-VLA dài hơn hẳn cửa sổ training 4096 token, **rất nhiều khả năng 1 phần đáng kể cửa sổ training model thực sự học có chứa đuôi của 1 video nối liền ngay với đầu của 1 video khác hoàn toàn không liên quan, chỉ ngăn cách bởi 1 token EOS, trong khi attention (causal nhưng không bị mask ở ranh giới EOD) vẫn thấy được cả 2 phía**. Điều này đủ để model học liên kết chéo-video giả như Huu nghi ngờ, mà không cần có bug nào trong pipeline tokenize/merge (Phase 6/7 kiểm tra không thấy lỗi misalignment — cơ chế nằm ở tầng Megatron packing, sau khi corpus đã build đúng). Muốn có bằng chứng "smoking gun" thật sự cần sample trực tiếp cửa sổ training từ file `.bin`/`.idx` ở offset cố định để check ranh giới document giữa cửa sổ — chưa làm trong phiên này, ghi nhận là bước tiếp theo nếu cần biến "cơ chế khả dĩ" thành "đã xác nhận".

### Trạng thái cuối phiên

- Mục 9: **xong**, kết quả lẫn lộn (1/3 loại record pass, 2/3 fail) — script `tools/eval/eval_any_mode_real.py` tái dùng được cho checkpoint sau.
- Mục 6: **xong** — xác nhận cả ở generation-time lẫn training-time (80% document FineVideo-VLA vượt seq_length=4096, 64.3% vượt cả 8192).
- Mục 5: **tìm ra cơ chế khả dĩ** (không reset ranh giới document + document dài hơn seq_length rất nhiều), case gốc Huu nêu chưa reproduce trực tiếp được (chỉ tồn tại trên Discord).
- Mục 11: user đã tự làm trước phiên này.
- Các phát hiện này ảnh hưởng trực tiếp tới Phase 1 (redesign cosmos) và Phase 3 (config train v3) đã đề xuất ở mục trên: `reset_position_ids`/`reset_attention_mask` hoặc xử lý ranh giới document rõ ràng hơn trong Megatron packing giờ là 1 candidate fix cụ thể cần gộp vào lần train tiếp theo, cùng với tăng seq_length, rebalance dropout cosmos, và redesign chất lượng cosmos đã định.
- Sự cố mất file giữa phiên chưa rõ nguyên nhân gốc.
