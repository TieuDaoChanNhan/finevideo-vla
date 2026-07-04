# PAB-Spline VLA — Tiến độ dự án

**Tác giả:** Van Khue Nguyen  
**Cập nhật lần cuối:** 04/07/2026  
**Cluster:** JUPITER (JSC), partition `booster`, GPU GH200  
**Mục tiêu:** Xây dựng mô hình VLA (Vision-Language-Action) — xem video, nghe tiếng, sinh ra token điều khiển robot.

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
- [ ] **Upload Phase 7 v4 lên HF** → `EmpathicRobotics/FineVideo-Phase7-Flattened`:
  ```bash
  export HF_TOKEN='hf_...'
  python tools/upload_flattened_hf.py
  ```
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
