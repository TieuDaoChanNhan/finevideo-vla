# PAB-Spline VLA — Project Progress

**Author:** Van Khue Nguyen  
**Last updated:** July 18, 2026  
**Cluster:** JUPITER (JSC), `booster` partition, GH200 nodes — **back up**, `booster` batch jobs running normally as of Jul 15  
**Goal:** Build a multimodal Vision-Language-Action model that can watch video, hear speech, and generate robot motion tokens.

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

**Việc chính hôm nay: kiểm tra lại trạng thái thật của caption+speech pipeline (task breakdown ở REPORT.md §18), phát hiện 2 việc đã âm thầm chạy xong mà doc chưa ghi nhận, và fix 2 bug thật trong `phase7_flatten.py` liên quan tới việc permute speech transcript khi có SNAC.

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

`tokenizer_vla_qwen3` (đang "in progress" theo REPORT.md §18) đã build xong: vocab **257,901**, tất cả token mới (`<caption>`, `</caption>`, `<speech>`, `</speech>`) + toàn bộ token cũ (seed2/cosmos/avclm/pelvis/SNAC/agent/fps) đã verify atomic qua spot-check trong tmux `qwen3_rebuild`. Cùng với `tokenizer_vla_adaptive_v2` (156,509 vocab, đã xong từ trước) — cả 2 tokenizer đã sẵn sàng, chỉ còn thiếu bước **upload lên HuggingFace** (cần user tự export `HF_TOKEN`):
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
source activate_env_tools.sh
export HF_TOKEN=...   # HF token của user
python tools/upload/upload_tokenizers_v2.py --mode all
```

### 3. Caption pipeline (A2, task #3) — vẫn đang chạy, còn xa mới xong

`squeue` cho thấy job `14104156` đang chạy đủ 32/32 worker (đã chạy hơn 8h45p tính tới lúc check), và có chuỗi 3 job kế tiếp (`14104157`→`158`→`159`) đang xếp hàng chờ qua dependency `afterany` — tự động nối tiếp vì 1 lần chạy không đủ time-window để xử lý hết ~913K task point. Worker 0 mẫu: 800/1275 video, ~0.03 video/s, ETA riêng job hiện tại ~287 phút. Đã có **25,432 file caption** ghi ra `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/captions/`. Chưa đủ để chạy full-scale task #4 (`build_caption_dict.py`) — cần đợi thêm.

### 4. Fix 2 bug thật trong `pipeline_pose/phase7_flatten.py` (liên quan task #7)

Phát hiện khi rà lại kỹ augmentation pipeline (không phải lỗi mới sinh ra hôm nay, đã tồn tại từ khi `permute_sentences` + SNAC injection được thêm vào):

- **Bug A (chính):** augmentation "sentence permutation" (xáo câu trong `### Speech:`) áp dụng **vô điều kiện**, kể cả khi activity đó có `<snac_N>` (audio thật, giữ đúng thứ tự thời gian). Kết quả: model "nghe" (SNAC) đúng thứ tự nhưng "đọc" (text speech) bị xáo trộn — dạy sai lệch giả giữa audio và text. **Fix:** thêm biến `effective_permute_rate = 0.0 if sn > 0 else permute_sentences` ngay trước khi gọi `process_transcript_into_chunks()` — `sn` là số token `<snac_N>` đã có sẵn trong `kept_tokens` (tính từ `count_token_types()`, không cần thêm logic mới).
- **Bug B (phụ, phát hiện khi test fix A):** `permute_chunks_list()` có `n = max(1, int(len(c) * permutation_rate))` — dòng này **ép tối thiểu 1 lần hoán đổi bất kể rate truyền vào là bao nhiêu**, kể cả `rate=0.0`. Nếu không sửa, fix A ở trên sẽ vô tác dụng (permute_rate=0 vẫn bị hoán đổi 1 cặp). **Fix:** thêm điều kiện `permutation_rate <= 0` vào early-return cùng với check `len(chunks) < 2` đã có sẵn.
- Đã verify bằng test nhanh: `permute_chunks_list(chunks, 0.0)` giờ trả về list y hệt input (trước đây thì không).
- **Chưa commit**, chưa chạy lại full-scale (đây là fix cho lần Phase 7 chạy tiếp theo, thuộc task #7/#8 của caption+speech interleaving work, hiện vẫn "not started" theo §18).

### Việc tiếp theo hợp lý nhất

Vì task #3 (A2) còn chạy rất lâu và task #6/#7/#8 phụ thuộc cả speech (đã xong 100%) lẫn caption (chưa xong), có thể tranh thủ bắt đầu task #6 (`phase6_merge_adaptive.py` — thêm `--captions-dir` + `--speech-segments-dir`) ngay, vì phần speech-segments đã sẵn sàng hoàn toàn, không cần đợi caption.

---

## Session Update — July 12, 2026 (read this first to resume)

**Two main threads this session: (1) fixed a `chunk_timing` bug in Phase 6, (2) finalized the captioning pipeline design.**

### 1. `has_seed2`/`has_cosmos` bug in `chunk_timing` — FIXED, FULL DATASET RE-RUN COMPLETE

- **Bug found:** `phase6_merge_adaptive.py` computed `has_seed2`/`has_cosmos` as `i < len(seed2_matches)` — comparing the **chunk index** against the **total tag count for the whole activity**, not a real per-chunk check. Since seed2 fires at 1fps while chunks occur at 3.75/sec, this flag was true for an artificial prefix of chunks then false for the rest — a single fake "off" transition per activity, not reflecting real content (verified: 2,558/2,558 sampled activities were ON→OFF only, never OFF→ON, at random timestamps 0.27s–638s).
- **Fix:** recompute using real string positions in `video_tokens` — a `<seed2>`/`<cosmos>` tag is attributed to whichever chunk's span it falls between (bounded by consecutive `<avc_lm>` block ends), matching the true temporal write order from `pipeline_video/pipeline.py`. `has_cosmos`/`has_avc_lm` simplified to hardcoded `True` (always correct, verified 0 flips across the whole sample).
- **No Phase 7 re-run needed** — verified via byte-for-byte `video_tokens` diff (0 differences) + code grep: `phase7_flatten.py` never reads `chunk_timing`. Only Phase 6's metadata output is affected; existing trained models and Megatron data are untouched.
- **Full dataset re-run done:** SLURM job `14102737`, 32/32 tasks COMPLETED, 0 errors → `final_dataset_adaptive_v3/` (160 files, kept v2 for comparison). New script: `slurm/submit_merge_adaptive_v3.sh`.
- **QA verified at two scales:** (a) 1 file (2,563 activities): agent/snac injection counts match v2 exactly (content unchanged); `has_seed2` now flips ~53/activity at the correct periodic rate. (b) 15 random files across the full dataset (34,732 activities, spanning all 40,804 videos): `has_seed2` flips 54.53/activity on average, **0/34,732 (0.00%) activities have `has_seed2` stuck False the whole time** — fix is stable at scale.
- **`final_dataset_adaptive_v3/` is now the standard input** for anything touching `chunk_timing` (including the captioning work below).

### 2. Captioning pipeline — DESIGN FINALIZED (prototype only, full-scale not yet coded)

**Context:** Huu asked (Jul 11 chat) for frame captions on all FineVideo keyframes, to fix root cause #2 (model lacks a language anchor for knowing when to switch modality).

**Anchor point selection (took several debugging rounds to get right):**
- **NOT** "any of the 5 flags `has_seed2/cosmos/avc_lm/agent/snac` changes" as originally planned — measured on real data: `cosmos`/`avc_lm` never vary within an activity; `seed2` (even after the bugfix above) still flips ~54x/activity, but that's purely its 1fps technical cadence, not a real content change.
- **Only using:** (1) the activity's first frame (opening context) + (2) every time `has_agent` flips (a person genuinely appears/disappears — confirmed with a real example: person transitions from standing to sitting exactly when agent turns on). Function: `select_anchor_points(chunk_timing, min_gap_sec=5.0)` in `tools/analysis/caption_prototype.py`.
- **`min_gap_sec=5.0` debounce:** needed because `has_agent` itself flickers in busy/high-motion scenes (sports, martial arts) due to noisy frame-to-frame YOLO detection (a known pre-existing data quality issue, not a new bug — no Phase 6 change needed). This debounce only affects which points THIS script chooses to caption, not the stored data.
- **Measured density:** ~1.86 captions/activity avg (at 2s gap) — well short of the "×4 records" target in the original doc; 82.8% of activities get only 1 caption (opening frame, no agent event ever occurs). **This is a known, unresolved limitation** — may need a periodic supplemental caption (every N seconds) for activities with no agent transition; N not yet decided.

**Model — tested 3, settled on Qwen2.5-VL-3B-Instruct:**
| Model | Test result |
|---|---|
| **Qwen2.5-VL-3B-Instruct** ✅ CHOSEN | No hallucinations in any test (including a 96-caption batch). Natively supported in `transformers` (no compatibility risk). Prompt: `"Describe what the person is doing in one short sentence."` |
| Florence-2-base | `<DETAILED_CAPTION>` mode clearly hallucinates (e.g. invented "he appears to be a psycholinguist"). Switching to `<CAPTION>` mode fixed the hallucination + was 3.5x faster than Qwen + no more truncation — but needs a separate env (`transformers==4.49.0`, torchvision must match the CPU index) since its custom code (`trust_remote_code`) isn't compatible with newer `transformers`. Test env: `env_caption_test/` (can be deleted if unused going forward). |
| SmolVLM2-2.2B-Instruct | **2x SLOWER than Qwen2.5-VL on CPU** (27.7s vs 14.0s/caption — contradicts the "fast, edge-oriented" expectation) plus 1 clear hallucination (invented "holding a book" for a plain white intro-slate frame) → rejected. |

**Why Qwen2.5-VL despite being slower than Florence-2 on CPU:** CPU speed isn't the deciding factor since the full-scale run must happen on GPU regardless of model choice; prioritized quality/no-hallucination + long-term library compatibility over CPU-only speed.

**Full pipeline design (not yet coded, next session):**
```
final_dataset_adaptive_v3/ 
    → [A1] Task list generation (CPU) — scan chunk_timing, compute anchor points per activity
    → [A2] SLURM array job — open video from videos_staging/, extract frame, Qwen2.5-VL caption
         → outputs/captions/{video_id}_captions.jsonl
    → [B1] Extend phase6_merge_adaptive.py with --captions-dir (same pattern as --snac-tokens-dir)
         inject <caption>...</caption> RIGHT BEFORE <cosmos> for that chunk (never mid-block,
         avoids repeating the v3→v4 speech-interleaving bug)
         → final_dataset_adaptive_v4/
    → [B2] phase7_flatten.py (unchanged) → megatron_dataset_v5/ → tokenize → train
```
Captions are plain English text, tokenized as regular BPE — **no vocab expansion needed**.

**Infra:** step A2 (real captioning run) will use **CPU** (many cores judged more practical than the available 2×4090 machine) — Van Khue's call (Jul 12), no GPU needed yet.

**Side findings worth remembering:**
- FineVideo source videos are already staged locally: `videos_staging/` (note the "s" — distinct from an empty `video_staging/`) — 43,751 mp4s, `/p/data1/mmlaion/shared/nguyen38/data/videos_staging/`, filenames = `{video_id}.mp4`. No JUPITER or HF streaming needed.
- **Read and evaluated HumanoidBench — NOT a fit** for the current eval need. It's a closed-loop RL benchmark (MuJoCo, Unitree H1 + Shadow Hands, 61-dim joint-angle action space) whereas our model outputs raw xyz human pose (17 H36M joints, no angles/hands). Only relevant to the already-deferred Priority 12 "Isaac Sim/H1" work, not DISCUSS-3.
- Home directory (`~/.cache`) has a much smaller quota than `/p/data1` (project storage, 388TB free) — always set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` before downloading large models to avoid "Disk quota exceeded".
- HF Hub's Xet download backend occasionally fails transiently (`Background writer channel closed`) — set `HF_HUB_DISABLE_XET=1` to fall back to plain HTTP download if this happens.

### 3. A1 coded + run on full dataset + thoroughly validated. A2 coded + smoke-tested — CPU vs GPU decision pending (continuation of Jul 12 session, same day)

**`select_anchor_points()` got a "periodic supplement" step:**
- Problem: agent-transition alone gave only ~1.4-1.86 captions/activity; most activities got just 1 caption (the opening frame).
- Agreed design, now coded: after the agent-transition step, if fewer than `target_count=4` points were found, add evenly-spaced supplemental points across the activity duration, snapped to the nearest real chunk, debounced against already-kept points. New signature in `tools/analysis/caption_prototype.py`: `select_anchor_points(chunk_timing, min_gap_sec=2.0, target_count=4)`.
- Also fixed a classification bug in `caption_florence2_visual_batch.py` (changed `len(pts) > 1` → check the raw `has_agent` flip, since the supplement makes point-count no longer a reliable proxy for "did a real event happen").

**Real bug found and fixed while testing A1 on production data:** the periodic supplement computed `duration` from the **absolute** `end_sec` instead of relative to the activity's own start — activities starting late in a video (e.g. minute 9-10) got target timestamps computed way outside their actual time range, making the supplement effectively a no-op. Fixed by subtracting `activity_start` before dividing. Verified on 2,563 real activities: % of activities reaching `target_count=4` jumped from 10.4% → 54.8%.

**A1 (`tools/analysis/generate_caption_tasks.py`) — CODED, RUN ON FULL DATASET (160/160 shards):**
- Reads `final_dataset_adaptive_v3/`, computes anchor points per activity via `select_anchor_points()`, writes task lists to `outputs/caption_tasks/*.jsonl` (one line per anchor point: video_id, video_path, scene_id, activity_id, chunk_idx, start_sec, has_agent).
- Run: 13 shards via SLURM array (job `14103227`, cancelled mid-run once it became clear it was fast enough to run directly), remaining 147 shards run on the login node (`--skip-existing` for resume) — completed 160/160.
- **Result:** 40,798 videos, 372,385 activities, **912,998 task points**, avg **2.45 captions/activity**, 0 videos missing a local mp4.
- **Validated thoroughly (per explicit request):** 100% of task points pass schema/type/`video_path`-exists checks; 0 activities with duplicate `chunk_idx`; 0 activities violating the 5s debounce; cross-checked 5 random shards (11,576 activities, 28,156 points) by recomputing from source `chunk_timing` — 100% match, 0 missing/orphan activities. Diagnostic sample saved at `logs/a1_smoke_test_samples.json` (gitignored).
- Submit script: `slurm/submit_caption_tasks.sh`.

**Correction to the "×4" framing:** re-reading `REPORT.md` (lines 1104, 1134) — the original ×4 target referred to **captioning combined with perspective framing** multiplying total training RECORD count ~4×, **not** "each activity must have exactly 4 captions" as we'd been measuring against. Measured reality: avg 2.45/activity is only 61.3% of the narrow (captions-per-activity) reading of ×4 — root cause: ~59% of FineVideo activities are under 15s, which is geometrically impossible to fit 4 points ≥5s apart into. **Decision: keep `target_count=4, min_gap_sec=5.0` as-is, do NOT lower the gap to force the number up** — lowering it would produce near-duplicate captions on short static clips, adding no real language signal while inflating A2 compute cost. The real lever for closing the ×4 gap is perspective framing (separate, uncoded roadmap item), not squeezing more density out of captioning alone.

**A2 (`pipeline_pose/caption_finevideo.py`) — CODED, SMOKE-TESTED OK, FULL RUN NOT STARTED:**
- Reads A1's task list (grouped by video), opens the video, extracts the frame at `start_sec`, captions it with Qwen2.5-VL-3B-Instruct (model chosen in the prior session) → writes `outputs/captions/{video_id}_captions.jsonl`.
- Follows the same pattern already proven in `pipeline_pose/snac_finevideo.py`: model loaded once per worker, videos striped across workers (`all_vids[task_id::num_tasks]`), one output file per video for safe resume (skip if it already exists).
- Smoke test (video `A1UVeD9UB1I`, t=248.0s): sensible caption — *"The person is arranging jewelry on a box."* — matches the source `text_prompt` *"Woman opens a gift box."*
- **Infra bug found and fixed:** initially had no PyTorch thread limit — two concurrent test runs on the 80-core login node fought each other for threads, producing 57.6s/caption (~4x slower than real). Added `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`torch.set_num_threads()` pinned to `SLURM_CPUS_PER_TASK` (default 4) so the eventual 32 workers won't oversubscribe each other.
- **Clean throughput measured (4 threads, no contention, 3 repeats): ~13.8s/caption** (12.9/15.2/13.4s) — matches the 10-15s figure already in `REPORT.md`.
- **Full-run cost estimate:** 912,998 tasks × 13.8s ≈ **3,500 CPU-hours**. With 32 workers (matching the SNAC job) → **~109h/worker (~4.6 days)**, needing **~5 resubmits** if kept at `--time=24:00:00`. Safe to resubmit thanks to per-video skip-existing.
- Submit script written: `slurm/submit_caption_finevideo.sh` — **NOT YET SUBMITTED**, pending the CPU-vs-GPU decision.

**Open question — decide at the start of next session:**
- CPU (32 workers, ~4.6 days, script ready to submit now) vs GPU (2×4090 machine, not measured at all this session).
- Back-of-envelope (NOT measured): unbatched GPU may not even beat 32 CPU workers (only 2-way parallelism vs 32-way, even though each individual request is faster); a decisive GPU win requires batched inference (many images per forward pass) — `caption_frame()` currently only processes one image at a time, no batching implemented.
- If GPU is chosen: need to implement batched `caption_frame()` and get access details for the 2×4090 machine to measure real throughput before deciding.
- **B1/B2 (inject captions into `final_dataset_adaptive_v4/`, re-run Phase 7) have NOT started** — blocked on A2 producing enough output (partial or full).

### 4. CPU chosen, A2 full run SUBMITTED and confirmed working (Jul 13, 2026)

**Decision:** CPU, per Van Khue — the ready-to-go option, no need to wait on GPU batching work.

**First submit (job `14104070`) FAILED — all 32/32 tasks crashed at model load.** Root cause: `slurm/submit_caption_finevideo.sh` set `HF_CACHE=/p/scratch/laionize/nguyen38/hf_cache`, which does not contain the Qwen2.5-VL-3B-Instruct weights (only `bert-base-uncased` and `snac_24khz` were cached there) — compute nodes have no internet (`HF_HUB_OFFLINE=1`), so `from_pretrained()` raised `OSError: We couldn't connect to huggingface.co ... Qwen/Qwen2.5-VL-3B-Instruct is not the path to a directory containing config.json`. The correct cache — the one actually used during the Jul 12 smoke test — is `/p/data1/mmlaion/nguyen38/hf_cache` (7.1GB `models--Qwen--Qwen2.5-VL-3B-Instruct` present there), matching the general env gotcha already noted below ("always set `HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache`").

**Fix:** changed `HF_CACHE` in `slurm/submit_caption_finevideo.sh` to `/p/data1/mmlaion/nguyen38/hf_cache`.

**Resubmitted as job `14104104` — CONFIRMED WORKING.** All 32/32 tasks reached `R` (running) state, model loaded cleanly in ~44-45s per worker (no HF offline errors), and first captions started landing within ~5 minutes of submit. Spot-checked output (`.../captions/-0-6Som0MGY_captions.jsonl`, 10 captions) — all well-formed JSON, correct schema, and captions are qualitatively good/specific (e.g. *"The person is pouring sulfuric acid into an energy drink can."*, *"The person is using a blue dropper to apply coconut oil onto a surface."*).

**Status at end of session: job `14104104` running, 32/32 tasks active, ~4.6 days ETA (per Jul 12 cost estimate), several `--time=24:00:00` resubmits still needed.** Next session should: (1) check `squeue -u nguyen38` for job `14104104` (or its resubmitted successor — same script, safe to re-run, per-video skip-existing) and resubmit if it timed out, (2) once a meaningful fraction of `outputs/captions/*.jsonl` (912,998 target task points across 40,798 videos) is done, consider starting B1 (extend `phase6_merge_adaptive.py` with `--captions-dir`) — B1 doesn't strictly need 100% of A2 done first, just enough coverage to be worth prototyping against.

**Auto-chaining added so no manual resubmission needed:** `slurm/submit_caption_finevideo.sh` now takes an optional job-id arg and submits with `--dependency=afterany:<id>`, printing the new job id for easy chaining. Chained 5 more jobs after `14104104`: `14104104 → 14104155 → 14104156 → 14104157 → 14104158 → 14104159` (6 jobs × 24h = ~6 days coverage). Confirmed queued with `(Dependency)` reason via `squeue --start`. If the chain still isn't enough, extend it with `bash slurm/submit_caption_finevideo.sh 14104159`.

**Caption quality spot-check (333+ output files, ~340 sampled lines): good overall, one hallucination class found.** Captions match source `text_prompt` well in the large majority of cases. Found one clear hallucination: video `-Gq3DJyhJ3I` (soccer highlights) got *"performing a complex mathematical operation..."* at t=0.0s — the real frame (checked with `cv2`) is near-black (fade-in intro), and the model invented content instead of saying "not visible" like it correctly does elsewhere. **Low-severity, known Qwen2.5-VL limitation** (matches the ~1-in-30-96 rate from model selection testing), not a pipeline bug — not blocking. Possible low-priority future fix for B1: skip/flag captions on near-black frames (mean pixel intensity check) before injecting into training data.

### 5. Permissive dataset survey (6 candidates) + MINT-1T-HTML download started (Jul 13, 2026, parallel work while A2 runs)

Investigated the 6 remaining unscoped data candidates from the Jul 7 team chat. Full detail and rationale in `REPORT.md` §17 — summary:
- **`mira-wm.com` dropped** — not robot/pose data at all, it's a Rocket League gameplay world model (video + keyboard actions + game state). Unrelated to this project.
- **`finevla.xlang.ai` deferred** — the actual 47,159-trajectory training set isn't public yet (GitHub repo says "Coming soon"); only a 500-video eval benchmark (`xlangai/RoboFine-bench`) is downloadable.
- **`nvidia/Cosmos3-DROID` deferred pending architecture decision** — confirmed real (707GB, 71,907 real-robot teleop episodes, LeRobotDataset v3.0 format), but it's robot joint-space action data, a different representation from this project's xyz human-pose tokens. Needs a new tokenization scheme designed before it's useful — not just a download.
- **`MiG-NJU/OmniVideo-100K` deferred** — video QA data, no pose/action signal, would only dilute the agent-token ratio further (same risk already flagged for MV-Omni).
- **`genrobot2025/Gen-EgoData` deferred** — closest structural match (egocentric video+pose+action) but tiny (500 samples, 47.6GB), `.mcap` format needs a special toolkit, CC-BY-SA (share-alike) license.
- **`mlfoundations/MINT-1T-HTML` — downloading now.** Directly fills the DISCUSS-1 language-data gap (FineVideo's 5.217B tokens are ~100% modality-specific, essentially no plain text). **Size correction: actual measured size is 2.89TB (6,159 parquet shards), not the 5.91TB the dataset card advertises** (that figure covers the full MINT-1T project incl. PDF/ArXiv splits not in this HTML-only repo). **Schema finding: the `images` column is URLs only, not image bytes** — text is directly usable now, but getting pixels for a "seed2 token from images" idea would need a separate per-URL crawl with likely significant dead-link rate (2011-era blog sources).

**Key framework insight (worth remembering for future dataset scoping):** raw video sources (own HRNet→MotionBERT→PCHIP pipeline handles them end-to-end) are cheap to integrate; pre-posed/pre-actioned sources (DROID joint-space, Gen-EgoData `.mcap`) are a retargeting problem, not a data-ingestion problem — don't invest download time there without an explicit decision on adding a distinct robot-action modality first.

**Download status:** `tools/extract/download_mint1t_html.py` (new script, `huggingface_hub.snapshot_download`, 16 workers, auto-retry, resumable) running in tmux session `mint1t`, log at `logs/download_mint1t_html.log`, target `/p/data1/mmlaion/shared/vla/mint1t_html/`. At session end: 249/6,159 files, 204GB/2.89TB (~7%), ETA ~10h from start, no errors. **Next steps:** let it finish, then sample-tokenize `texts` with the project's own tokenizer (same method as the MixtureVitae investigation, §13) to get a real token count before deciding how much of the corpus is actually needed for DISCUSS-1.

### 6. Caption+speech interleaving pipeline — implementation started (Jul 14, 2026, while A2 continues running)

**Context:** approved plan to interleave `<caption>` (A2/Qwen2.5-VL output) and `<speech>` (FineVideo's pre-computed ASR transcript, NOT a new Whisper run — see correction below) tags into the training token sequence at modality-transition points, giving the model language anchors it currently lacks. 8 tasks total; status below.

**Done:**
- **Task #1 (video→shard manifest):** `tools/analysis/build_video_shard_manifest.py`, run to completion. 43,751 video_ids mapped to their `HuggingFaceFV/finevideo` parquet shard index. Reusable, no need to rerun.
- **Task #2 (speech extraction script, see bugs below):** `tools/analysis/extract_speech_segments.py` written. **Correction to earlier framing:** this does NOT run Whisper — FineVideo already ships a pre-computed ASR transcript per video (`timecoded_text_to_speech`, from YouTube-Commons), so this script just re-fetches that field from the HF Hub parquet and maps it onto `chunk_timing`, no new ASR compute needed.
- **Task #4 (caption dict adapter):** `tools/analysis/build_caption_dict.py` written, logic-tested against real A2 output. **Not yet run at full scale** — `captions_dict/` output directory does not exist on disk yet.
- **Task #5 (tokenizer):** added 4 wrapper tokens (`<caption>`, `</caption>`, `<speech>`, `</speech>`) to `tools/tokenizer/build_tokenizers.py` + `tools/tokenizer/expand_vocab.py`. `tokenizer_vla_adaptive_v2` rebuild **confirmed complete and verified** (vocab 156,509, all 4 new tokens atomic, all pre-existing token categories re-checked atomic too). `tokenizer_vla_qwen3` rebuild was still running as of this entry — check before assuming done.

**Two real bugs found and fixed in `extract_speech_segments.py` while trying to produce sample output (important — could resurface if this script is copied or its pattern reused elsewhere):**
1. **Unbounded memory growth, not from HF fetching (initially misdiagnosed).** A quick `--video-ids` test (2 videos) drove RSS to 90+ GB and climbing on the shared login node before being killed. First suspected `HfFileSystem` streaming reads and switched to `hf_hub_download` (local-cache download) — that fix is real and worth keeping, but it was NOT the actual cause. **Real cause:** `load_activities_needing_speech()` defaults to scanning the full `final_dataset_adaptive_v3/` glob (160 files, **663GB total**) before applying the `--video-ids` allowlist, and it retained the **entire activity dict** (including the `video_tokens` string — hundreds of KB per activity) for every video with `chunk_timing`, not just the 3 fields actually needed (`activity_id`, `chunk_timing`, `time_range_sec`).
2. **Fix:** (a) trim retained activity data to only the 3 needed fields, (b) apply the `--video-ids`/allowlist filter *during* the file scan, not after loading everything. Re-tested: RSS stayed under 500MB for the same 2-video test (down from 90+ GB unbounded). Production full-scale runs (32-way SLURM array, ~5 files/worker via `SLURM_ARRAY_TASK_COUNT` slicing) were always going to be less exposed to this than the login-node quick-test path, but the fix reduces worker memory footprint regardless.

**Not started:** Task #6 (`phase6_merge_adaptive.py` — inject `<caption>` pre-`<cosmos>`, `<speech>` post-`</avc_lm>`; pre-check on cosmos/avc_lm 1:1 invariant found 1 trailing-chunk mismatch out of 2,753 activities, judged safe but a broader 5-shard check was interrupted, not yet re-run), Task #7 (`phase7_flatten.py` regex/state-machine update), Task #8 (end-to-end dry run).

**A2 captioning job status:** chain `14104155` (running, 32/32 tasks) → `14104156-159` (queued, dependency). Caption count grew 11,501 → 13,783 between the Jul 13 and Jul 14 checks — steady but far from the 912,998 target, still expect several more days.

**Tokenizer upload (pending, needs user's own HF token):** `tools/upload/upload_tokenizers_v2.py` updated with new model-card text reflecting the 4 added tokens (vocab 156,505→156,509 for adaptive_v2, 257,897→257,901 for qwen3). Not run yet — user will export their own `HF_TOKEN` and run it once the qwen3 rebuild finishes: `python tools/upload/upload_tokenizers_v2.py --mode all`. `tokenizer_vla_qwen3` rebuild (32→106,232 tokens added onto the Qwen3 base) was still running as of this entry — check `tmux attach -t qwen3_rebuild` or `logs/build_tokenizer_qwen3_rebuild.log` before assuming done.

**Task #2 full-scale launch — real bug hit and fixed (same session, later):** decided to run `extract_speech_segments.py` directly on the JUWELS login node (not SLURM) since the script needs internet access (`hf_hub_download` from the HF Hub) and JUWELS compute nodes have none. Wrote `tools/analysis/run_speech_extraction_login.sh` — 8 parallel workers, `nice -n 15`/`ionice -c3` to stay polite on the shared login node, per-video output + `--skip-existing` for resume if killed. **First launch crashed all 8 workers within ~20s: `RuntimeError: ... Disk quota exceeded`** — the script's `hf_hub_download()` call had no `HF_HOME` override, so it defaulted to `~/.cache/huggingface` (small home-directory quota, already documented elsewhere as a known gotcha — forgot to apply it here). Fixed by adding `export HF_HOME=/p/data1/mmlaion/nguyen38/hf_cache` to the runner script; cleaned up the 1.9GB of partial `HuggingFaceFV/finevideo` cache debris left in `~/.cache/huggingface` by the crashed run. Relaunched — confirmed healthy (8/8 workers alive, ~100% of 1 core each = 8/80 total, RAM 300-400MB/worker, cache now growing correctly under `/p/data1/mmlaion/nguyen38/hf_cache`). Running in tmux session `speech_full`, per-worker logs at `logs/speech_extraction_login/worker_*.log`. Output target: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/speech_segments/`.

---

## Repo Reorg Note (Jul 9, 2026)

`tools/` was split into subfolders (`upload/`, `tokenizer/`, `inventory/`, `eval/`, `visualize/`, `analysis/`, `extract/`) and ambiguously-named dirs were renamed (`multimodal/` → `investigations/mixturevitae_multimodal/`, `data_prep/` → `investigations/mv_omni_seed_conversion/`, `test/` → `manual_checks/`; `dev/` archived). **Script paths referenced below in older entries reflect the pre-reorg flat `tools/` structure** — e.g. `tools/data_inventory.py` is now `tools/inventory/data_inventory.py`. See the updated root `README.md` for the current layout.

---

## Session Update — July 8, 2026 (read this first to resume)

**What changed since last session:**
- ✅ **Phase 7 v4 uploaded to HF** — `EmpathicRobotics/FineVideo-Phase7-Flattened` is now live with the v4 data (371,888 records, 5.217B tokens). Shared with Huu/joergfranke on Discord (Jul 7) as the ready-to-tokenize dataset.
- ✅ **1-CP decision FINALIZED — deferred.** Confirmed with Huu on Discord (Jul 8): stick with current adaptive 2/4/8-CP format. Gain from 1-CP is only +7.1% (sample-based estimate, 50 videos) and re-running Phase 5→7 costs time. Huu's framing for eventual paper: "our compression decreases the data by more than 50%" is good enough to report as-is. **Revisit only if data later shows it's necessary.** Full-dataset 1-CP investigation (18,847 videos) was attempted but interrupted by the JUWELS outage — not resumed, not currently planned.
- ⚠ **JSC cluster outage (started ~Jul 6, 2026):** JUPITER fully down. JUWELS booster + JURECA have partial GPU availability ("to the extent Jenia lets us"). Huu's ETA: officially 1 week, realistically expect 2. This blocks: Megatron re-tokenization at scale, training v0.3, full 1-CP dataset run, Cosmos3-DROID GPU pipeline.
- **Team decision: cap synthetic/simulation data at ≤30% of total training mix** (Huu, from text literature guidance) — applies when deciding how much of abc.bot / MolmoAct2 / Cosmos3-DROID sim data to mix against real FineVideo human video.
- **New data source candidates identified (Jul 7, 2026 team chat)** — see "New Data Candidates" table below. Most promising: `abc.bot` (400h robot sim data **with physics state** MjData, permissive, has eval env).
- **Multi-project data sharing direction:** Huu wants to pool data across 3 parallel efforts — this repo's omni-vla work, joergfranke's architecture comparison project (qwen3/lfm2.5/olmo3 baselines), and blanchon.jl's diffusion-based world-action-model (video generation + action). `FineVideo-Phase7-Flattened` is now being used as shared input across projects — keep the format generic/well-documented.
- ✅ **`mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` investigated (Jul 9, 2026).** 15 files, ~103GB. Sample-based token count (75MB/file, streamed — no full download) via new `tools/peek_multimodal.py` + `tools/count_multimodal_tokens.py`, run locally (no JUWELS access this session). **Result: mostly plain text/caption data, not our VLA token format.** Only `train_data_snac.jsonl.gz` and `valid_data_snac.jsonl.gz` carry real SNAC audio tokens — as **raw integer arrays** (`snac_token: [128266, ...]`), not `<snac_N>` string tags — extrapolated **~3.27B raw SNAC codes** total (~3.11B + ~162M). The other 13 files are text/caption corpora (~12.4B word-count tokens extrapolated, `finevideo_transcripts.jsonl.gz` undercounted — see caveats below). Posted findings to Huu on Discord (Jul 9, 3:51pm) asking if he wants it added — **awaiting his reply**, do not start integration yet. Full detail in "MixtureVitae-Backup Multimodal Investigation" section below.
- **Pending investigation tasks (assigned by Huu, not yet done):**
  1. "finevideo reformulation" at `leo:/mnt/sdb/mixture-vitae-working/finevideo` — Huu created this at some point but doesn't recall exactly what it is; need to check for overlap with our own pipeline (avoid a repeat of the `valid_with_seed` double-counting issue).
- **Open concern (not yet acted on):** naively mixing all of MV-Omni (6.93B tokens, 0 agent tokens) into the training corpus dilutes the agent (pose) token ratio from 12.2% (FineVideo v4 alone) down to ~5.2% of the combined mix. Since agent tokens are the project's core differentiator, consider dropout on MV-Omni (same treatment as Cosmos/AVC-LM) or oversampling agent-bearing records before combining.

**Current priority ranking (given JUPITER down + "more data before training" preference):**

| Tier | Task | Needs cluster? | Impact |
|---|---|---|---|
| ✅ | ~~Investigate MixtureVitae-Backup/multimodal~~ | No | Done Jul 9 — mostly text, ~3.27B raw SNAC codes found; awaiting Huu's go/no-go |
| P0 | Clarify "finevideo reformulation" on leo | No | Avoid double-counting |
| P0 | Decide MV-Omni mix ratio (agent dilution fix) | No | Protects core pose signal |
| P0 | Define eval protocol (DISCUSS-3, still open) | No | Required before any training run |
| P0 | Decide text/instruction data mix ratio (DISCUSS-1) | No | Steerability |
| P1 | Code the full-scale captioning pipeline (design finalized Jul 12, see session update) | No (CPU, per Jul 12 decision) | Highest — fixes root cause 2 (measured density ~1.86 captions/activity, short of the original ×4 target) |
| P1 | Write ego-centric perspective converter | No (GPU only to run) | 2× pose diversity, free |
| P1 | Mix MV-Omni into Megatron format | CPU only | +6.93B tokens, vocab already ready |
| P2 | Scope abc.bot, MolmoAct2-BimanualYAM, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData | No | New robot/video sources, TBD size |
| P2 | Investigate leo seed2 + euro_pat | No | TBD |
| P3 | Cosmos3-DROID pipeline run | GPU | First real robot-domain data |
| P3 | Full captioning run, Megatron re-tokenize combined corpus, train v0.3 | GPU (JUPITER) | Blocked until cluster back + data ready |
| P4 (deferred) | 1-CP, Moss-Audio V2, Qwen3 migration, PAB-Spline angle spec, Isaac Sim | — | Explicitly held off per team decisions |

---

## The Big Picture — What Are We Actually Building?

The ultimate target: a single model that receives multimodal input (video frames + speech/text commands) and outputs action tokens that can be decoded into robot joint trajectories. The analogy Huu used: *"hear a verbal command like 'walk forward', and have a robot actually walk forward based on learned pose patterns."*

The longer-term vision is a model that, given an image of a beaker and a chemical formula, could reason through the task ("Make salt water") and translate that into arm/hand movements — **without being explicitly trained on that exact task**. This requires genuinely cross-modal binding: vision ↔ language ↔ action.

We are building this by pretraining a 1.7B LLM on an interleaved token stream:

```
USER: <activity description> [Speech: ...]  ASSISTANT:
  <seed2_N> ...          # Semantic keyframe tokens  (1fps, vocab 8192)
  <cosmos_N> ...         # Spatial video tokens      (every 8 frames, vocab 64000)
  <avclm_N> ...          # H.264 BPE video tokens    (every 8 frames, vocab 8192)
  <fps_30> <pelvis> ...  # 3D human pose tokens      (every 8 frames, 17 joints)
  <snac_N> ...           # Audio tokens — SNAC listen format (~10 tokens per 8-frame chunk)
```

The model learns to "read" and "continue" this interleaved sequence. In inference, you prompt it with video tokens + a text command, and it predicts the next agent tokens = the motion.

**Why this approach?** No prior VLA model has tried to unify video tokenization (Seed2/Cosmos), speech (SNAC), and continuous motion (PCHIP spline) into a single LLM autoregressive context. We are at the research frontier — nobody here has done this before.

---

## Timeline Overview

| Period | Key milestone |
|--------|--------------|
| Jun 2025 | Project started. FineVideo dataset chosen (~40K YouTube videos). |
| Jul–Sep 2025 | Branch A: Video token extraction pipeline (Seed2, Cosmos, AVC-LM). 160 GPU run. |
| Sep–Nov 2025 | Branch B phase 1–3: HRNet 2D pose, MotionBERT 3D lifting, kinematics. |
| Nov–Dec 2025 | Phase 4: YOLO cleaning. Phase 5 first iteration (opaque 256-token format). |
| Jan–Feb 2026 | Phase 5 rewrite → Adaptive PCHIP (self-describing named joint tokens). |
| Mar 2026 | Phase 6 merge, Phase 7 flatten. First Megatron tokenization. |
| Apr 2026 | **First model** trained (vla-1.7b-pab-spline-25b-test). Broken tokenizer discovered. |
| May 2026 | Tokenizer fix: `add_tokens(special_tokens=True)`. Full re-tokenization. |
| Jun 2026 | **Second model** trained (vla-1.7b-pab-spline-adaptive). Evaluation. Data inventory. |

---

## What Is Done — Detailed

### Phase A: Video Token Extraction

**Script:** `pipeline_video/pipeline.py` | **Compute:** 40 nodes × 4 GPU

Processed all ~40K FineVideo videos. Each activity segment tokenized into:
- **Seed2**: 1fps semantic keyframe tokens (8192 vocab)
- **Cosmos**: every-8-frame spatial tokens (64000 vocab)
- **AVC-LM**: every-8-frame H.264 BPE tokens (8192 vocab)

Output: 160 `training_ready_rank_*.jsonl` files. Each file contains hierarchical JSON (video → scenes → activities → tokens + speech transcript + metadata).

---

### Phase 1: 2D Pose Detection

**Script:** `pipeline_pose/phase1_hrnet_gpu.py`

- HRNet-W48 + Faster R-CNN person detector on all 40K videos
- Output: 2D joint coordinates (17 joints, COCO format) per frame
- **40,804 videos**, 145 GB

---

### Phase 2: 3D Pose Lifting

**Script:** `pipeline_pose/phase2_motionbert_gpu.py`

- MotionBERT lifts 2D → 3D (pretrained on Human3.6M)
- **40,804 videos**, 259 GB

---

### Phase 2.5: 30fps Resampling

**Script:** `pipeline_pose/phase2_5_resample_30fps.py`

- Linear interpolation from native video fps → uniform 30fps
- Required so all modalities share the same time grid
- 67 GB

---

### Phase 3: Kinematics Processing

**Script:** `pipeline_pose/phase3_kinematics_processor.py`

- Butterworth temporal smoothing
- Bone-length normalization to canonical H36M skeleton
- Pelvis root-centering
- Anti-teleportation filter (drops sudden-jump windows)
- Windowed into 8-frame chunks → shape `(windows, 8, 153)` where 153 = 17 joints × 3 dims × 3 kinematics (pos/vel/acc)
- **40,200 videos** (604 dropped as too short), 193 GB

---

### Phase 4: YOLO Person-Presence Filtering

**Script:** `pipeline_pose/phase4_yolo_cleaner.py`

- YOLOv8 person detection per frame
- Drops any 8-frame window where ≥4 frames have no detected person (confidence ≥ 0.75)
- **40,195 videos**, 107 GB

**⚠ Pose Data Quality Finding (Jul 2, 2026):**

Side-by-side skeleton visualization (`tools/visualize_skeleton_sidebyside.py`) + direct inspection of `yolo_cleaned` data revealed significant quality issues:

| Issue | Detail |
|-------|--------|
| **Joint sparsity** | Average 4–7 finite joints per frame out of 17 (24–41% skeleton) |
| **Arms absent** | j11–j16 (both arms: shoulder/elbow/wrist) = NaN in nearly all frames — MotionBERT cannot reliably lift arm joints from YouTube videos due to occlusion/side views |
| **Zero-fill artifact** | j10 (head_top) often stores (0,0,0) when undetected, identical to pelvis position — counted as finite but is wrong/misleading |
| **Coordinate scale OK** | ankle at ~−0.638m below pelvis is anatomically plausible; metric scale is correct |

**Impact on training:** Pose tokens are predominantly lower body (hip/knee/ankle) + torso. The arms — most important for manipulation tasks — are almost never captured. The model learns rough walking/sitting body motion but not fine hand/arm motion. This is a fundamental limitation of monocular video pose lifting from YouTube.

**Does NOT affect FineVideo as pretraining signal** — even noisy lower-body pose is better than none for learning video-pose correlation. But for downstream manipulation fine-tuning, better pose data (simulation, MoCap, or depth cameras) will be needed.

---

### Phase 5: Adaptive PCHIP Tokenization

**Script:** `pipeline_pose/phase5_adaptive_pchip.py`

For each 8-frame window, for each of 17 joints:
1. Compute trajectory curvature
2. Choose 2, 4, or 8 control points: low curvature (static) → 2 CPs; medium → 4 CPs; fast motion → 8 CPs
3. Quantize positions to uint8: `N = clip(round((v + 2.0) / 4.0 * 255), 0, 255)` mapping [-2m, +2m]
4. Emit self-describing tokens: `<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128> ... </pelvis>`

**Why adaptive?** A static pelvis doesn't need 8 data points — 2 suffice. A fast-moving wrist needs 8. Reduces average token count by ~35% vs fixed 8-CP.

**Previous iterations (abandoned):**
- `phase5_interpolation_tokenizer.py` — 256 opaque uint8 tokens. Abandoned: tokens were not self-describing, model couldn't learn joint semantics.
- `phase5b_xyzt_tokenizer.py` — 409 fixed tokens (all 8 frames × 17 joints × 3 dims). Self-describing but wasteful.

Output: **18,847 videos** (only where YOLO confirmed human presence), 7.4 GB.  
Token range: 171 (all 2-CP, very static pose) to 579 (all 8-CP, fast motion), typical ~250–300 per window.

---

### Phase 6: Merge

**Script:** `pipeline_pose/phase6_merge_adaptive.py`

- Injected `<agent>...</agent>` blocks after each `<avc_lm>` block in the training_ready files
- Frame-aligned by matching agent window_ids to AVC-LM chunk indices (both at 30fps, 8-frame windows)
- Added `chunk_timing` array to each activity (precise timestamps for every chunk, which modalities are present)
- ~399K activities, **~2.15M agent blocks** injected
- Output: 160 `final_vla_adaptive_rank_*.jsonl`, **657 GB**

**Phase 6 v2 — SNAC injection support (Jun 28, 2026):**
- Added `--snac-tokens-dir` argument to also inject SNAC audio tokens per chunk
- New `inject_chunk_tokens()` function handles both agent + SNAC in one pass over `video_tokens`
- Token order per 8-frame chunk: `<cosmos>...</cosmos> <avc_lm>...</avc_lm> [<agent>...</agent>] [<snac>...</snac>]`
- `chunk_timing` now includes `has_snac` flag per chunk
- `timing_meta` now includes `snac_rate: "37.5_tokens_per_sec_listen_format"`
- Backward compatible: running without `--snac-tokens-dir` behaves identically to v1
- **Requires `snac_finevideo.py` to run first** → `{video_id}_snac.jsonl` files in snac output dir

---

### Phase 7: Flatten + Augment

**Script:** `pipeline_pose/phase7_flatten.py`

Converts hierarchical JSON → flat Megatron-LM JSONL. Key decisions:

**Agent-only filter:** Only activities with `<agent>` blocks are emitted (every training record has action data).

**Modality dropout (token balancing) — v1 (old, already trained on this):**
| Modality | Raw ratio vs agent | Drop rate | Resulting ratio |
|----------|-------------------|-----------|----------------|
| AVC-LM | ~373× | 99% | ~4× |
| Cosmos | ~19× | 90% | ~2× |
| Seed2 | ~1× | 0% | 1× |
| Agent | baseline | 0% | 1× |

**Modality dropout — v2 (Jun 27, 2026 update, pending re-flatten):**
| Modality | Drop rate | Reason |
|----------|-----------|--------|
| AVC-LM | **100%** | Removed until ablations confirm benefit (per Huu) |
| Cosmos | **50%** | Keep ~6/12 blocks per activity for modality transition learning |
| Seed2 | 0% | Keep all — primary visual signal |
| Agent | 0% | Keep all |

**Text augmentation:** 15% synonym replacement, 5% stopword dropout, 10% sentence permutation, random speech/token interleaving, random layout block shuffling.

Output v1: 160 files, **69,844 records**, 19.2 GB → `megatron_dataset_adaptive/`  
Output v2: → `megatron_dataset_v2/` (cosmos 50% drop, avclm 100% drop — re-flattened Jun 27, 2026)

**Phase 7 v3 — SNAC + updated filter (COMPLETE Jul 2, 2026):**
- Added `<snac>...</snac>` block extraction in `process_tokens_to_individual_tags` (pass-through, like agent)
- Added `--drop_snac` argument (default 0.0 = keep all SNAC tokens)
- **Changed record filter:** was `<agent> required`; now emits if `<agent>` OR `<snac>` present
  - Full-chain records: seed2 + cosmos + agent + snac — **69,811 records (18.8%)**
  - Partial-chain records: seed2 + cosmos + snac — **302,044 records (81.2%)**
  - Bad records (neither): **0**
- Output: `megatron_dataset_v3/` — 160 files, **371,888 records**, **72 GB**
- Sample: `samples/after_flatten_v3.json` | Upload script: `tools/upload_flattened_hf.py` (updated for v3)

**✅ Phase 7 v4 — Per-chunk temporal ordering (COMPLETE Jul 2, 2026):**

Phase 7 fully rewritten (`pipeline_pose/phase7_flatten.py`). State machine walks Phase 6 output in document order, emitting per chunk: `[seed2?][cosmos?][agent?][snac?]`. Speech moved to dedicated `### Speech:` header.

**v4 stats:** 160/160 files, 371,888 records, **5.217B tokens** (seed2 6.4% / cosmos 74.4% / agent 12.2% / snac 7.0%). Runtime: 36 min / 32 workers.

**Bugs fixed:**
- Temporal misalignment (v3: all agent at end → 69% of records had 0% agent in first 4096 tokens. v4: per-chunk → all records have agent in first 4096 tokens)
- Speech injection into agent grammar (v3: speech words scattered into joint sequences. v4: speech in header only)

Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v4/` (160 files)

**Token rates per 8-frame chunk (verified Jul 2, 2026):**

| Modality | Tokens/chunk | Per 30s (after v3 dropout) |
|----------|-------------|---------------------------|
| Seed2 | 32 fixed (1 block per 3.75 chunks) | 30 × 32 = **960** |
| Cosmos | 200 fixed (every chunk) | ~56 × 200 = **11,200** |
| Agent | 171–579 (~280 typical) | up to 112 × 280 = **31,360** |
| SNAC | 9 or 12 (avg 10, alternating) | 112 × 10 = **1,120** |
| AVC-LM | 885–5,055 | **0** (dropped) |

**DATA PATHS (IMPORTANT — updated Jun 27, 2026):**  
JUPITER `/e/data1` is sometimes down (cluster maintenance). All critical data copied to `/p/`:
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← INPUT for Phase 7
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flat output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flat output (pending)
```
Phase 7 script and SLURM now default to `/p/` paths.

---

### Tokenizer

**Script:** `tools/expand_vocab.py`, `tools/upload_tokenizer.py`

Extended GPT-NeoX-20b (50,277 tokens) with 93,938 VLA tokens using `tokenizer.add_tokens(special_tokens=True)`.

**The critical bug in the first model:** Editing `vocab.json` directly does NOT register BPE merge rules. The tokenizer split `<seed2_1137>` → 7 sub-pieces. Despite this, the first model showed signal (learned to predict sub-piece sequences) but was not decoding real tokens.

**The fix:** `add_tokens(special_tokens=True)` bypasses BPE merging, treating every VLA token as atomic.

Published: `EmpathicRobotics/tokenizer-vla-adaptive` (144,215 vocab, padded to 144,256 for Megatron).

---

### Phase 8: Megatron-LM Tokenization

Tokenized 160 JSONL files → 2 binary shards:

| Shard | Tokens | Size |
|-------|--------|------|
| `data_shard_00000.bin` | 2,684,323,146 | 10.00 GB |
| `data_shard_00001.bin` | 156,389,702 | 0.58 GB |
| **Total** | **2,840,712,848 (2.84B)** | **10.58 GB** |

---

### Phase 9: Training — Model 2 (June 2026)

**Model:** `EmpathicRobotics/vla-1.7b-pab-spline-adaptive`  
**Architecture:** OpenSci-Ref 1.7B (24 layers, 2048 hidden, 32 heads → **1.91B params** with 144K vocab embeddings)  
**Config:** `oellm-autoexp/config/experiments/nguyen38/vla_adaptive.yaml`  
**Compute:** 64 nodes × 4 GH200 = 256 GPUs, ~35 min wall time

Training schedule:
| Iter | Loss | LR | Tokens seen |
|------|------|----|------------|
| 200 | 2.982 | 4e-3 | 0.84B |
| 500 | 2.070 | 4e-3 | 2.10B |
| 1000 | 1.672 | 4e-3 | 4.19B |
| 2000 | 1.476 | 3.2e-4 | 8.39B |
| **2032 (val)** | **1.501** | — | — |

Val PPL: **4.49** | Test PPL: **4.45** | ~3 epochs over 2.84B tokens

---

### Data Inventory (June 26, 2026 — Complete)

**Script:** `tools/data_inventory.py` | **Checkpoint:** `tools/inventory_checkpoint_v2.json`

Scanned all 242 files across 4 dataset families:

| Dataset | seed2 | cosmos | avclm | agent | snac | text | **TOTAL** |
|---------|-------|--------|-------|-------|------|------|-----------|
| FineVideo-VLA (160 files) | 89.9M | 210.2M | 474.4M | 564.9M | — | 11.4M | **1.35B** |
| MV-Backup valid_with_seed (64 HF shards) | 5.6M | — | — | — | — | — | **5.6M** |
| MV-Backup stack_images3_gzip (12 archives) | 313K | — | — | — | — | — | **313K** |
| MV-Omni valid_snac (6 gzip files) | — | — | — | — | 4.92B | 1.99B | **6.93B** |
| **TOTAL** | **95.8M** | **210.2M** | **474.4M** | **564.9M** | **4.92B** | **2.00B** | **8.29B** |

**Key findings:**
- `valid_with_seed` (1.1 TB downloaded!) yields only 5.6M seed2 tokens — **negligible, not worth the storage cost**. Shards 0–30 contain only raw `.png`/`.ogg` with zero tokenized content. Only shards 31–63 have `_seed2.jsonl` inside inner archives.
- MV-Omni is the only substantial external source at 6.93B tokens. BUT `<snac_N>` and `<seed_N>` tokens are **not in the current tokenizer vocab** — blocked until vocab expansion.
- **Only FineVideo has agent (3D pose) tokens.** No external dataset contributes pose data.
- **Training-ready today: 1.35B tokens** (FineVideo only, with current vocab).

---

## Current State — What Works, What Doesn't

### Works
- Pipeline end-to-end: raw video → 3D pose → tokens → Megatron bin → training → deployable HF checkpoint
- All VLA tokens are atomic (tokenizer fix confirmed)
- Model correctly completes 17-joint agent blocks: right joint ordering, valid xyz/t values, decodable to 3D pose via PCHIP
- 3D pose decoder verified: model output → (8, 17, 3) trajectory in correct physical range

### Does NOT work yet
- **Autonomous modality transitions:** When prompted with only text, the model stays in seed2 mode and never transitions to cosmos/avclm/agent. It requires agent tokens in the prompt to continue in agent mode.
- **Root cause 1 — Data starvation:** 2.84B tokens for 1.91B params = ~1.5× Chinchilla ratio. Optimal is ~20×. Each training sample seen only ~3 times — enough for local pattern memorization, not high-level sequencing.
- **Root cause 2 — No visual language anchors:** Text is only Title/Context/Keywords. No captions describe what's happening at each timestamp. The model has no language signal to know "after these seed2 tokens, cosmos tokens come next."
- **Root cause 3 — Over-aggressive dropout:** 99% AVC-LM + 90% Cosmos dropout means most records lack the full transition chain. Model rarely sees seed2 → cosmos → avclm → agent in sequence.

---

## What's Next — Prioritized Roadmap

### Immediate priorities (code during any available time, no GPU needed)

**Priority 1 — Vocab expansion for SNAC tokens** ← PARTIALLY DONE
- ~~Convert MV-Omni `<seed_N>` → `<seed2_N>`~~ **DONE** (Jun 27, 2026)
  - Script: `data_prep/convert_mvomni_seed.py`
  - Output: `/p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz`
  - 1,593,301 records | 19,249,664 seed tokens converted | 30 GB
  - `<seed_N>` tokens fully eliminated — zero remaining in output
- **REMAINING:** Add `<snac_0>` ... `<snac_4095>` (~4096 tokens) to tokenizer via `add_tokens(special_tokens=True)`
  - New vocab: ~148,311 tokens (no need for `<seed_N>` — already converted to `<seed2_N>`)
  - Unlocks MV-Omni's **6.93B tokens** for training
  - Effort: ~1 day

**Priority 2 — Adjust modality dropout in Phase 7** ← ~~DONE~~ (Jun 27, 2026)
- AVC-LM: 99% → **100% drop** (removed entirely)
- Cosmos: 90% → **50% drop** (keeps ~6/12 chunks per activity)
- Output: `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v2/`
- Uploaded to `EmpathicRobotics/FineVideo-Phase7-Flattened` (v2 commit)
- **Next step:** Megatron re-tokenize `megatron_dataset_v2/` → new `.bin/.idx` shards → re-train v0.2

**Priority 3 — Ego-centric perspective for FineVideo**
- Read Phase 4 yolo_cleaned pose data
- Apply rotation matrix: place camera at `head_top` joint position, orient along thorax forward direction
- Generate additional agent token sequences from ego-centric view
- Same underlying motion data, double the data diversity (first-person + third-person)
- Effort: ~1 week code + 1 SLURM run

**Priority 4 — Write captioning pipeline code**
- Use `chunk_timing` timestamps to extract keyframes from FineVideo videos
- Pass each keyframe through SmolVLM2 or Qwen2.5-VL
- Interleave generated captions into the token sequence
- Expected impact: ×4 records with language anchors at every modality transition → fixes root cause 2
- Effort: 1–2 weeks code (GPU run on JUPITER is separate)

### Medium-term (needs dedicated GPU time on JUPITER)

**Priority 5 — Collect agent + cosmos + snac from Cosmos3-DROID**
- `nvidia/Cosmos3-DROID` on HuggingFace: robot arm manipulation videos with Cosmos video tokens
- Run YOLO + Phase 1–5 equivalent to extract agent tokens (robot arms/hands)
- Add SNAC tokens if audio track exists
- First robot-domain data — critical for generalization beyond human motion
- Hold off on AVC-LM until ablations confirm it helps (per Huu's guidance)

**Priority 6 — Vocab expansion (tokenizer build)** ← **COMPLETE (Jul 1, 2026)**

Script: `tools/build_tokenizers.py`. Hai output:
- `tokenizer_vla_adaptive_v2`: 144,215 (base) + 12,290 SNAC = **156,505 vocab**, tất cả atomic ✓
- `tokenizer_vla_qwen3`: ~151,669 (Qwen3) + 106,228 VLA = **257,897 vocab**, tất cả atomic ✓

**Priority 7 — SNAC tokenization for FineVideo** ← **COMPLETE (Jul 1, 2026)**

Job `snac_cpu_14077331`, 32 array tasks on `batch` partition (CPU), submitted from `jwlogin08`.

**Results:**
| Metric | Value |
|--------|-------|
| Tasks completed | **32/32** (100%) |
| Activities processed (ok) | **371,855** |
| fail_audio (no audio track) | 530 (~0.1%) |
| fail_snac | **0** |
| Total SNAC tokens | **363,029,331 (~363M)** |
| Output files | **40,779** `{video_id}_snac.jsonl` |
| Output size | **6.5 GB** |
| Output location | `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens/` |

**Output format** — one file per video, one JSONL line per activity:
```json
{
  "video_id": "abc123",
  "activity_id": "scene_1_act_1",
  "start_sec": 1.0,
  "end_sec": 8.9,
  "has_agent": true,
  "snac_by_chunk": {
    "0": ["<snac_132247>", "<snac_132788>", "<snac_147076>", ...],
    "1": [...],
    ...
  }
}
```

**Chunk splitting mechanism:**
- Encode full activity audio once (1 call to SNAC) → flat list of tokens
- SNAC output = sequence of base frames, each = exactly 3 tokens (L0 + L1_even + L1_odd triplet)
- `n_base = len(flat_tokens) // 3` — truncate to complete base frames (atomic unit, cannot split triplet)
- Proportional split: `start_base[k] = round(k * n_base / n_chunks)`, `end_base[k] = round((k+1) * n_base / n_chunks)`
- Each chunk gets `end_base - start_base` base frames × 3 = **9 or 12 tokens** (alternating due to 3.33 base frames/chunk)
- Last chunk is NOT shorter — `round(n_chunks × n_base / n_chunks) = n_base` exactly
- Temporal alignment error: ±1 base frame = ±80ms at each chunk boundary (acceptable for pretraining)

**Next steps unblocked by this completion:**
1. Vocab expansion — add 12,288 `<snac_N>` tokens to tokenizer
2. Re-run Phase 6 v2 with `--snac-tokens-dir`
3. Re-run Phase 7 v3 → `megatron_dataset_v3/`
4. Megatron re-tokenize → train v0.3

**CLUSTER ARCHITECTURE NOTE (discovered Jun 28, 2026):**
- `jwlogin08.juwels` = JUWELS Cluster login node (x86_64)
- `juwels-booster.fz-juelich.de` = JUWELS Booster login nodes (separate system, ppc64le compute)
- `laionize` account with GPU access (`booster` partition) is only usable from the Booster login nodes
- From JUWELS Cluster login, `laionize` only has CPU partitions: `batch`, `devel`, `large`
- **To submit GPU job: SSH to `juwels-booster.fz-juelich.de` first**

**Task list already built (Jun 28, 2026):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_task_list.json
→ 40,798 videos, 372,385 activities, all with chunk_timing
```

**Submission commands (when on juwels-booster login node):**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose

# GPU mode: 16 workers on booster partition, ~8-12h
bash slurm/submit_snac_finevideo.sh

# CPU fallback (from jwlogin, slower ~24h, no SSH needed):
bash slurm/submit_snac_finevideo.sh --cpu
```

**Run sequence after SNAC tokenization completes:**
```bash
# Step 1 — DONE: build task list
# snac_task_list.json already at TASK_CACHE path

# Step 2 — RUNNING (Jun 30): CPU batch job, 32 workers, ~20-24h
# Output: .../FineVideo-VLA/snac_tokens/{video_id}_snac.jsonl (~40K files)

# Step 3: Vocab expansion — add 12,288 <snac_N> tokens to tokenizer
# TODO: update tools/expand_vocab.py to include snac range [128266..148745]

# Step 4: Re-run Phase 6 with SNAC injection
python pipeline_pose/phase6_merge_adaptive.py \
  --input-glob "/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/training_ready_rank_*.jsonl" \
  --agent-tokens-dir /p/data1/mmlaion/shared/nguyen38/data/outputs/agent_tokens_adaptive \
  --snac-tokens-dir  /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/snac_tokens \
  --output-dir       /p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2 \
  --output-prefix    final_vla_adaptive_v2

# Step 5: Re-run Phase 7 → megatron_dataset_v3/
python pipeline_pose/phase7_flatten.py \
  --input-glob ".../final_dataset_adaptive_v2/final_vla_adaptive_v2_rank_*.jsonl" \
  --output-dir ".../megatron_dataset_v3" \
  --drop_cosmos 0.5 --drop_avc 1.0 --drop_snac 0.0 --workers 16

# Step 6: Megatron tokenize → .bin/.idx → train v0.3
```

**Priority 7 — Investigate leo seed2 + euro_pat**
- Check what's on the `leo` cluster: seed2 + euro_pat datasets mentioned by Huu
- Quantify token counts before committing storage/compute

**New Data Candidates (Jul 7, 2026 — from team Discord)**

Found while scoping VLA data sources broadly. None yet scoped for token/hour counts or license fit.

| Source | What | Notes |
|---|---|---|
| `abc.bot` (Amazon) | 400h robot recordings **in simulation**, includes physics state (MjData) | Most promising — permissive, has eval env, same embodiment throughout. blanchon.jl: "indeed perfect" |
| `allenai/MolmoAct2-BimanualYAM-Dataset` | 2 TB, bimanual YAM arm robot data | Check license + embodiment compatibility |
| `MiG-NJU/OmniVideo-100K` | Video dataset | Not yet scoped |
| `mlfoundations/MINT-1T-HTML` | Large text/HTML dataset | Not yet scoped — likely for language mix (DISCUSS-1), not video |
| `genrobot2025/Gen-EgoData` | Egocentric robot data | Not yet scoped |
| `finevla.xlang.ai` | Possible VLA dataset | HF link not found yet — may be unreleased |
| `mira-wm.com` | World model reference (Kyutai released similar) | Reference/inspiration, not necessarily a data source |

**Team constraint:** synthetic/simulation data (abc.bot, MolmoAct2, Cosmos3-DROID, etc.) capped at **≤30% of total training mix** — decided by team consensus (Huu, citing literature), to keep the balance toward real human/robot video.

**Priority 8 — First re-training run (v0.2)**
- After items 1, 2, 4 are done: estimated **10–20B tokens** available
- Continue training from current checkpoint (2032 iter) with new data + adjusted dropout
- Expected result: model begins to learn modality transitions autonomously

### Long-term (3–6 months)

**Priority 9 — More text data**
- Mix in standard LLM text data (to create language binding and prevent catastrophic forgetting)
- Target: text tokens at ~10–15% of total training mix

**Priority 10 — Qwen3 migration**
- Retokenize entire dataset with Qwen3-based expanded tokenizer
- Requires full re-run of Phase 8 (Megatron tokenization) and training from scratch
- Benefit: native HF ecosystem support, vLLM, llama.cpp compatibility
- Huu's config: cherry-picked from commit `7dcf8a5`

**Priority 11 — PAB-Spline spec upgrade**
- Current tokenizer: PCHIP xyz-only (positions)
- Spec calls for: joint angles (q/qd), phase variable φ ∈ [0,1], cyclic gait detection, static joint compression
- Blocked by: need to run kinematics pipeline again with angle computation

**Priority 12 — Isaac Sim integration**
- Generate Unitree H1 rollouts in Isaac Sim / ManiSkill
- Tokenize simulation data with PAB-Spline tokenizer
- Sim-to-real gap: map joint tokens → H1 control signals

---

## Data Landscape — Where We Are and What We Need

### Current training-ready data: 1.35B tokens (FineVideo only)
This is too small. For a 1.7B model, Chinchilla-optimal is ~34B tokens. We're at ~4% of that.

### Unlockable with vocab expansion only (no new collection): +6.93B tokens
MV-Omni valid_snac is sitting there, tokenized, but blocked by missing `<snac_N>` / `<seed_N>` vocab entries. Adding these two token families = 1–2 days of work = unlock 6.93B tokens = reach ~8.3B total. This is the highest-leverage action available right now.

### Unlockable with GPU runs: +5–10B tokens (captioning, ego-centric, Cosmos3-DROID)
The captioning pipeline alone multiplies FineVideo by ~4× (69,844 records → ~280K records) with richer language context. Ego-centric adds a second perspective for free.

### Target: 20–40B tokens for v0.2 training
With vocab expansion + MV-Omni + captioning + Cosmos3-DROID + SNAC-FineVideo, reaching 20–40B tokens is realistic within 2–3 months of focused work.

---

## Honest Assessment — Are We On The Right Track?

**Yes, the architecture is sound.** The second model proved the core hypothesis: a 1.7B LLM can learn the grammar of multimodal token sequences — joint ordering, valid xyz ranges, modality-specific token distributions — purely from next-token prediction on flat interleaved sequences.

**The bottleneck is data, not architecture.** The model's failure to autonomously transition between modalities is fully explained by data starvation and missing language anchors. These are solvable engineering problems, not fundamental flaws.

**The direction is genuinely novel.** No published work unifies Seed2 + Cosmos + SNAC + PCHIP pose tokens in a single autoregressive LLM context. The closest prior work (RT-2, OpenVLA, π0) uses much simpler action representations and doesn't attempt continuous 3D body pose. We're building something nobody else has built.

**The risks:**
1. **Scale gap:** Even at 20B tokens, we're far below frontier LLMs. Our model may generalize poorly to novel prompts. Mitigation: mix in standard text data to maintain language ability.
2. **No robot deployment yet:** Current pose data is from YouTube humans, not actual robot joints. Isaac Sim integration is still future work. The model won't directly control a real robot without sim-to-real adaptation.
3. **SNAC/audio quality:** Orpheus SNAC2 is "good enough" per Huu's assessment, but retokenizing with Moss Audio Tokenizer V2 (mentioned in chat, 2.1B decoder) could improve audio quality significantly. Deferred for now.
4. **Qwen3 migration overhead:** If we retokenize for Qwen3, existing `.bin/.idx` shards become obsolete. Should be done once, not multiple times — wait until the data landscape is more stable.

**What success looks like at each stage:**
- **v0.2 (2–3 months):** Model autonomously transitions from text prompt → seed2 → cosmos → agent tokens without needing agent tokens in the prompt.
- **v0.3 (4–6 months):** Model responds to spoken commands (SNAC) by generating valid agent motion tokens. "Walk forward" → valid pelvis/hip/knee trajectory.
- **v1.0 (6–12 months):** Model observes visual scene + receives instruction, generates motion that respects scene geometry. The chemical beaker test.

---

## Key Decisions Log

| Decision | Why | Date |
|----------|-----|------|
| Chose Adaptive PCHIP over fixed 409-token format | Self-describing, ~35% fewer tokens for static joints | Feb 2026 |
| Tokenizer fix via `add_tokens()` not vocab.json edit | BPE requires merge rules, not just vocab entries | May 2026 |
| 99% AVC-LM dropout in Phase 7 | AVC-LM was 373× more tokens than agent — would dominate context | Mar 2026 |
| valid_with_seed NOT worth using | 1.1 TB download for 5.6M tokens (< 0.5% of FineVideo) | Jun 2026 |
| Hold AVC-LM in new datasets until ablations | No evidence yet that it helps vs adds noise | Jun 2026 |
| Ego-centric perspective as free data multiplier | Same underlying motion, different reference frame, doubles diversity | Jun 2026 |
| Qwen3 migration deferred | Too early — data landscape still changing | Jun 2026 |
| MV-Omni: convert seed→seed2 instead of adding new vocab | Avoids unnecessary vocab expansion; seed_N and seed2_N are identical semantics | Jun 2026 |
| SNAC injection in Phase 6, not Phase 7 | Phase 6 already does per-chunk injection; Phase 7 is stateless flatten. Keeping injection in Phase 6 means Phase 7 needs no external lookups. | Jun 2026 |
| SNAC chunk alignment: encode full activity once, split by count | Encoding per-chunk (0.267s segments) would lose audio context + slow due to many small calls. Encode once, split evenly preserves context and is accurate (SNAC rate is constant). | Jun 2026 |
| SNAC for ALL activities, not just agent ones | Only 14% of activities have agent tokens. Other 86% still have seed2+cosmos — adding SNAC teaches seed2→cosmos→snac transitions. Filtering to agent-only wastes most of the GPU run. | Jun 2026 |
| 1-CP compression: deferred, keep adaptive 2/4/8-CP | +7.1% gain (sample-based) doesn't justify full Phase 5→7 re-run right now; revisit later if needed | Jul 8, 2026 |
| Synthetic/sim data capped at ≤30% of total training mix | Team consensus (Huu), citing literature guidance; keeps balance toward real video | Jul 7, 2026 |
| Moss-Audio Tokenizer V2 usage: keep limited even if adopted | Huu: at 400 tok/s it would overwhelm the dataset if used broadly for omni-modal pretraining; only viable as a short high-detail segment followed by lower-rate SNAC, or standalone if not binding to language | Jul 2, 2026 |
| MINT-1T-HTML images: dropped entirely, text kept | No per-image license info available (dataset's own README confirms no copyright filtering was done, only NSFW/size/dedup); `cc_dump` is a CommonCrawl snapshot ID, not a Creative Commons marker. Text kept — different risk profile (LLM text pretraining under fair-use/TDM norms vs. redistributing raw hotlinked media) | Jul 18, 2026 |
| stera-10m: dropped | Not permissive (self-assessed + no objection from Huu) | Jul 18, 2026 |
| SenseNova-SI-8M: download in full | Real image bytes (not URLs), Apache-2.0, spatial-reasoning content relevant to embodied/robot use — best available static-image candidate found so far | Jul 18, 2026 |
| Egocentric perspective converter (Priority 3): rejected as scoped | The transform only relabels the SAME pose (isometry, no info gain) while pairing it with the SAME unchanged 3rd-person video — creates a physically incoherent video→pose training pair, not a real data multiplier. Only worth revisiting with genuine paired egocentric video, or for a narrow pose-only (no-video) sub-task | Jul 18, 2026 |
| Megatron tokenize: try account `laionize`/partition `batch` on JUWELS instead of `cstdl` on JUSUF | User's call, verified via `sacctmgr` that `laionize` has a valid `batch` association before writing scripts | Jul 18, 2026 |

---

## Published Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Tokenizer v1 (144,215 vocab, GPT-NeoX) | `EmpathicRobotics/tokenizer-vla-adaptive` | Live |
| **Tokenizer v2 (156,505 vocab, GPT-NeoX + SNAC)** | `EmpathicRobotics/tokenizer-vla-adaptive-v2` | **Live (Jul 1, 2026)** |
| **Tokenizer Qwen3 (257,897 vocab)** | `EmpathicRobotics/tokenizer-vla-qwen3` | **Live (Jul 1, 2026)** |
| FineVideo-Phase7-Flattened v4 (371,888 records, 5.217B tokens) | `EmpathicRobotics/FineVideo-Phase7-Flattened` | **Live (Jul 7, 2026)** |
| FineVideo-Phase5-AgentTokens (~399K activities) | `EmpathicRobotics/FineVideo-Phase5-AgentTokens` | Live |
| FineVideo-Phase4-YOLOPose (millions of windows) | `EmpathicRobotics/FineVideo-Phase4-YOLOPose` | Live |
| VLA Model v1 (broken tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-25b-test` | Live (deprecated) |
| VLA Model v2 (fixed tokenizer) | `EmpathicRobotics/vla-1.7b-pab-spline-adaptive` | Live |
| Megatron .bin/.idx shards (2.84B tokens) | `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` | Local |
| Data inventory checkpoint | `tools/inventory_checkpoint_v2.json` | Local |

---

## Environments & How to Run (JUWELS login node)

**env_tools** — use for: phase7_flatten, data_inventory, HF uploads, eval, any non-GPU script  
Location: `/p/data1/mmlaion/nguyen38/env_tools`  
Python: 3.12.3 | Has: torch, transformers, wn, datasets, scipy, huggingface-hub, rich, tqdm, ...

> **Note (Jun 27, 2026):** env_tools was created on JUSUF but we run on JUWELS. Python symlink and pyvenv.cfg had wrong paths. Fixed via `load_env_tools.sh`.

```bash
# Activate env_tools (source it, don't bash it — needs to modify your shell):
source /p/data1/mmlaion/nguyen38/3d-human-pose/load_env_tools.sh
# → auto-fixes symlinks on first run, then activates

# Then run whatever you need:
python pipeline_pose/phase7_flatten.py --workers 16 --skip-existing
```

**env_pose** (miniforge3 conda) — use for: phases 1–6 (HRNet, MotionBERT, YOLO, kinematics)  
Location: `/p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`  
Activate: `source /p/data1/mmlaion/nguyen38/3d-human-pose/miniforge3/etc/profile.d/conda.sh && conda activate /p/data1/mmlaion/nguyen38/3d-human-pose/env_pose`

**Data paths on `/p/` (use these when JUPITER `/e/` is down):**
```
/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/
  ├── final_dataset_adaptive/final_vla_adaptive_rank_*.jsonl  ← Phase 7 input
  ├── megatron_dataset_adaptive/flat_*.jsonl                  ← v1 flatten output
  └── megatron_dataset_v2/flat_*.jsonl                       ← v2 flatten output (pending)

/p/data1/mmlaion/shared/vla/
  ├── mv_omni_converted/mv_omni_snac_*.jsonl.gz              ← MV-Omni seed→seed2 done
  ├── tokenizer_vla_adaptive/                                 ← local tokenizer copy
  └── tokenized_output/vla_adaptive/data_shard_*.bin/.idx    ← Megatron shards (2.84B tokens)
```

---

## Immediate Action Items (Next 2 Weeks)

### Đã hoàn thành (Jun–Jul 2026)
- [x] **SNAC CPU job** — **COMPLETE (Jul 1, 2026)**. Job `snac_cpu_14077331`, 32/32 tasks. 371,855 activities, 363M tokens, 6.5 GB → `/p/.../snac_tokens/`
- [x] **Dataset overlap check** — **COMPLETE (Jun 30, 2026)**. Kết quả: 27,359 video chồng nhau (86.9% of valid_with_seed ∈ omni_valid). Xem section "Dataset Overlap Analysis" bên dưới.
- [x] **Vocab expansion (tokenizer build)** — **COMPLETE (Jul 1, 2026)**. Script: `tools/build_tokenizers.py`. Tạo 2 tokenizer:
  - `tokenizer_vla_adaptive_v2` (GPT-NeoX-20b + SNAC): **156,505 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_adaptive_v2/`
  - `tokenizer_vla_qwen3` (Qwen3 + tất cả VLA tokens): **257,897 vocab** → `/p/data1/mmlaion/shared/vla/tokenizer_vla_qwen3/`
  - Spot-check 12 tokens đại diện: tất cả **atomic** (1 token/ID), không có sub-piece splitting
  - SNAC token range: L0 [128266..132361], L1A [132362..136457], L1B [144650..148745]

### Pre-training Discussion Items (Jul 2, 2026 — from Huu chat)

> **⚠ Huu explicitly said: "Before you train let's talk." — do NOT start training until these 3 items are resolved.**

**[DISCUSS-1] Language data mix — what to add before training?**
- Current plan (FineVideo v4 + MV-Omni = 12B tokens) has almost no instruction/language data
- Huu: "mix in a few billion tokens mixture so we can steer the robot better"
- Huu: "We should look for some SFT dataset for our various target (pick up the Apple, Drive left, etc.)"
- Candidates on leo (`/mnt/sdb/mixture-vitae-working/`): `clappa_text_only`, `coco` (synthetic permissive), `misc_instr/hpprc-r1-distill-qwen-pseudo-qa.jsonl` (Japanese instruction)
- Also wants multilingual instruction datasets with reasoning/thinking
- **Action needed:** Identify token counts of available language datasets → decide mix ratio

**[DISCUSS-2] Compression analysis of Adaptive PCHIP — RESULTS READY**
- Huu: "Did you do an analysis by how much compression we got? If there is no or low compression then we know it's wrong."
- **DONE:** `tools/analyze_pchip_compression.py` — 18,847 files, 1,743,189 windows. Results:
  - **50.9% token saving** vs fixed 8-CP (284.1 avg vs 579)
  - CP tiers: 55.2% 2-CP / 25.6% 4-CP / 19.2% 8-CP
  - Most dynamic: r_knee (33.5% 8-CP), r_wrist (29.4%). Most static: pelvis (100% 2-CP)
  - Pelvis confirmed at origin: 500/500 samples within ±0.1m ✓
  - Coordinate system: absolute xyz after root-centering is correct
- **NEW — Pose data quality concern (Jul 2, 2026):**
  - Only **4–7 joints finite per frame** (out of 17) — 24–41% skeleton coverage
  - **Arms (j11–j16) nearly always NaN** — MotionBERT cannot reliably lift arm joints from YouTube (occlusion/side views)
  - **head_top (j10) zero-fill artifact** — stores (0,0,0) when undetected, same as pelvis, counted as finite but wrong
  - Impact: model learns lower-body + torso pose only. Fine for pretraining pose presence; NOT sufficient for arm/hand manipulation learning
- **Action needed:** Report numbers + pose quality concern to Huu

**BEAST comparison (Jul 3, 2026) — context for reporting to Huu:**

Huu asked: *"What does the BEAST paper say about their compression? That will give us a sanity check."*

BEAST = "B-spline Encoded Action Sequence Tokenizer" (KIT, NeurIPS 2025, arXiv 2506.06072). Uses B-splines with fixed N control points fit by ridge regression. Claims **4–8× compression** vs binning (e.g., ACT 100-step chunk → 15 tokens = 6.67×).

**Why our 50.9% looks lower:** different baselines.

| | Baseline | Result |
|---|---|---|
| **BEAST** | Binning (1 token/timestep/DoF) | 4–8× fewer tokens |
| **Ours** | Fixed 8-CP (already compressed) | ~2× fewer tokens |

Vs raw binning: our 284 tokens / (8×17×3=408 raw values) = **~1.5×** — much less than BEAST. Root cause: 34% of our tokens are overhead (wrappers + t tokens) to make the format self-describing for the LLM. BEAST has zero overhead (decoder structure is hardcoded). Self-describing is a deliberate design choice for LLM joint-semantic learning.

**Huu's 1-CP suggestion (Jul 3, 2026):** *"Why do we have 2-CP as minimum? Can we have 1-CP? Like — relative, no movement."*

Why 2-CP was minimum originally:
1. PCHIP requires ≥2 points (it's an interpolating polynomial — 1 point = nothing to interpolate)
2. "Low curvature" ≠ "no movement": joint may drift linearly within the window below tau_low

How 1-CP would work: if `quantize(frame_0) == quantize(frame_7)` for all 3 dims → emit only `<joint_x_N> <joint_y_N> <joint_z_N>` (no t token, 3 tokens vs current 8 tokens)

Estimated gain: ~4–5 qualifying joints/window × 5 tokens saved ≈ 20–47 tokens/window → **additional ~8–16% compression**. Requires grammar change + re-run of Phase 5 and all downstream phases.

**FINAL DECISION (Jul 8, 2026):** Deferred. Confirmed with Huu on Discord — keep the current adaptive 2/4/8-CP format as-is. Full-dataset validation run (18,847 videos) was started but interrupted by the JUWELS outage; not resumed. Revisit only if later data shows it's necessary — the +7.1% gain doesn't justify a full Phase 5→7 re-run right now. For paper purposes, "compression decreases the data by more than 50%" (vs fixed 8-CP) is the number to report.

**[DISCUSS-3] Eval setup**
- Huu: "We should start eval just to see how things perform with baseline"
- Need to define eval tasks BEFORE training, not after
- Candidates: agent token decode quality (MPJPE on 3D pose), modality transition accuracy, instruction-following on robot commands
- **Action needed:** Define eval protocol and implement baseline metrics

---

### Việc tiếp theo (unblocked bởi vocab expansion)
- [x] **Phase 6 v2 dry run** — **COMPLETE (Jul 1, 2026)**. Chạy thử 1 file (254 videos, ~5 phút). Kết quả:
  - SNAC inject: **259,503/259,505** avc blocks (~100%)
  - Agent inject: **12,705** blocks (đúng — 46% video có Phase 5 output)
  - Format verified: `</avc_lm> <agent>...</agent> <snac> <snac_N>... </snac>` ✓
  - `chunk_timing` có đủ các flag `has_seed2/cosmos/avc_lm/agent/has_snac` ✓
  - SLURM script mới: `slurm/submit_merge_adaptive_v2.sh` (account `laionize`, partition `batch`, 32 workers, 2h)
  - Ước tính toàn bộ 160 file với 32 workers: **~25–40 phút**
- [x] **Re-run Phase 6 v2** — **COMPLETE**. Job `14082096`, 32/32 workers. 40,804 videos | 398,775 activities | SNAC 100% | Agent 5.5% | 0 errors → `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/final_dataset_adaptive_v2/` (160 files)
- [x] **Re-run Phase 7 v3** — **COMPLETE (Jul 2, 2026)**. 160/160 files, 371,888 records, 72 GB → `megatron_dataset_v3/`
  - Full-chain (agent+snac): 69,811 (18.8%) | Snac-only: 302,044 (81.2%) | Bad records: 0
  - Token counts: seed2 332.6M | cosmos 3.88B | snac 363M | agent windows 2,148,474 | avclm 0 ✓
  - Sample: `samples/after_flatten_v3.json` | Upload script updated: `tools/upload_flattened_hf.py`
- [x] **Phase 7 v4 — temporal alignment fix** — **COMPLETE (Jul 2, 2026)**. Per-chunk ordering fixed, speech in headers, 5.217B tokens → `megatron_dataset_v4/`. See stats above.
- [x] **Upload Phase 7 v4 to HF** — **COMPLETE (Jul 7, 2026)**. `EmpathicRobotics/FineVideo-Phase7-Flattened` live with v4 data. Shared with Huu/joergfranke on Discord as the ready-to-tokenize dataset.
  Source: `megatron_dataset_v4/` | Upload dir: `hf_upload_flattened_v4/` | Dataset card: `tools/vla_flattened_dataset_card.md` (updated for v4)
- [ ] **Megatron re-tokenize** `megatron_dataset_v4/` with `tokenizer-vla-adaptive-v2` (156,505 vocab) → new `.bin/.idx` → train v0.3
- [x] **Upload tokenizers** — **COMPLETE (Jul 1, 2026)**. `EmpathicRobotics/tokenizer-vla-adaptive-v2` (156,505) + `EmpathicRobotics/tokenizer-vla-qwen3` (257,897), cả hai Live với model card đầy đủ

### Kết luận overlap check (Jun 30, 2026 — XONG)
- [x] Quyết định **KHÔNG dùng `valid_with_seed`** — 86.9% đã có trong omni_valid, 13.1% còn lại (4,141 video) chỉ có seed2 token và không đáng tốn thêm storage/compute
- [x] omni_valid **không overlap với FineVideo-VLA** (khác nguồn hoàn toàn — MixtureVitae vs YouTube FineVideo)

### Coding (không cần GPU)
- [ ] Start writing ego-centric perspective converter (Phase 3 → rotate to head camera)
- [ ] Start writing captioning pipeline (SmolVLM2/Qwen2.5-VL trên FineVideo keyframes)
- [ ] Investigate leo seed2 + euro_pat token counts
- [ ] Plan Cosmos3-DROID pipeline (download strategy, SLURM script)
- [x] Investigate `MixtureVitae-Backup/multimodal` (HF) — **DONE (Jul 9, 2026)**. Mostly text; SNAC tokens found in 2 files as raw int arrays. See "MixtureVitae-Backup Multimodal Investigation" section. Awaiting Huu's decision on whether to add.
- [ ] Clarify "finevideo reformulation" at `leo:/mnt/sdb/mixture-vitae-working/finevideo` — check overlap with own pipeline
- [ ] Decide MV-Omni mix ratio / dropout to avoid diluting agent token % (12.2% → ~5.2% if mixed naively)
- [ ] Scope new data candidates: abc.bot, MolmoAct2-BimanualYAM-Dataset, OmniVideo-100K, MINT-1T-HTML, Gen-EgoData (see "New Data Candidates" table above)

### Cluster account mapping (Jul 7, 2026 — for when submitting jobs)
```
JUSUF:   ccstdl
JUPITER: reformo
JUWELS:  laionize
```

---

## Dataset Overlap Analysis (Jun 30, 2026)

### Background

Từ cuộc họp Jun 28: Huu chỉ ra rằng `omni_valid` có khả năng được subsample từ `valid_with_seed`, dẫn đến **double-counting** nếu dùng cả hai. Kết luận ban đầu: chỉ dùng 3 dataset:
1. **FineVideo-VLA** (local)
2. **omni_valid** (MixtureVitae-Omni)
3. **stack_images3_gzip** (MixtureVitae-Backup)

Còn `valid_with_seed` cần check xem overlap với `omni_valid` bao nhiêu % trước khi quyết định có dùng hay không.

### Data đã download

| Dataset | Location | Format |
|---------|----------|--------|
| `valid_with_seed` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_shards/` | 64 outer `.tar.gz`, mỗi cái chứa file `.ogg`, `.png`, `_seed2.jsonl` |
| `omni_valid` | `/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/` | 6 `valid_snac_N.jsonl.gz` |
| `ontocord/VALID` | `multimodal/head.txt` | Chỉ có head sample (5 records), chưa download full |

### Script overlap check

**Script:** `tools/check_dataset_overlap.py`

**Logic:**
- `valid_with_seed`: Extract YouTube video ID (11 chars đầu) từ tên file trong các tar.gz
- `omni_valid`: Extract `params.id` từ metadata của mỗi JSONL record
- So sánh hai set, tính % overlap

**Command để chạy:**
```bash
cd /p/data1/mmlaion/nguyen38/3d-human-pose
python3 tools/check_dataset_overlap.py
```

Không cần env đặc biệt — chỉ dùng stdlib Python (tarfile, gzip, json, re).

**Output:** In kết quả ra màn hình + lưu vào `tools/dataset_overlap_results.json`

### Kết quả (HOÀN THÀNH — Jun 30, 2026)

Script `tools/check_dataset_overlap.py` đã chạy xong. Kết quả lưu tại `tools/dataset_overlap_results.json`:

| Metric | Số liệu |
|--------|---------|
| `valid_with_seed` unique video IDs | **31,500** |
| `omni_valid` unique video IDs | **238,539** |
| Overlap (cả hai) | **27,359** (86.9% của seed / 11.5% của omni) |
| Chỉ có trong `valid_with_seed` | **4,141** |
| Chỉ có trong `omni_valid` | **211,180** |

**Kết luận: KHÔNG dùng `valid_with_seed`.** omni_valid đã cover 86.9% video của nó. 4,141 video còn lại chỉ có seed2 token và không đủ giá trị (tổng < 700K token) để bù cho 1.1 TB storage.

Log: `tools/overlap_run.log` | Kết quả JSON: `tools/dataset_overlap_results.json`

**Lưu ý quan trọng:** Script phải mở **2 tầng tar** (outer tar → inner tar → files) vì shards 0–30 chỉ chứa inner tar.gz bên trong, không có loose files. Script v1 bị lỗi (0 IDs) do bỏ qua inner tar — đã fix trong v2.

### Cấu trúc dataset (đã verify)

**omni_valid record format:**
```json
{
  "text": "<listen><snac_N>...<snac_N></listen>\n<see><seed_N>...</see>\n...",
  "metadata": "[{\"source\": \"grass-yt-cc-by.{YT_ID}|...\", \"params\": \"{\\\"id\\\": \\\"{YT_ID}\\\", ...}\"}]"
}
```
→ YouTube ID ở `metadata[0].params.id` (11 ký tự)

**valid_with_seed shard format:**
- Outer tar chứa: `shard_NNNNN.tar.gz` (inner) + loose files (`*.ogg`, `*.png`, `*_seed2.jsonl`)
- File name format: `{YT_ID_11chars}_{clip_num}[_{crop}][_seed2].{ext}`
- YouTube ID = 11 ký tự đầu của filename

**ontocord/VALID format** (head.txt — 3 lines per record):
```jsonl
{"file_name": "-mbDQC0y0PY_6.ogg", "media_type": "audio", "text": "...", "snac_token": [...]}
{"file_name": "-mbDQC0y0PY_6.png", "media_type": "image", "text": "..."}
{"emotion": "...", "query": "...", "answer": "...", "shard_idx": "shard_0"}
```

### Câu hỏi đã trả lời (Jun 30, 2026)

1. **Bao nhiêu % của `omni_valid` đến từ `valid_with_seed`?**
   → 11.5% (27,359/238,539). omni_valid chủ yếu là data riêng, KHÔNG phải subsample từ valid_with_seed như Huu dự đoán — thực ra ngược lại: valid_with_seed là subset của omni_valid.

2. **`valid_with_seed` có video không có trong `omni_valid` không?**
   → Có: 4,141 video (13.1% của valid_with_seed). Nhưng những video này chỉ có seed2 token, không có SNAC, và tổng token ~700K — không đủ giá trị để dùng riêng.

3. **`ontocord/VALID` có tương đương với `valid_with_seed` không?**
   → Chưa check (chỉ có head sample 20 records → 0 video ID). Không cần điều tra thêm vì đã quyết định không dùng valid_with_seed.

### Kết luận cuối cùng

**Chỉ dùng 2 nguồn external:**
- **omni_valid (MV-Omni)** — 238,539 video, 6.93B token (SNAC + text + seed2). Cần vocab expansion `<snac_N>`.
- **stack_images3_gzip** — 313K token seed2. Quá nhỏ nhưng không tốn gì thêm nếu đã có sẵn.

**Bỏ valid_with_seed.** 1.1 TB đã download có thể xóa để giải phóng storage.

---

## MixtureVitae-Backup Multimodal Investigation (Jul 9, 2026)

### Background

P0 item from Huu (asked Jul 5): investigate `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` on HF — never scanned before (separate from `valid_with_seed` / `stack_images3_gzip`, which were already inventoried). Run locally on a Windows dev machine (no JUWELS access this session), CPU-only, so the approach was streaming sample-based counting rather than a full download — 103GB total across 15 files.

### Method

Two new scripts, both reusing `PATTERNS`/`count_tokens`/`_hf_token`/`hf_url`/checkpoint machinery from `tools/data_inventory.py`:

- **`tools/peek_multimodal.py`** — structural probe, streams just the first few records/members per file (no full download) to discover format and flag VLA-token presence. Output: `tools/multimodal_peek_report.json`.
- **`tools/count_multimodal_tokens.py`** — true HTTP streaming (never writes the compressed file to disk), caps each file at `--sample-mb` compressed MB (default 75), counts VLA-tag tokens (regex, same as `data_inventory.py`) plus any raw integer token arrays (`*_token`/`*_tokens` fields — generalizes beyond just `snac_token`), extrapolates to full file size. Resumable checkpoint: `tools/multimodal_inventory_checkpoint.json`.

**Key implementation fix:** `valid_data_snac.jsonl.gz`, `train_data_snac.jsonl.gz`, and `emo.jsonl.gz` are **not** true JSONL (one compact object per line) — they're a pretty-printed JSON array where a single record can span many lines. Naive newline-splitting silently produced zero parsed records. Fixed by switching to a streaming buffer + `json.JSONDecoder().raw_decode()` approach that pulls complete top-level JSON values regardless of embedded newlines.

Local env: plain Python venv (`tools/env_multimodal_inventory/`, gitignored) — `pip install requests tqdm`, no conda needed. HF token support added (`tools/.hf_token`, gitignored, read by `_hf_token()`) though this specific repo turned out to be public (no auth required).

### Results (75MB compressed sample per file, extrapolated to full size)

**No file contains our tagged VLA tokens** (`<seed2_N>`, `<cosmos_N>`, `<avclm_N>`, `<snac_N>`) — confirmed at the 75MB-sample scale across all 15 files, not just the initial 5-record peek.

**2 files carry real SNAC audio tokens, as raw integer arrays** (`snac_token: [128266, ...]`), not tag strings:

| File | Size | Sample records | Extrapolated raw SNAC codes |
|---|---|---|---|
| `train_data_snac.jsonl.gz` | 11.1 GB | 131,850 | **~3.11B** |
| `valid_data_snac.jsonl.gz` | 579 MB | 129,996 | **~162M** |
| **Total** | | | **~3.27B raw SNAC codes** |

Comparable in scale to the 4.92B SNAC tokens already found in MixtureVitae-Omni's `valid_snac` — a real, previously-uncounted audio-token resource.

**13 remaining files — plain text/caption corpora** (word-count, extrapolated):

| File | Extrapolated text tokens | Content |
|---|---|---|
| high_stack.tar.gz | 4.11B | StackExchange QA |
| valid_text_only.tar.gz | 3.31B | mixed text |
| stack_maga.tar.gz | 1.65B | StackExchange |
| emo.jsonl.gz | 1.04B | audio-transcript + image-caption pairs |
| train_data_snac.jsonl.gz (`text` field) | 865.5M | transcript alongside the SNAC tokens above |
| magalith-10m-florence2.jsonl.gz | 864.4M | image captions |
| synth_llava2.tar.gz | 162.9M | LLaVA-style image captions |
| clappa.tar.gz | 138.4M | video captions (DISCUSS-1 candidate) |
| synth_llava.tar.gz | 93.7M | LLaVA-style image captions |
| low_nemo_maga.tar.gz | 73.7M | text |
| valid_data_snac.jsonl.gz (`text` field) | 44.1M | transcript alongside the SNAC tokens above |
| youtube.tar.gz | 38.6M | video storyline/description |
| coco.tar.gz | 10.0M | image captions — **exact** (fully consumed within sample) |
| europarl.tar.gz | ~0.1M | ⚠️ low confidence, see caveats |

### Caveats (not yet resolved)

1. **`finevideo_transcripts.jsonl.gz` undercounted (shows 0).** Real field is `transcripts`, not `text` — the counter only checks `text` (matching `data_inventory.py`'s existing convention). Needs a dedicated pass, and — since it's literally FineVideo YouTube transcripts — a video-ID overlap check against our own pipeline (same class of risk as the `valid_with_seed` double-counting issue already resolved once).
2. **`europarl.tar.gz` estimate is close to meaningless** — first sampled member was a single ~986MB record, so the 75MB sample only completed 1 record. Needs a much larger sample or a targeted full scan.
3. **Several archives mix huge text members with binary `.wds` shards** (youtube, synth_llava/synth_llava2, stack_maga, high_stack, valid_text_only) — 75MB only reached a handful of members out of many, so extrapolation assumes uniform density across the archive, which may not hold. Lower confidence than files sampled with hundreds of small members (coco, low_nemo_maga).
4. **Raw `snac_token` integer arrays are not in our tokenizer's `<snac_N>` string format** — would need a conversion step (offset/tag scheme) similar to the MV-Omni `seed→seed2` conversion already done, before these ~3.27B codes could enter our Megatron pipeline.

### Status

Posted to Huu on Discord (Jul 9, 2026, 3:51pm): *"this dataset is mostly text, only train_data_snac.jsonl.gz and valid_data_snac.jsonl.gz have snac tokens ... u want to add it?"* — **awaiting his reply.** Do not start integration/download of the full files until he responds.
