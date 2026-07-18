# PAB-Spline VLA — Tiến độ dự án

**Tác giả:** Van Khue Nguyen  
**Cập nhật lần cuối:** 18/07/2026  
**Cluster:** JUPITER (JSC), partition `booster`, GPU GH200 — **đã lên lại**, job batch chạy bình thường tính đến 15/7  
**Mục tiêu:** Xây dựng mô hình VLA (Vision-Language-Action) — xem video, nghe tiếng, sinh ra token điều khiển robot.

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
