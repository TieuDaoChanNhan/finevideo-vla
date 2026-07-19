> **CẬP NHẬT 19/07/2026 — task này ĐÃ XÁC NHẬN XONG HOÀN TOÀN, đọc trước khi bắt đầu lại từ đầu.**
> Driver đã viết: `data_prep/omnivideo_100k/step_a_tokenize_video.py` (+ `submit_step_a_full.sbatch`, `submit_step_a_pilot.sbatch`).
> Job full-scale `970099` (8 node×4 GPU=32 GPU, 5,214 video, có resume) — **`sacct` xác nhận COMPLETED, exit 0:0**, chạy 18/7 19:30→22:01 (2h30'). Verify output thật: 32/32 file `step_a_rank_*.jsonl` (39GB), đúng 5,214/5,214 dòng, log lỗi chỉ có warning vô hại, sample 163 video (`rank_0`) không có video nào bị seed2=0 (bug cũ không tái phát ở full-scale), đủ cả 4 loại token (seed2/cosmos/avclm/caption/speech). **Không cần chạy lại.** Bước Megatron tokenize (`tokenizer_vla_qwen3`) sẽ làm ở JUWELS, ngoài phạm vi task này.
>
> **3 bug thật đã bắt được, đọc kỹ trước khi debug lại nếu cần viết driver tương tự cho dataset khác:**
> 1. `env_stable_vla` có `transformers==4.57.6` (khác `4.52.4` checkpoint gốc yêu cầu) — làm seed2 tokenizer hỏng 2 lớp (import path bị dời + `tie_weights()` crash trên `Qformer.cls=None` cố ý). Đã fix bằng monkeypatch **chỉ trong `step_a_tokenize_video.py`**, không đụng `seed2_tokenizer.py`/`pipeline.py`/env chung. Nếu seed2 lại ra `0` hoặc lỗi mới liên quan `transformers`, đọc lại monkeypatch ở đầu file trước khi sửa thêm.
> 2. **Đừng bao giờ trích toàn bộ frame 1 video ra đĩa tạm cùng lúc** (bug tự gây ra ở lần full-scale đầu, job `970087`, làm tràn quota đĩa khi 32 rank chạy song song — pilot 8 rank không lộ ra vì footprint đồng thời còn thấp). Code hiện tại đã sửa thành streaming từng chunk 8-frame (`extract_chunk_frames()`), giới hạn ~8 frame/rank tại một thời điểm, có resize 512×512. Nếu viết lại driver mới cho dataset khác, giữ đúng pattern streaming này.
> 3. Anchor caption/speech: chỉ chèn **1 lần/segment** (chunk đầu tiên của segment), không phải mọi chunk overlap — segment ~11s (~40 chunk) nhưng caption dài 300-500 từ, chèn mọi chunk sẽ lặp lại đoạn văn hàng chục lần.
>
> Xem `PROGRESS.md`/`PROGRESS_VI.md`/`REPORT.md` (mục "OmniVideo-100K Step A", phiên 18/07 tối muộn) để biết chi tiết đầy đủ + trạng thái job mới nhất.

---

# Task: Step A (video tokenization) cho OmniVideo-100K trên JUPITER

**Bối cảnh:** đây là dataset external mới (không phải FineVideo), 5,214 video YouTube thật (tin tức/cartoon/thử thách...), đã được điều tra + chuẩn bị dữ liệu ở JUWELS. Việc còn thiếu duy nhất là chạy Step A (video → token) — chỉ chạy được ở JUPITER vì cần GPU GH200 (JUWELS chỉ có CPU cho phần này).

## 1. Dữ liệu đã có sẵn (sau khi move từ JUWELS sang)

Sau lệnh `rsync` di chuyển (xem cuối file), dữ liệu nằm ở:

```
$DATA/omnivideo_100k/videos/           # 5,214 file .mp4 thật (49GB)
                                        # tên file = <video_id>.mp4
```

Và trên `/p` (JUWELS, KHÔNG mount được từ JUPITER — cần tự copy 2 file JSONL này qua, chúng nhỏ nên không cần "cut"):

```
/p/data1/mmlaion/shared/vla/omnivideo_100k_flat/omnivideo_100k_segment_captions.jsonl
    # 5,214 dòng, mỗi dòng: {video_id, video_path, duration, video_summary,
    #   segments: [{start_sec, end_sec, caption, speech}, ...]}
    # Đây là caption+speech ĐÃ MAP theo giây thật (không phải chunk Step A —
    # xem mục 3 để biết cách join 2 cái này lại).
```

`data_prep/omnivideo_100k/flatten_qa_text.py` và `build_segment_captions.py` (đã có trong repo, code build ra 2 file JSONL trên) — không cần chạy lại, chỉ cần hiểu format output nếu muốn debug.

**Phần QA-text riêng (`train_oe_70k`/`train_mcq_30k`) đã tokenize XONG ở JUWELS rồi, KHÔNG liên quan tới task này** — task này chỉ lo phần video+caption/speech.

## 2. ⚠️ QUAN TRỌNG — `pipeline_video/pipeline.py` KHÔNG dùng thẳng được, đọc kỹ trước khi bắt đầu

CLAUDE.md ghi `pipeline_video/` là "(complete, do not modify)" — điều đó đúng cho **các class tokenizer cấp thấp**, nhưng file `pipeline.py` hiện tại có 1 class orchestration (`VLADatasetBuilder`) **hard-code cho đúng format FineVideo** — đọc dữ liệu qua `datasets.load_from_disk("/e/scratch/reformo/nguyen38/finevideo_disk")`, parse `original_video_filename`/`scenes`/`activities`/`global_context` từ metadata HF-dataset kiểu FineVideo. OmniVideo-100K KHÔNG có cấu trúc đó (chỉ có mp4 phẳng + 1 JSONL đơn giản), nên **không thể trỏ thẳng `pipeline.py` vào folder video mới rồi chạy**.

**Cách đúng: viết 1 driver script MỚI, tái dùng 3 class tokenizer cấp thấp (đừng sửa code trong `pipeline_video/pipeline.py`, chỉ import hoặc copy-paste 3 class này):**

```python
# từ pipeline_video/pipeline.py, dòng 51-165 — 3 class này KHÔNG phụ thuộc FineVideo:

class Seed2Tokenizer:
    def encode_image(self, image_input) -> list[int]
    # input: 1 PIL Image (1 khung hình, lấy mẫu 1fps theo convention project)

class CosmosVideoTokenizer:
    def encode_video_chunk(self, frame_list, target_size=160) -> list[int]
    # input: list PIL Image (1 chunk 8 khung hình liên tiếp)

class AVCLMTokenizer:
    def encode_mp4_segment(self, mp4_file_path, start_sec, duration_sec) -> list[int]
    # input: path mp4 THẬT + mốc giây bắt đầu/độ dài — tự gọi ffmpeg trích đoạn +
    # encode H.264 + BPE-tokenize bên trong, KHÔNG cần tự trích frame cho phần này
```

`VLADatasetBuilder` (dòng 166+) chỉ nên đọc để THAM KHẢO cách 3 class trên được gọi + cách token được interleave (`<seed2_N> ... <cosmos_N> ... <avc_lm_N> ...`) — không import class này, viết driver mới đơn giản hơn nhiều vì OmniVideo-100K không có scenes/activities lồng nhau (1 video = 1 record, không phải 1 video nhiều activity như FineVideo).

## 3. Việc cần làm — driver script mới (chưa viết, đây là việc chính của task này)

Cho mỗi video trong `videos/*.mp4` (dùng `video_id` = tên file không đuôi):

1. Lấy `duration` từ `omnivideo_100k_segment_captions.jsonl` (hoặc đọc trực tiếp từ mp4 qua ffprobe nếu cần chính xác hơn).
2. Chia video thành các chunk 8-frame theo đúng convention hiện tại của project (fps 30, 8 frame/chunk — xem CLAUDE.md "Agent Token Format" để tham khảo nhịp tương tự, dù OmniVideo-100K không có `<agent>` pose).
3. Với mỗi chunk: gọi `CosmosVideoTokenizer.encode_video_chunk()` + `AVCLMTokenizer.encode_mp4_segment()`. Với mỗi giây (hoặc mỗi N chunk theo đúng nhịp 1fps của Seed2 hiện tại): gọi `Seed2Tokenizer.encode_image()`.
4. **Chèn caption/speech:** với mỗi chunk có khung thời gian `[chunk_start_sec, chunk_end_sec]`, tìm segment nào trong `segments[]` (từ file JSONL đã map) overlap với khung đó, lấy `caption`/`speech` chèn vào — **đây là bước MỚI hoàn toàn, chưa có code sẵn**, nhưng tinh thần y hệt `phase6_merge_adaptive.py --captions-dir/--speech-segments-dir` đang làm cho FineVideo (chèn caption trước `<cosmos>`, speech sau `<avc_lm>`) — chỉ khác nguồn caption là theo giây thay vì theo window_id có sẵn.
5. Ghép toàn bộ thành 1 chuỗi token/video, bọc header dùng `video_summary` làm `### Context:` (giống FineVideo dùng activity description).
6. Xuất `{"text": "..."}` — **1 dòng/video, đúng như bạn hỏi: mỗi video → 1 dãy token duy nhất** (không chia nhỏ thêm như FineVideo có multi-activity).

## 4. Tokenize Megatron — BẮT BUỘC dùng đúng tokenizer, bài học đau từ phiên trước

**Dùng `tokenizer_vla_qwen3` (257,901 vocab), TUYỆT ĐỐI KHÔNG dùng `tokenizer_vla_adaptive_v2`** (GPT-NeoX cũ, 156,509 vocab). Phiên làm việc ở JUWELS vừa phát hiện + fix bug này — cả 3 job tokenize từng chạy nhầm tokenizer cũ, phải xoá 215GB output resubmit lại. Tokenizer Qwen3 đã copy sẵn trong repo tại `vocab/qwen3_tokenizer/` (đã commit + push, `git pull` sẽ có).

Pattern tokenize tham khảo (nếu JUPITER cũng có `mv_preprocess_data.py`/Megatron sẵn — nếu không thì cần hỏi lại, có thể hạ tầng tokenize khác ở JUPITER so với JUWELS `mv-scale/`):
```
--tokenizer-type HuggingFaceTokenizer
--tokenizer-model <path-tới-tokenizer_vla_qwen3-trên-JUPITER>
--json-keys text
```

## 5. Việc KHÔNG nằm trong scope task này (để sau)

- RoboVQA — leader bảo tạm để sau, không đụng vào.
- SenseNova-SI-8M ảnh — vẫn đang chờ quyết định license, không xử lý.
- Tỷ lệ trộn (blend ratio) với FineVideo-VLA khi train — quyết định lúc train, không phải lúc tokenize.

---

## Lệnh di chuyển video (chạy 1 lần, TRƯỚC khi bắt đầu task trên, từ node có cả `/p` và `/e`)

**⚠️ Đây là lệnh MOVE thật (cut, không phải copy) — script tự xoá nguồn trên `/p` sau khi verify copy xong.** Không chạy được từ JUWELS login node (`/e` không mount ở đó) — chạy từ JUPITER login node hoặc node nào có cả 2 mount.

```bash
SRC="/p/data1/mmlaion/shared/vla/omnivideo_100k/videos"
DST="/e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k/videos"

mkdir -p "$DST"
rsync -avh --progress "$SRC/" "$DST/"

SRC_COUNT=$(find "$SRC" -type f -name '*.mp4' | wc -l)
DST_COUNT=$(find "$DST" -type f -name '*.mp4' | wc -l)
echo "src=$SRC_COUNT dst=$DST_COUNT"

if [ "$SRC_COUNT" -eq "$DST_COUNT" ] && [ "$SRC_COUNT" -gt 0 ]; then
    echo "Counts match -- deleting source (real move)"
    rm -rf "$SRC"
else
    echo "COUNT MISMATCH -- NOT deleting source, investigate first"
fi
```

Sau khi move, copy thêm 2 file JSONL nhỏ (không cần cut, chỉ vài chục MB):
```bash
mkdir -p /e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k
scp /p/data1/mmlaion/shared/vla/omnivideo_100k_flat/omnivideo_100k_segment_captions.jsonl \
    <jupiter-node>:/e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k/
# (đổi lệnh scp/rsync tuỳ theo cách 2 cluster nối với nhau thật sự)
```
