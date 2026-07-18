# Datasets — PAB-Spline VLA

**Cập nhật:** 18/07/2026 (chiều — thêm khảo sát dataset mới ngoài HF + đính chính SenseNova/Gen-EgoData)
**Mục đích:** tổng hợp tất cả dataset đã/đang/có thể dùng cho project — trạng thái tải, mức độ tương thích vocab hiện tại, và việc còn thiếu trước khi đưa vào Megatron tokenize + training. File này bổ sung cho `REPORT.md`/`PROGRESS.md` (lịch sử chi tiết từng phiên làm việc), ở đây chỉ giữ lại kết luận theo từng dataset để tra cứu nhanh.

**Nguyên tắc license (nhắc lại, hay bị vi phạm nếu chỉ nhìn tag HF):** tag `license:` ở đầu trang HF chỉ đáng tin nếu dataset tự chứa (ảnh/video bytes thật, không phải URL) VÀ README nói rõ nguồn gốc raw content. Với dataset dạng "registry"/"curated từ nguồn khác" (Open X-Embodiment, SenseNova-SI-8M, MINT), tag đó nhiều khả năng chỉ áp cho lớp annotation/wrapper, không phải bản quyền nội dung gốc — luôn tìm phần "Data Source"/"Collection" trong README/paper trước khi kết luận "permissive".

**Vocab hiện có** (tham khảo nhanh, xem CLAUDE.md để đầy đủ): `tokenizer-vla-adaptive` (144,215, không có SNAC), `tokenizer-vla-adaptive-v2` (156,509, GPT-NeoX + SNAC + caption/speech wrapper), `tokenizer-vla-qwen3` (257,901, Qwen3 + toàn bộ VLA token). Token modality gồm: `seed2` (1fps semantic), `cosmos` (spatial, mỗi 8 frame), `avclm` (H.264 BPE, mỗi 8 frame), `agent` (3D pose 17 khớp, PCHIP), `snac` (audio), `caption`/`speech` (text anchor, không cần vocab riêng vì là BPE thường + 4 wrapper token).

---

## Bảng tổng quan nhanh

| Dataset | Đã tải? | Vocab tương thích? | Ready Megatron tokenize? | Ghi chú ưu tiên |
|---|---|---|---|---|
| **FineVideo-VLA** (nội bộ) | ✅ Có (nguồn gốc) | ✅ 100% (chính chủ) | ⚠️ Flat JSONL sẵn sàng (v5), nhưng **chưa chạy Phase 8 với bản v5** — bản `.bin/.idx` thật hiện có là v1 cũ (2.84B token, trước SNAC/caption/speech) | Cần re-tokenize v5 → train v0.3 |
| **MixtureVitae-Omni** (`valid_snac`) | ✅ Có | ✅ (seed→seed2 đã convert, snac đã khớp range vocab) | ✅ Format đã đúng `{"text":...}`, chỉ thiếu bước chạy tokenize thật | Quick win — gần như miễn phí |
| **MINT-1T-HTML** | ✅ Có (2.7TB, full, chỉ phần text) | ✅ (text thường, không cần token đặc biệt) | ⚠️ Chưa sample-tokenize để biết số token thật | **Phần ảnh đã BỎ (18/7, quyết định Huu — không track được license)**, chỉ dùng text |
| **SenseNova-SI-8M** | 🟡 Đang tải (~97%, 1.1/1.13TB) | ✅ phần text / ❌ phần ảnh (chưa có converter) | ❌ | ⚠️ **License ảnh gốc KHÔNG xác minh được** (đính chính so với kết luận sáng 18/7 — xem mục 4) |
| **OmniVideo-100K** | ❌ Chưa | ❌ (raw video, cần qua pipeline) | ❌ | Rẻ nhất trong nhóm robot/video — video thật + có sẵn caption/script |
| **MolmoAct2-BimanualYAM** | ❌ Chưa | ❌ (robot joint-space, LeRobot) | ❌ | Cần modality mới cho robot-action |
| **Cosmos3-DROID** | ❌ Chưa | ❌ (chưa xác nhận khớp `<cosmos_N>`) | ❌ | Cần modality mới cho robot-action |
| **Gen-EgoData** | ❌ Chưa | ❌ (eef-pose+gripper, khác agent-token 17-khớp) | ❌ | Nhỏ (500 sample), CC-BY-SA-4.0 (share-alike, cần Huu duyệt điều khoản pháp lý) — xem mục 8 (đã viết lại) |
| **MixtureVitae-Backup/multimodal** | ⚠️ Đã sample (75MB/file), chưa tải full | ⚠️ 2 file có SNAC dạng int thô, còn lại text | ❌ | Đang chờ Huu quyết định (đã hỏi 9/7, chưa có câu trả lời rõ trong docs) |
| **VALID** (`ontocord/VALID` trên HF) | ❌ Chưa (leader muốn tìm bản trên Leo trước) | ❌ | ❌ | **Deprioritized** — bản HF có thể cũ, chờ path từ Leo |
| **stera-10m** | ❌ Chưa (gated, cần accept license) | ❓ Chưa biết | ❌ | **Not permissive — đã bị loại (18/7, đồng thuận Huu + Van Khue trong chat)** |
| **MINT PDF data** (Huu đã tải, đã lọc permissive) | ✅ Huu đã có sẵn | ❓ Chưa biết | ❓ Chưa biết | **Cần tìm trên Leo** (`account/datasets/` hoặc `.../working/`) — user tự tìm |
| **FineVLA** (`xlang-ai/FineVLA`) | ❌ Không thể | — | — | Data training thật chưa public ("coming soon") |
| **abc.bot** | ❌ Không có trên HF | — | — | Cần tìm cơ chế tải riêng, chưa điều tra |
| **RoboVQA** (google-deepmind) | ❌ Chưa | ❌ (raw video, cần qua pipeline) | ❌ | ✅ **Permissive thật** (CC-BY-4.0 + Apache-2.0, xác nhận từ GitHub chính chủ) — 238h, 3 embodiment, nhưng cách tải thật (GCS bucket qua Colab) chưa điều tra |
| **Open X-Embodiment** (`jxu124/OpenX-Embodiment`) | ❌ Chưa | ❌ (raw video/RLDS, cần qua pipeline) | ❌ | ⚠️ **Registry 55-60 dataset con, license KHÔNG đồng nhất** — top-level tag CC-BY/Apache chỉ áp phần lớn, không phải tất cả — cần audit từng sub-dataset trước khi dùng |
| **NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim** | ❌ Chưa | ❌ (LeRobot-style, robot joint action — modality mới) | ❌ | ✅ **Permissive thật** (CC-BY-4.0), **345K+ trajectory humanoid/robot-arm simulation**, đúng vai trò "robot-action" mà project đang thiếu — ứng viên mạnh nhất mới tìm được, xem mục 17 |
| **AgiBot World** (`agibot-world/*`) | ❌ Đã check, loại | — | — | ❌ **Not permissive** — CC BY-NC-SA 4.0 (NonCommercial) |
| **Apple EgoDex** | ❌ Đã check, loại | — | — | ❌ **Not permissive** — CC-BY-NC-ND (NonCommercial + No-Derivatives), tiếc vì rất khớp use-case (829h ego dexterous manipulation + pose) |
| **Meta ego-1k / EgoBrain** | ❌ Đã check, loại | — | — | ❌ Not permissive (FAIR Noncommercial / CC-BY-NC) — EgoBrain còn lạc chủ đề (EEG/neuroscience, không phải robot) |

---

## 1. FineVideo-VLA (nội bộ — flagship dataset)

**Tổng quan:** Dataset chính của project, tự build từ ~40K video YouTube (nguồn FineVideo) qua toàn bộ pipeline Step A (video token) + Phase 1–7 (pose). Là dataset duy nhất có đủ cả 4 modality gốc + caption/speech mới thêm.

**Đã tải/path:** Tự sinh ra, không phải "tải" từ nguồn ngoài.
- Bản mới nhất (v5, có caption+speech): `/p/data1/mmlaion/shared/nguyen38/data/FineVideo-VLA/megatron_dataset_v5/` — 160 file, 72GB, 371,888 record.
- Bản đã public trên HF: `EmpathicRobotics/FineVideo-Phase7-Flattened` — hiện **vẫn là bản v4** (chưa có caption/speech), sắp upload bản mới theo lời bạn báo Huu.
- Bản `.bin/.idx` Megatron thật (đã dùng để train model thứ 2): `/p/data1/mmlaion/shared/vla/tokenized_output/vla_adaptive/` — 2 shard, **2.84B token, dùng vocab v1 cũ (không SNAC/caption/speech)**.

**Tokenize theo modality (bản v5, verify 18/7):**

| Modality | Token | % | Trạng thái |
|---|---|---|---|
| seed2 | 332,592,448 | 6.3% | ✅ |
| cosmos | 3,882,954,800 | 73.9% | ✅ |
| agent | 637,924,374 | 12.1% | ✅ |
| snac | 363,029,331 | 6.9% | ✅ |
| caption | 12,076,047 | 0.2% | ✅ mới thêm 17/7 |
| speech_inline | 27,012,397 | 0.5% | ✅ mới thêm 17/7 |
| **Tổng** | **5,255,589,397** | — | |

**Structure:** Flat `{"text": "### Title:...### Context:...[### Speech:...]\n<seed2_N>...<cosmos_N>...<agent><fps_30><pelvis>...</agent><snac_N>...<caption>...</caption><speech>...</speech>"}`, thứ tự per-chunk temporal (v4 fix), agent/caption/speech không bị dropout/augment (giữ nguyên văn vì gắn đúng 1 thời điểm).

**Bổ sung token nào được:** Đây là dataset duy nhất có agent (pose) — không có gì để "bổ sung" thêm loại token, ngược lại đây là nguồn cung cấp agent-token DUY NHẤT cho toàn bộ corpus (không dataset ngoài nào có pose).

**Ready Megatron tokenize/training:** ⚠️ **Chưa hoàn toàn.** Flat JSONL (v5) đã sẵn sàng về mặt nội dung, nhưng **chưa chạy lại Phase 8 (Megatron `.bin/.idx`)** với tokenizer mới (`tokenizer-vla-adaptive-v2` hoặc `tokenizer-vla-qwen3`) trên bản v5. Model thứ 2 hiện tại được train trên bản v1 cũ hơn nhiều (2.84B token, thiếu SNAC/caption/speech).

---

## 2. MixtureVitae-Omni (`valid_snac`)

**Tổng quan:** Nguồn external substantial nhất đã inventory — 238,539 video, tổng 6.93B token (SNAC + text + seed). Không có agent/cosmos.

**Đã tải/path:**
- Raw gốc (6 file, trước convert): `/p/data1/mmlaion/nguyen38/inventory_cache/hf_snac/` — 30GB (`valid_snac_0..5.jsonl.gz`).
- Đã convert `<seed_N>`→`<seed2_N>`: `/p/data1/mmlaion/shared/vla/mv_omni_converted/mv_omni_snac_*.jsonl.gz` — 6 file, **30GB, ~1.78M record** (verify thật 18/7: shard 0 riêng có 296,651 dòng).

**Tokenize theo modality:**

| Modality | Trạng thái |
|---|---|
| seed2 | ✅ Đã convert từ `<seed_N>`, verify 0 token cũ còn sót (full dataset, session 27/6) |
| snac | ✅ Range gốc `[128266..148745]` đã khớp thẳng vocab `tokenizer-vla-adaptive-v2`/`qwen3`, không cần convert |
| agent | ❌ Không có — MV-Omni không chứa pose |
| cosmos | ❌ Không có — verify 18/7 (0 match trong sample 500 dòng) |
| text | ✅ Có sẵn, dạng câu hỏi bọc quanh token (VD: *"Q: Listen to this and tell me what you heard. \<listen\>\<snac_...\>...\</listen\>"*) |

**Structure (verify thật 18/7):** `{"text": "Q: ...<listen><snac_N>...</listen>...", "metadata": "[{\"source\": ..., \"params\": {\"id\": youtube_id, ...}}]"}` — **đã đúng schema flat `{"text":...}` giống hệt output Phase 7 của mình**, không phải hierarchical nên KHÔNG cần bước "flatten" kiểu Phase 7.

**Bổ sung token nào được:** Không có loại token mới nào — chỉ tăng khối lượng seed2 + snac hiện có. Rủi ro: trộn thô sẽ pha loãng tỷ lệ agent-token từ 12.2% xuống ~5.2% của corpus gộp (đã cảnh báo nhiều lần trong REPORT.md).

**Ready Megatron tokenize/training:** ✅ **Gần như sẵn sàng** — chỉ còn 2 việc: (1) quyết định tỷ lệ trộn/dropout với FineVideo để không pha loãng agent-token (chưa quyết), (2) chạy Megatron tokenize thật (chưa chạy lần nào). Không cần code thêm gì lớn.

---

## 3. MINT-1T-HTML

**Tổng quan:** Web corpus text+ảnh xen kẽ (CommonCrawl HTML 2017–2024), nhắm vào DISCUSS-1 (thiếu language/instruction data — FineVideo gần như 100% modality-specific, gần như không có text tự nhiên).

**Đã tải/path:** ✅ **Tải xong hoàn toàn** (verify 18/7 qua log `snapshot_download completed successfully` + đếm file thật). `/p/data1/mmlaion/shared/vla/mint1t_html/data_v1_1/` — 6,159/6,159 file parquet, **2.7TB**.

**Tokenize:** Chưa — chưa chạy bước nào. Không cần vocab VLA (không có seed2/cosmos/snac/agent).

**Structure:**
```
texts:            list<string>   — nội dung text thật, xen kẽ trong trang
images:            list<string>   — CHỈ LÀ URL ảnh, KHÔNG có bytes
metadata:          string (JSON)
url:               string
cc_dump:           string
```

**Bổ sung token nào được:** Không thêm loại token VLA nào — đây là nguồn **text thuần túy**, tokenize bằng BPE thường (không cần tag đặc biệt).

**⚠️ QUYẾT ĐỊNH (18/7, chat với Huu): BỎ HẲN phần ảnh, không phải tạm dừng.**
- Điều tra license thật (đọc README chính thức mlfoundations): dataset **không hề lọc bản quyền ảnh** — pipeline filter của họ chỉ có text quality/dedup/NSFW-safety/size, và README tự nhận *"users are responsible for ensuring its legal use... independently verify compliance"*. Tag `license: cc-by-4.0` trên HF chỉ áp dụng cho bản thân bộ dữ liệu đã curate (text/URL list), KHÔNG phải license của từng ảnh gốc (ảnh vẫn thuộc bản quyền chủ blog gốc, phần lớn không có license rõ ràng).
- `cc_dump` **không phải license info** — chỉ là mã đợt crawl CommonCrawl (VD `CC-MAIN-2017-22`), dễ nhầm "cc" = Creative Commons nhưng thực ra là CommonCrawl.
- Đã thử pilot tải 20 shard (~9.2M ảnh) để benchmark tốc độ — **đã dừng, không tiếp tục** theo quyết định của Huu: *"if the mint doesn't have images ignore it"* (chỉ có URL, không track được license → coi như "not a dataset for us").
- Script `tools/extract/extract_mint1t_manifest.py` + `tools/extract/download_mint1t_images.py` giữ lại trong repo (đã fix bug tốc độ per-domain rate-limit) nhưng **không dùng nữa cho tới khi có nguồn ảnh khác đã xác định rõ license**.

**Phần text vẫn dùng bình thường** ("the hf dataset is fine" — Huu, 18/7). Muốn có `<seed2_N>` từ ảnh của nguồn khác, ưu tiên tìm nguồn permissive rõ ràng thay vì crawl URL MINT.

**Ready Megatron tokenize/training (phần text):** ⚠️ Chưa — cần sample-tokenize `texts` bằng tokenizer thật của mình để ra số token thật (742B là con số theo tokenizer của họ, không phải của mình), rồi mới quyết định lấy bao nhiêu % của 2.7TB (mục tiêu DISCUSS-1 chỉ cần "vài tỷ token", không cần hết).

---

## 4. SenseNova-SI-8M

**Tổng quan:** Dataset **spatial-intelligence VQA** chính thức của SenseNova-SI series (SenseTime + collaborators, paper arXiv:2511.13719) — dùng để train model `SenseNova-SI-1.1-InternVL3-8B`, dataset chính thức lớn nhất hiện có cho "spatial reasoning" (định vị vật thể, hướng tương đối, nhận diện vật thể chung giữa nhiều ảnh — la bàn 8 hướng, khoảng cách...). Không phải QA chung chung — **rất gần với reasoning không gian mà robot/embodied agent cần** (đúng loại Huu muốn: *"VLA instructions and VLA reasoning"*).

**Đã tải/path:** ❌ Chưa tải full. HF ID: `sensenova/SenseNova-SI-8M`, license Apache-2.0, **1.13TB thật** (đo qua HF tree API 18/7). 2 config: `preview` (1000 sample, đã tải+inspect thật 18/7) và `full` (8.16M sample).

**Tokenize:** Chưa — 0% vocab VLA (không seed2/cosmos/agent/snac).

**Structure (verify thật từ data, không phải README):**
```
id:             int64
conversations:  string (JSON) — dạng ShareGPT: [{"from":"human","value":"<image> <image> câu hỏi..."}, {"from":"gpt","value":"đáp án"}]
image:          list<{bytes: binary, path: string}>  — ẢNH THẬT (bytes, JPG/PNG), KHÔNG PHẢI URL như MINT
```
- **Ảnh là bytes thật embed sẵn trong parquet** — khác hẳn MINT-1T-HTML (chỉ URL) — **không có rủi ro dead-link, tự chứa (self-contained)**.
- Trung bình **~2-4 ảnh/record** (8.16M sample / 2.72M ảnh unique ≈ tỷ lệ ảnh dùng lại ~3x giữa các câu hỏi khác nhau — cùng 1 ảnh được hỏi nhiều câu khác nhau, nên số ảnh unique cần tải/xử lý ít hơn nhiều so với số record).
- Format câu hỏi: trắc nghiệm A/B/C/D về không gian trong nhà (VD: *"Select the closest object to `light` among `couch`, `coffee table`, `bed`, `stool`"*, *"You observe that window lies to the East of shoe. Can you locate washbasin with respect to toilet seat?"*) — bối cảnh chủ yếu **indoor scene** (phòng khách, bếp, phòng tắm...).
- Ảnh kích thước vừa phải (1024×768 – 1296×968), ~100-575KB/ảnh trong sample.

**Bổ sung token nào được:**
- **Phần text (conversations) dùng được ngay** làm instruction/reasoning data cho DISCUSS-1, format ShareGPT chuẩn dễ convert, không cần vocab mới.
- **Phần ảnh là bytes thật** — khác MINT (chỉ có option lý thuyết vì URL chết/license mù mờ), đây là ứng viên **thực tế nhất** để bổ sung `<seed2_N>` từ ảnh tĩnh: chỉ cần viết converter ảnh-đơn→Seed2 (pipeline hiện tại chỉ tokenize frame từ video, chưa có nhánh ảnh đơn lẻ, nhưng input đã sẵn sàng — không vướng license/dead-link như MINT).
- Vì ảnh dùng lại nhiều lần giữa các câu hỏi, nên tokenize theo **ảnh unique** (dedupe trước) sẽ tiết kiệm compute đáng kể so với tokenize theo record.

**⚠️ ĐÍNH CHÍNH (18/7 chiều) — license KHÔNG xác minh được như kết luận sáng nay, đây là bẫy cùng dạng MINT `cc_dump`:**
Huu đặt nghi vấn (dựa trên ChatGPT) rằng ảnh trong dataset có thể không permissive hoàn toàn. Điều tra lại kỹ (HF README, GitHub `OpenSenseNova/SenseNova-SI`, paper arXiv:2511.13719 abstract+PDF, và tự đọc `image` column thật trong parquet — path dạng `images/059/034763.jpg`, đã bị đánh số lại, không còn dấu vết nguồn) — **không tìm thấy bất kỳ tài liệu nào nói rõ ảnh gốc lấy từ đâu**. Paper/GitHub đều dùng từ "**curated**" cho 8.16M sample/2.72M ảnh — gợi ý đây là tổng hợp từ nguồn khác (kiểu ScanNet/Matterport3D/ARKitScenes hay dùng cho bài toán spatial VQA này), không phải SenseNova tự chụp mới. Tag `apache-2.0` trên HF nhiều khả năng chỉ áp cho lớp annotation/QA text họ tự viết, **không phải guarantee cho bản quyền ảnh gốc bên dưới**.

**Kết luận cập nhật:** không còn coi đây là "an toàn permissive hơn MINT" nữa — **license ảnh thật sự chưa xác minh được, mở, chưa nên coi là sẵn sàng train** cho tới khi tìm được appendix paper (chưa extract hết được) hoặc tác giả xác nhận nguồn ảnh trực tiếp. Download **không dừng** (gần xong, phí bandwidth đã bỏ ra không đáng huỷ) nhưng quyết định sử dụng bị treo lại, cần Huu/Van Khue xác nhận trước khi đưa vào training corpus.

**Ready Megatron tokenize/training:** ❌ Chưa tải xong, và giờ thêm điều kiện license chưa rõ trước khi dùng.

---

## 5. OmniVideo-100K

**Tổng quan:** Dataset instruction-tuning cho audio-visual reasoning qua structured script + evidence chain (paper arXiv 2606.14702).

**Đã tải/path:** ❌ Chưa tải. HF ID: `MiG-NJU/OmniVideo-100K`, Apache-2.0, **52.9GB thật** (đo qua HF tree API).

**Tokenize:** Chưa.

**Structure:** `videos.tar.part_xx` (**video thật, không phải URL**) + `train_oe_70k.jsonl`/`train_mcq_30k.jsonl` (QA gốc) + `scripts.jsonl` (**structured script cho toàn bộ video — coi như caption/language-anchor có sẵn**).

**Bổ sung token nào được:** Đây là "lớp 1" (raw video) — chạy thẳng qua Step A (Seed2/Cosmos/AVC-LM) + Phase 1–7 pose y hệt FineVideo, có thể ra đủ cả `seed2`/`cosmos`/`avclm`/(`agent` nếu có người trong khung hình). Bonus lớn: `scripts.jsonl` đã là caption/language-anchor sẵn, đỡ phải tự caption như vừa làm cho FineVideo.

**Ready Megatron tokenize/training:** ❌ Chưa tải, nhưng **là ứng viên rẻ nhất trong nhóm dataset raw-video/robot** vì tái dùng toàn bộ pipeline có sẵn, chỉ cần đổi input glob.

---

## 6. MolmoAct2-BimanualYAM-Dataset

**Tổng quan:** Bộ demo thao tác robot 2 tay (bimanual manipulation) quy mô lớn dùng để train MolmoAct2 — hơn 720 giờ demo, đa dạng task tabletop.

**Đã tải/path:** ❌ Chưa tải. HF ID: `allenai/MolmoAct2-BimanualYAM-Dataset`, Apache-2.0, **2.35TB thật** (đo qua HF tree API — khớp với ước tính "2TB" ghi trong docs cũ).

**Tokenize:** Chưa — 0% vocab VLA.

**Structure:** LeRobot v3 format (`format:parquet`, `library:lerobot`), có video (`modality:video`), robot state/action dạng timeseries (`modality:timeseries`), và **instruction text đã annotate sẵn** (`meta/tasks_annotated.parquet`, index theo `episode_index`).

**Bổ sung token nào được:** Video có thể qua Step A để ra `seed2`/`cosmos`. Phần **action robot (joint-space bimanual) là loại token hoàn toàn mới chưa tồn tại trong project** — khác về bản chất với agent-token hiện tại (xyz người 17 khớp H36M) — cần thiết kế modality "robot-action" riêng (retargeting), chưa ai bắt đầu.

**Ready Megatron tokenize/training:** ❌ Chưa tải, cần quyết định kiến trúc trước khi đầu tư code.

---

## 7. Cosmos3-DROID

**Tổng quan:** DROID (dữ liệu teleop robot thật, tay đơn/đôi) đóng gói lại thành LeRobotDataset v3.0 bởi NVIDIA — 71,907 episode (57,639 success + 14,268 failure), ~22.4M frame @15fps.

**Đã tải/path:** ❌ Chưa tải. HF ID: `nvidia/Cosmos3-DROID`, license OpenMDW-1.1 (cho phép dùng thương mại), **707GB** (đã confirm từ session trước).

**Tokenize:** Chưa.

**Structure:** LeRobotDataset v3.0, 3 camera stream (2 exterior + 1 wrist) + joint/cartesian/gripper state+action. Tên "Cosmos3" gợi ý có liên quan tới Cosmos video tokenizer nhưng **chưa xác nhận token `<cosmos_N>` của họ có khớp trực tiếp vocab `<cosmos_N>` của mình hay không** — cần kiểm tra kỹ trước khi giả định tương thích.

**Bổ sung token nào được:** Tương tự MolmoAct2 — phần action robot cần modality mới; phần video **có khả năng** tận dụng được `<cosmos_N>` nếu vocab khớp (chưa verify).

**Ready Megatron tokenize/training:** ❌ Chưa tải, cần quyết định kiến trúc robot-action trước (cùng nhóm với MolmoAct2).

---

## 8. Gen-EgoData

**Tổng quan (viết lại 18/7 chiều sau khi điều tra kỹ toolkit + schema):** **Không phải** "video ego + pose người" kiểu để làm giàu FineVideo — đây là dữ liệu thu bằng **thiết bị cầm tay chuyên dụng "DAS device"** của GenRobot (người thao tác cầm thiết bị demo task, kiểu handheld gripper-interface giống UMI), có action robot thật, không phải suy luận từ video quan sát người.

**Đã tải/path:** ❌ Chưa tải. HF ID: `genrobot2025/Gen-EgoData`, **CC-BY-SA-4.0** (share-alike — ràng buộc pháp lý: model train ra có thể phải release dưới license copyleft tương thích, khác hẳn Apache/MIT — cần Huu duyệt điều khoản này, không chỉ coi là "permissive" ngang các license khác trong bảng), **47.6GB thật**, chỉ 500 sample / 4.23 giờ, 10 task (organize_utensils, v.v., phân cấp Domain > Scenario > Task > Skill).

**Tokenize:** Chưa. Toolkit đọc file: `genrobot-ai/das-datakit` (MIT license — đây là license của TOOLKIT/code, không phải của DATA, đừng nhầm 2 cái).

**Structure (đã tra được từ README toolkit, 18/7):** File `.mcap` (kiểu ROS), mỗi file = 1 skill instance. 3 camera thật (`camera0` mid-fisheye, `camera1`/`camera2` stereo trái-phải — nhiều góc cố định, không hẳn "first-person" thuần). Action/pose: `/robot0/vio/eef_pose` = **end-effector pose 6-DoF (Pos_X/Y/Z, Q_X/Y/Z/W) + `Gripper_width`** — đây là action space tay-đơn (single-arm eef+gripper), **khác hẳn** format 17-khớp body pose (H36M) mà `<agent>` token hiện tại dùng.

**Bổ sung token nào được:** Đây là loại dữ liệu cùng nhóm vai trò với MolmoAct2/Cosmos3-DROID/GR00T-Sim (mục 17, mới) — **cần modality "robot-action" mới hoàn toàn** (eef-pose + gripper-width), không tái dùng được `<agent>` (17-khớp) hay pipeline HRNet/MotionBERT hiện tại.

**Nên tải không?** Đúng loại dữ liệu về khái niệm (action-grounding thật, camera gần góc thao tác thật) nhưng **quy mô quá nhỏ** (500 sample) để đáng công sức code modality mới riêng cho nó — nên gộp chung quyết định với MolmoAct2/Cosmos3-DROID/GR00T-Sim (mục "robot-action modality" ở cuối file) thay vì làm riêng lẻ.

**Ready Megatron tokenize/training:** ❌ Ưu tiên thấp — quy mô nhỏ nhất trong nhóm robot-action, thêm điều kiện license share-alike cần duyệt riêng.

---

## 9. MixtureVitae-Backup — folder `data/multimodal`

**Tổng quan:** 15 file, 103GB, đã sample-scan (75MB/file, không tải full) trong session 9/7.

**Đã tải/path:** ⚠️ Chỉ sample, chưa tải full. Chưa có path local cho bản đầy đủ.

**Tokenize:** Chưa — nhưng đã biết trước:
- `train_data_snac.jsonl.gz` (11.1GB) + `valid_data_snac.jsonl.gz` (579MB): có SNAC dạng **integer array thô** (`snac_token: [128266, ...]`), KHÔNG phải string tag `<snac_N>` — cần bước convert giống MV-Omni trước khi dùng được (~3.27B code ước tính).
- 13 file còn lại: text/caption thuần (StackExchange, LLaVA-caption, v.v.) — không có token VLA nào.

**Structure:** Đa dạng — `.tar.gz` (StackExchange, LLaVA captions) + `.jsonl.gz` (SNAC, không phải JSONL chuẩn — đã có script `count_multimodal_tokens.py` xử lý format pretty-print JSON array riêng).

**Bổ sung token nào được:** SNAC audio (~3.27B code, quy mô tương đương MV-Omni) nếu convert; phần còn lại chỉ là text thường cho language-mix.

**Ready Megatron tokenize/training:** ❌ **Đang chờ quyết định của Huu** — đã hỏi trên Discord (9/7): *"this dataset is mostly text, only train_data_snac.jsonl.gz and valid_data_snac.jsonl.gz have snac tokens ... u want to add it?"*, chưa thấy câu trả lời rõ ràng ghi lại trong docs. Không nên tải full/tích hợp tới khi có xác nhận.

---

## 10. VALID (`ontocord/VALID` trên HuggingFace)

**Tổng quan:** Video-Audio Large Interleaved Dataset — ~720K video CC-BY từ YouTube, format `<video><caption><image>...</caption></video><transcript><audio>...</transcript>text`, rất khớp ý tưởng interleave-modality mà Huu mô tả trong chat.

**Đã tải/path:** ❌ Chưa tải. HF ID: `ontocord/VALID`, **662GB hiện có trên HF** (publisher ghi rõ đây là bản PREVIEW, mục tiêu cuối ~14TB / 12K shard / 7M record).

**⚠️ Trạng thái ưu tiên: TẠM DỪNG theo yêu cầu (18/7)** — leader nhiều khả năng muốn bản trên cluster `leo` (đường dẫn dạng `account/datasets/...`), bản trên HF có thể đã cũ/khác so với bản leader đang nhắc tới. Không nên tải bản HF này cho tới khi xác nhận lại với Huu hoặc tìm thấy path thật trên leo.

**Tokenize/Structure/Ready:** Chưa đánh giá — để sau khi rõ nguồn.

---

## 11. stera-10m

**Tổng quan:** Đã từng bị gắn cờ "restrictive license" trong data inventory (Jun 2026).

**Đã tải/path:** ❌ Chưa tải — **gated (auto-approve)**, cần bấm "Agree" trên trang HF trước khi script tải chạy được. HF ID: `fpvlabs/stera-10m`, license `other` (không phải license chuẩn permissive).

**Tokenize/Structure:** Chưa xem được — bị chặn bởi gate, chưa inspect schema.

**❌ ĐÃ LOẠI (18/7) — kết luận cuối, không cần điều tra thêm.** Xác nhận rõ trong chat với Huu: Van Khue tự đánh giá "not permissive", Huu không phản đối. Cùng nhóm quyết định với việc bỏ ảnh MINT — project chỉ nhận permissive data.

---

## 12. FineVLA (`finevla.xlang.ai` / `xlang-ai/FineVLA`)

**Tổng quan:** Ứng viên VLA training set (47,159 trajectory từ 10 nguồn robot dataset khác nhau).

**Đã tải/path:** ❌ Không thể — repo trả về HTTP 401, và GitHub repo `xlang-ai/FineVLA` ghi rõ "coming soon" cho phần data/checkpoint thật. Chỉ có `xlangai/RoboFine-bench` public (500 video, **eval benchmark**, KHÔNG phải training data).

**Kết luận:** Không có gì để làm ở đây cho tới khi upstream release.

---

## 13. abc.bot (Amazon)

**Tổng quan:** 400 giờ robot recording trong simulation, có physics state (MjData) — được đánh giá "promising nhất" trong các candidate robot-sim từ session 8/7.

**Đã tải/path:** ❌ Không tìm thấy repo nào trên HF (search API trả về rỗng cho cả model/dataset). Nhiều khả năng phải tải thủ công từ site riêng của Amazon — **chưa điều tra cơ chế tải**.

**Kết luận:** Cần điều tra riêng (không đi qua pipeline HF-download hiện có) trước khi đánh giá được size/structure/license thật.

---

## 14. MINT PDF data (Huu đã tải sẵn — trên Leo)

**Tổng quan:** Nhắc tới trong chat 18/7 — Huu: *"Look for the mint pdf data i already downloaded. I think I filtered out permissive already"*. Đây nhiều khả năng là split PDF của MINT-1T (phân biệt với `MINT-1T-HTML` đang dùng — MINT-1T gốc có 3 split: HTML/PDF/ArXiv, dự án mới chỉ động tới HTML). Điểm quan trọng: **Huu nói đã tự lọc permissive rồi** — nếu đúng, đây có thể là nguồn ảnh permissive thay thế cho phần ảnh MINT-1T-HTML vừa bị loại.

**Đã tải/path:** ✅ Huu đã tải, ❓ chưa rõ path chính xác — gợi ý duy nhất từ chat: *"Datasets. Or working"* (thư mục trên Leo). **User tự tìm** (đã có leo access).

**Tokenize/Structure/License:** Chưa biết gì — chờ tìm ra path rồi mới đánh giá được.

**Việc cần làm:** Sau khi tìm thấy trên Leo, kiểm tra lại: (1) thật sự đã lọc permissive chưa hay chỉ là giả định của Huu cần verify, (2) structure/format, (3) có ảnh thật (bytes) hay cũng chỉ là URL như MINT-1T-HTML.

---

## 15. RoboVQA (Google DeepMind)

**Tổng quan:** VQA đa embodiment cho long-horizon instruction — video ghi 3 kiểu "diễn viên" thực hiện cùng 1 việc (robot thật / người / người dùng công cụ) trong cùng môi trường (3 toà nhà văn phòng thật), ghép với text hỏi-đáp/mô tả instruction dài. Paper: arXiv:2311.00899.

**Đã tải/path:** ❌ Chưa tải. Repo chính chủ: `github.com/google-deepmind/robovqa`. **Cách tải thật chưa rõ** — README chỉ trỏ tới notebook Colab (`data_loading_and_eval.ipynb`, load từ GCS bucket, không phải HF `snapshot_download` đơn giản) — có bản mirror không chính chủ trên HF (`xuexinda/robovqa`) nhưng chưa verify license có giữ nguyên qua mirror hay không.

**License — permissive thật, xác nhận từ GitHub chính chủ (không phải chỉ tag HF của bản mirror):** *"All software is licensed under the Apache License, Version 2.0... All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)."* Rõ ràng, không mập mờ kiểu SenseNova/MINT.

**Quy mô (theo paper, chưa tự đo):** 829,502 cặp (video, text), 29,520 instruction unique, 238 giờ, 3 embodiment (robot/human/human+tool), 3 toà nhà văn phòng.

**Bổ sung token nào được:** Video thật → có thể qua Step A (Seed2/Cosmos/AVC-LM) như OmniVideo-100K nếu tải được dạng file thật (chưa xác nhận format file, .mp4 hay gì). Text là VQA/instruction dài — hợp DISCUSS-1 (language anchor), không có robot-action/joint data (đây là VQA quan sát, không phải teleop demo).

**Ready Megatron tokenize/training:** ❌ Chưa tải — **việc cần làm trước:** tìm hiểu cơ chế tải thật (chạy thử Colab notebook hoặc tìm GCS bucket path trực tiếp), verify license của bản mirror HF nếu dùng đường đó thay vì GCS gốc.

---

## 16. Open X-Embodiment (`jxu124/OpenX-Embodiment` trên HF)

**Tổng quan:** Registry robot-learning lớn nhất hiện có — 1M+ trajectory thật, 22 loại robot, 21 viện nghiên cứu đóng góp, nền tảng cho RT-X. Định dạng RLDS.

**Đã tải/path:** ❌ Chưa tải. **1.13TB** (đo qua HF).

**⚠️ License — KHÔNG đồng nhất, đừng coi tag đầu trang là áp dụng cho tất cả:** Trang HF ghi "Apache 2.0 (software) + CC-BY-4.0 (materials)" nhưng đây **là registry gồm 55-60 dataset con**, mỗi cái nguồn gốc từ 1 lab/paper khác nhau — đa số CC-BY-4.0/Apache-2.0 thật, nhưng theo khảo sát ngoài (web search, chưa tự verify từng cái) **có 1 số subset research-only/non-commercial** lẫn trong đó. **Không được tải nguyên registry rồi coi cả khối là permissive** — phải liệt kê danh sách 55-60 dataset con kèm license riêng, lọc bỏ phần non-commercial trước khi dùng, giống bài học MINT/Open X không phải single-license.

**Structure:** RLDS (TFRecord-based), có video/camera stream + robot state/action thật theo từng dataset con.

**Bổ sung token nào được:** Robot-action thật (đa dạng embodiment) — cùng nhóm quyết định kiến trúc "robot-action modality" với MolmoAct2/Cosmos3-DROID/Gen-EgoData/GR00T-Sim.

**Ready Megatron tokenize/training:** ❌ Chưa tải. **Việc cần làm trước khi tải:** audit license từng dataset con trong 55-60 cái (không audit thì không được coi là "permissive full" theo đúng yêu cầu) — việc này tốn công, nên làm sau khi đã quyết định kiến trúc robot-action (không audit trước khi biết có dùng hay không).

---

## 17. NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

**Tổng quan:** Dữ liệu simulation dùng để post-train GR00T N1 (foundation model robot của NVIDIA) — **ứng viên mạnh nhất mới tìm được trong phiên này** cho vấn đề "robot-action modality" đang thiếu, vì: (1) license permissive thật rõ ràng, (2) đã có humanoid (GR1) — khớp mục tiêu "generalist humanoid VLA" của chính project, (3) là dữ liệu **simulation** — cùng bản chất với pipeline Isaac Sim của project (chưa integrate) chứ không phải robot thật khác hệ, có thể dùng làm nguồn thay thế/bổ sung trong lúc Isaac Sim riêng chưa xong.

**Đã tải/path:** ❌ Chưa tải. HF ID: `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.

**License:** ✅ **CC-BY-4.0, permissive thật** (xác nhận trực tiếp từ tag + không thấy điều khoản NVIDIA riêng nào khác trong README).

**Quy mô thật (breakdown theo README):**
| Nhóm | Trajectory | Robot |
|---|---|---|
| Cross-embodied bimanual | 9,000 | Panda, GR1 arms |
| Humanoid tabletop | 240,000 | GR1 arms+waist |
| Humanoid downsampled | 24,000 | GR1 unified |
| Robot arm kitchen | 72,000 | Panda gripper |
| Unitree G1 locomotion | 102 | Unitree G1 |
| **Tổng** | **~345,102** | |

**Structure:** Chưa rõ chi tiết (README không show schema đầy đủ qua WebFetch) — cần tự tải sample kiểm tra thật (giống cách đã làm với SenseNova/MINT), nhưng tên folder gợi ý format LeRobot-style, có config `gr1_arms_waist`/`gr1_full_upper_body` (humanoid joint-space thật).

**Bổ sung token nào được:** Robot-action joint-space cho humanoid GR1 — đây chính là loại token "robot-action" mà project đang thiếu (theo đúng kiến trúc `pipeline_video`/`pipeline_pose` hiện tại chỉ có agent-token người, chưa có robot-action token nào).

**Ready Megatron tokenize/training:** ❌ Chưa tải, chưa inspect schema thật. **Việc cần làm:** tải sample nhỏ để tự đọc schema thật (đừng tin README suông), rồi đưa vào cùng quyết định kiến trúc robot-action-modality với MolmoAct2/Cosmos3-DROID.

---

## Việc còn mở / quyết định cần leader

1. ~~**MINT-1T-HTML ảnh**~~ — **ĐÃ QUYẾT ĐỊNH BỎ (18/7)**, không track được license. Phần text vẫn giữ, cần sample-tokenize `texts` để ra số token thật, rồi quyết định lấy bao nhiêu % cho DISCUSS-1.
2. **MV-Omni + MINT text** — **Megatron tokenize đang chạy thật** (job `14118392`/`14118393`, submit 18/7 ~11:54, RUNNING >1h tính đến lúc ghi entry, không lỗi) — check lại xem đã COMPLETED thật chưa, verify output trước khi coi là xong (đừng tin SLURM state suông, xem [[project_vla_status]] về vụ MV-Omni từng fail âm thầm lần 1). Quyết định tỷ lệ trộn MV-Omni/dropout (tránh loãng agent-token) vẫn CHƯA chốt, để dành lúc train.
3. **FineVideo-VLA v5** — tokenize thật đã COMPLETED (job `14117681`, 4 shard/~378GB/5.256B token) — sẵn sàng dùng, không cần việc gì thêm.
4. **MolmoAct2 / Cosmos3-DROID / Gen-EgoData / Open X-Embodiment / GR00T-Sim** — **5 ứng viên robot-action-modality cùng nhóm quyết định kiến trúc** (chưa ai bắt đầu code): cần 1 quyết định duy nhất — thiết kế modality "robot-action" (eef-pose+gripper hay joint-space, format token gì) trước khi đầu tư tải bất kỳ cái nào trong 5. **Gợi ý sau khảo sát 18/7 chiều: GR00T-Sim (mục 17) là ứng viên tốt nhất để bắt đầu** — permissive rõ ràng, đã có humanoid GR1, và có thể đóng vai trò "thế chỗ tạm" cho Isaac Sim pipeline (đang not-yet-integrated) trong lúc chờ.
5. **MixtureVitae-Backup/multimodal** — đang chờ Huu trả lời câu hỏi đã hỏi 9/7.
6. **VALID** — chờ path thật từ Leo (user tự tìm), không dùng bản HF preview vội.
7. **abc.bot** — cần điều tra cơ chế tải riêng ngoài HF.
8. ~~**stera-10m**~~ — **ĐÃ LOẠI (18/7)**, not permissive, đồng thuận Huu.
9. **MINT PDF data** (mới, 18/7) — cần tìm trên Leo (user tự tìm), verify permissive thật hay chưa.
10. **SenseNova-SI-8M license** — **MỞ LẠI (18/7 chiều), không còn coi là an toàn.** Không tìm được nguồn gốc ảnh gốc trong bất kỳ tài liệu nào (README/GitHub/paper) — nghi ngờ của Huu (qua ChatGPT) có cơ sở thật, cùng dạng bẫy `cc_dump` như MINT. Cần Huu/Van Khue quyết định có chấp nhận rủi ro này không, hoặc tìm cách liên hệ tác giả xác nhận nguồn ảnh, trước khi đưa vào training corpus. Download vẫn tiếp tục (gần xong, ~97%).
11. **RoboVQA** (mới, 18/7) — license permissive thật (CC-BY-4.0+Apache-2.0), nhưng **cách tải thật chưa điều tra** (GCS bucket qua Colab, không phải HF snapshot đơn giản) — cần thử chạy notebook chính chủ hoặc tìm bucket path trực tiếp trước khi quyết định tải.
12. **Kiến trúc ego/exo cho FineVideo-VLA** (thảo luận dài 18/7 chiều, xem thêm [[project_vla_status]]) — **kết luận: KHÔNG cần sửa FineVideo-VLA**, pose token vốn đã root-centred/body-relative, không có khái niệm "pose exocentric" để sửa. Vấn đề thật (nếu có) là domain-gap giữa video train (3rd-person) và video lúc robot deploy (camera gắn robot) — giải pháp đúng là **ưu tiên integrate Isaac Sim pipeline** (hoặc dùng GR00T-Sim mục 17 làm tạm), không phải đi tìm thêm ego-video dataset để pretraining.
