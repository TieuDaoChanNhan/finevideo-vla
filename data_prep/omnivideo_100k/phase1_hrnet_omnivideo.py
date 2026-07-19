"""Phase 1 (2D pose: HRNet + Faster R-CNN) cho tap con sports cua OmniVideo-100K,
chay tren JUPITER. Xem data_prep/omnivideo_100k/JUPITER_POSE_PILOT_TASK.md cho
boi canh day du (chi 1,256/5,214 video duoc chon qua select_sports_subset.py).

Khong sua pipeline_pose/phase1_hrnet_gpu.py -- file goc hard-code doc video tu
FineVideo HF arrow dataset (load_from_disk("/e/scratch/.../finevideo_disk") +
cached_video_ids.json), khong dung duoc cho OmniVideo-100K (chi la file mp4
phang tren disk, khong co arrow dataset). Script nay tai su dung phan model-
agnostic cua file goc (model init, coco_to_h36m) nhung doc video truc tiep tu
$DATA/omnivideo_100k/videos/{video_id}.mp4 va shard theo sports-subset list,
giong pattern RANK::WORLD_SIZE cua step_a_tokenize_video.py.

Khac voi ban goc MOT diem quan trong (theo yeu cau giu confidence, dung vut):
coco_to_h36m() o day GIU NGUYEN confidence score lien tuc tu HRNet/detector,
thay vi nhi phan hoa thanh 1.0/0.0 nhu ban goc (phase1_hrnet_gpu.py dong 37-38:
`if c < CONF_THRESHOLD: return ..., 0.0` / `return ..., 1.0`). Downstream,
MotionBERT (third_party/MotionBERT/infer_wild.py qua WildDetDataset) doc truc
tiep cot conf nay lam input feature cho model lifting -- gia tri lien tuc giup
model phan biet "hoi khong chac" voi "gan nhu chac chan khong thay", thay vi
chi co 2 muc nhu ban goc. Vi tri (x, y) van bi zero-hoa khi duoi nguong (giu
nguyen logic goc, tranh dua toa do rac cho MotionBERT) -- chi confidence la
duoc giu nguyen gia tri that.

Output giu dung format {"frame_id", "keypoints": [[x,y,conf]x17]} cua ban goc,
ghi vao outputs/2d_json/{video_id}_2d.json (dung thu muc voi FineVideo, theo
dung thiet ke trong JUPITER_POSE_PILOT_TASK.md muc 3, de Phase 2 doc duoc
khong can sua gi -- an toan vi video_id 2 nguon khong trung nhau).
"""
import argparse
import json
import os

import cv2
import numpy as np

# ================= MODEL CONFIGURATION (giu nguyen path tuong doi tu ban goc,
# script phai chay voi CWD = 3d-human-pose/) =================
POSE_CONFIG = "hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
POSE_CHECKPOINT = "hrnet_storage/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
DET_CONFIG = "hrnet_storage/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT = "hrnet_storage/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

CONF_THRESHOLD = 0.5

DATA_ROOT = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/omnivideo_100k"
DEFAULT_VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEFAULT_VIDEO_IDS_FILE = os.path.join(os.path.dirname(__file__), "sports_subset_video_ids_filtered.txt")
DEFAULT_OUTPUT_DIR = "outputs/2d_json"  # cung thu muc voi FineVideo -- xem docstring

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
DEVICE = f"cuda:{LOCAL_RANK}"


def coco_to_h36m(coco_kpts):
    """Nhu coco_to_h36m() trong phase1_hrnet_gpu.py, nhung KHONG binarize
    confidence -- xem module docstring."""
    h36m = np.zeros((17, 3), dtype=np.float32)

    def get_pt(idx):
        x, y, c = coco_kpts[idx]
        c = float(c)
        if c < CONF_THRESHOLD:
            return np.array([0.0, 0.0]), c
        return np.array([x, y]), c

    nose, c_nose = get_pt(0)
    lsho, c_lsho = get_pt(5)
    rsho, c_rsho = get_pt(6)
    lhip, c_lhip = get_pt(11)
    rhip, c_rhip = get_pt(12)

    h36m[1, :2], h36m[1, 2] = get_pt(12)
    h36m[2, :2], h36m[2, 2] = get_pt(14)
    h36m[3, :2], h36m[3, 2] = get_pt(16)
    h36m[4, :2], h36m[4, 2] = get_pt(11)
    h36m[5, :2], h36m[5, 2] = get_pt(13)
    h36m[6, :2], h36m[6, 2] = get_pt(15)
    h36m[11, :2], h36m[11, 2] = get_pt(5)
    h36m[12, :2], h36m[12, 2] = get_pt(7)
    h36m[13, :2], h36m[13, 2] = get_pt(9)
    h36m[14, :2], h36m[14, 2] = get_pt(6)
    h36m[15, :2], h36m[15, 2] = get_pt(8)
    h36m[16, :2], h36m[16, 2] = get_pt(10)
    h36m[9, :2], h36m[9, 2] = nose, c_nose

    # Gating giu nguyen ">= nguong" (tuong duong "> 0" cua ban goc, vi ban goc
    # da binarize truoc do) -- chi confidence duoc ghi la gia tri that (min cua
    # 2 diem gop thanh, thay vi hardcode 1.0 nhu ban goc).
    if c_lhip >= CONF_THRESHOLD and c_rhip >= CONF_THRESHOLD:
        h36m[0, :2], h36m[0, 2] = (lhip + rhip) / 2.0, min(c_lhip, c_rhip)
    if c_lsho >= CONF_THRESHOLD and c_rsho >= CONF_THRESHOLD:
        h36m[8, :2], h36m[8, 2] = (lsho + rsho) / 2.0, min(c_lsho, c_rsho)
    if h36m[0, 2] >= CONF_THRESHOLD and h36m[8, 2] >= CONF_THRESHOLD:
        h36m[7, :2], h36m[7, 2] = (h36m[0, :2] + h36m[8, :2]) / 2.0, min(h36m[0, 2], h36m[8, 2])
    if c_nose >= CONF_THRESHOLD and h36m[8, 2] >= CONF_THRESHOLD:
        h36m[10, :2], h36m[10, 2] = nose + 0.8 * (nose - h36m[8, :2]), min(c_nose, h36m[8, 2])

    return h36m


def process_video_to_json(video_path, output_json_path, det_model, pose_model):
    from mmpose.apis import inference_topdown
    from mmdet.apis import inference_detector
    from mmengine.registry import init_default_scope

    cap = cv2.VideoCapture(video_path)
    all_frames_data = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        init_default_scope("mmdet")
        det_result = inference_detector(det_model, frame)
        pred_instances = det_result.pred_instances
        person_mask = (pred_instances.labels == 0) & (pred_instances.scores > 0.5)
        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()

        frame_data = {"frame_id": frame_idx, "keypoints": []}
        if len(bboxes) > 0:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best_bbox = bboxes[np.argmax(areas)]

            init_default_scope("mmpose")
            pose_results = inference_topdown(pose_model, frame, bboxes=[best_bbox])

            if len(pose_results) > 0:
                kpts = pose_results[0].pred_instances.keypoints[0]
                scores = pose_results[0].pred_instances.keypoint_scores[0]
                coco_final = np.concatenate([kpts, scores[:, None]], axis=1)
                h36m_keypoints = coco_to_h36m(coco_final)
                frame_data["keypoints"] = h36m_keypoints.tolist()

        if not frame_data["keypoints"]:
            frame_data["keypoints"] = [[0.0, 0.0, 0.0]] * 17

        all_frames_data.append(frame_data)
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"   -> [Rank {RANK}] Frame: {frame_idx}", end="\r")

    cap.release()
    with open(output_json_path, "w") as f:
        json.dump(all_frames_data, f)
    return frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--video-ids-file", default=DEFAULT_VIDEO_IDS_FILE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="Chi xu ly N video dau tien (pilot run). 0 = tat ca.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.video_ids_file) as f:
        video_ids = [line.strip() for line in f if line.strip()]
    if args.limit > 0:
        video_ids = video_ids[: args.limit]
    my_ids = video_ids[RANK::WORLD_SIZE]

    print(f"[Rank {RANK}/{WORLD_SIZE}] {len(my_ids)}/{len(video_ids)} video duoc giao, device={DEVICE}")

    print(f"[Rank {RANK}] Dang khoi tao HRNet & Faster R-CNN tren {DEVICE}...")
    from mmpose.apis import init_model as init_pose_model
    from mmdet.apis import init_detector

    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
    det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

    n_done = n_skip = n_error = 0
    for i, video_id in enumerate(my_ids):
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        final_json = os.path.join(args.output_dir, f"{video_id}_2d.json")
        tmp_json = final_json + f".tmp_rank{RANK}"

        if os.path.exists(final_json):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) Skip (da co): {video_id}")
            n_skip += 1
            continue
        if not os.path.exists(video_path):
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) LOI: khong tim thay {video_path}")
            n_error += 1
            continue

        try:
            n_frames = process_video_to_json(video_path, tmp_json, det_model, pose_model)
            os.rename(tmp_json, final_json)
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) OK: {video_id} ({n_frames} frame)")
            n_done += 1
        except Exception as e:
            print(f"[Rank {RANK}] ({i + 1}/{len(my_ids)}) LOI {video_id}: {e}")
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
            n_error += 1

    print(f"[Rank {RANK}] XONG. done={n_done} skip={n_skip} error={n_error}")


if __name__ == "__main__":
    main()
