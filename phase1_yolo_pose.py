import cv2
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = "videos/sample.mp4"
OUTPUT_JSON = "outputs/keypoints.json"
CONF_THRESHOLD = 0.5

def coco_to_h36m(coco_kpts):
    """
    coco_kpts: NumPy array of shape (17, 3) containing [x, y, confidence]
    Returns: h36m array of shape (17, 3) in MotionBERT coordinate format.
    """
    h36m = np.zeros((17, 3), dtype=np.float32)

    # Internal helper to safely retrieve a keypoint
    def get_pt(idx):
        x, y, c = coco_kpts[idx]
        if c < CONF_THRESHOLD:
            return np.array([0.0, 0.0]), 0.0
        return np.array([x, y]), 1.0

    # Extract original COCO keypoints
    nose, c_nose = get_pt(0)
    lsho, c_lsho = get_pt(5)
    rsho, c_rsho = get_pt(6)
    lhip, c_lhip = get_pt(11)
    rhip, c_rhip = get_pt(12)

    # --- DIRECT MAPPING OF AVAILABLE KEYPOINTS ---
    # Right leg
    h36m[1, :2], h36m[1, 2] = get_pt(12)  # RHip
    h36m[2, :2], h36m[2, 2] = get_pt(14)  # RKnee
    h36m[3, :2], h36m[3, 2] = get_pt(16)  # RAnkle

    # Left leg
    h36m[4, :2], h36m[4, 2] = get_pt(11)  # LHip
    h36m[5, :2], h36m[5, 2] = get_pt(13)  # LKnee
    h36m[6, :2], h36m[6, 2] = get_pt(15)  # LAnkle

    # Left arm
    h36m[11, :2], h36m[11, 2] = get_pt(5) # LShoulder
    h36m[12, :2], h36m[12, 2] = get_pt(7) # LElbow
    h36m[13, :2], h36m[13, 2] = get_pt(9) # LWrist

    # Right arm
    h36m[14, :2], h36m[14, 2] = get_pt(6) # RShoulder
    h36m[15, :2], h36m[15, 2] = get_pt(8) # RElbow
    h36m[16, :2], h36m[16, 2] = get_pt(10)# RWrist

    # Nose
    h36m[9, :2], h36m[9, 2] = nose, c_nose

    # --- VECTOR-BASED COMPUTATION FOR DERIVED KEYPOINTS ---
    # Pelvis (midpoint of hips)
    if c_lhip > 0 and c_rhip > 0:
        pelvis = (lhip + rhip) / 2.0
        h36m[0, :2], h36m[0, 2] = pelvis, 1.0
    else:
        pelvis = np.array([0.0, 0.0])
        h36m[0, 2] = 0.0 # Occluded

    # Neck (midpoint of shoulders)
    if c_lsho > 0 and c_rsho > 0:
        neck = (lsho + rsho) / 2.0
        h36m[8, :2], h36m[8, 2] = neck, 1.0
    else:
        neck = np.array([0.0, 0.0])
        h36m[8, 2] = 0.0

    # Torso (spine - midpoint between pelvis and neck)
    if h36m[0, 2] > 0 and h36m[8, 2] > 0:
        h36m[7, :2] = (pelvis + neck) / 2.0
        h36m[7, 2] = 1.0

    # Head Top (extrapolated from neck toward nose)
    if c_nose > 0 and h36m[8, 2] > 0:
        vec_neck_to_nose = nose - neck
        h36m[10, :2] = nose + 0.8 * vec_neck_to_nose
        h36m[10, 2] = 1.0

    return h36m

def extract_pose():
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results_json = []
    MISS_THRESHOLD = 30
    prev_kpts = None
    miss_count = 0

    print("Starting 2D pose extraction using YOLO11...")
    for frame_id in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # Check if no person is detected in the frame
        if results.keypoints is None or len(results.boxes) == 0:
            miss_count += 1
            if prev_kpts is None or miss_count > MISS_THRESHOLD:
                # If tracking is lost for too long, reset to zeros
                kpts = np.zeros((17, 3), dtype=np.float32)
            else:
                kpts = prev_kpts.copy()
        else:
            miss_count = 0
            # GET FULL 3D ARRAY INCLUDING CONFIDENCE (use .data instead of .xy)
            kpts_all = results.keypoints.data.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            # Select the person with the highest confidence (main subject)
            best_id = np.argmax(scores)
            
            # Apply H36M mapping
            kpts = coco_to_h36m(kpts_all[best_id])
            prev_kpts = kpts.copy()

        # MotionBERT infer_wild.py expects keypoints as a flattened 1D list
        results_json.append({
            "idx": frame_id,
            "keypoints": kpts.reshape(-1).tolist()
        })

    cap.release()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_json, f)

    print(f"Done! Saved {len(results_json)} frames to: {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_pose()