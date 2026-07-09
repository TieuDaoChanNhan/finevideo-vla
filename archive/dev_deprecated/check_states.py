import json
import numpy as np

def verify_states(jsonl_path="../outputs/state.jsonl"):
    print(f"🔍 Checking file: {jsonl_path}...")

    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ JSONL file not found.")
        return

    if not lines:
        print("❌ File is empty!")
        return

    print(f"📂 Total windows (8-frame clips) extracted: {len(lines)}")

    # Collect all data for statistics
    all_windows = []
    for line in lines:
        data = json.loads(line)
        all_windows.append(data["states"])

    # Convert to numpy; JSON `null` values become np.nan automatically
    arr = np.array(all_windows, dtype=float)

    # --- REPORT ---
    print("\n" + "="*50)
    print("📊 DATA QUALITY REPORT AFTER PHASE 3")
    print("="*50)

    print(f"✅ Array shape (N, 8, 17, 3): {arr.shape}")

    # Check for nulls — checking joint 0 (pelvis) is enough since whole frames go NaN together
    total_frames = arr.shape[0] * arr.shape[1]
    null_frames = np.sum(np.isnan(arr[:, :, 0, 0]))
    print(f"👻 Frames removed (Null)    : {null_frames} / {total_frames} ({(null_frames/total_frames)*100:.2f}%)")

    # Check normalisation (scale)
    if null_frames < total_frames:
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        print(f"📏 Spatial range (Min/Max)  : {min_val:.4f} -> {max_val:.4f}")

        if -2.0 <= min_val and max_val <= 2.0:
            print("   ✨ OK: Coordinates are in metric space (~metres). No more giant pixel values!")
        else:
            print("   ⚠️ WARNING: Coordinates still too large — normalize function may have an issue!")

    print("="*50)

if __name__ == "__main__":
    verify_states()
