import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Tìm các đoạn video có điểm số V-JEPA cao nhất.")
    parser.add_argument("--result-json", type=str, required=True, help="Đường dẫn tới file JSON kết quả")
    parser.add_argument("--top-k", type=int, default=15, help="Số lượng đoạn muốn hiển thị")
    args = parser.parse_args()

    # Đọc file JSON
    with open(args.result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clip_scores", [])
    if not clips:
        print("❌ Không tìm thấy dữ liệu 'clip_scores' trong file JSON.")
        return

    # Sắp xếp các clip theo điểm similarity giảm dần (từ cao xuống thấp)
    clips_sorted = sorted(clips, key=lambda x: x["similarity"], reverse=True)

    print("=" * 60)
    print(f"🏆 TOP {args.top_k} ĐOẠN CÓ ĐIỂM SỐ CAO NHẤT (THEO V-JEPA) 🏆")
    print("=" * 60)
    
    for i, clip in enumerate(clips_sorted[:args.top_k]):
        start = clip["start_frame"]
        end = clip["end_frame_inclusive"]
        sim = clip["similarity"]
        is_anomaly = clip["is_anomaly"]
        
        # Đánh dấu xanh/đỏ tùy theo nó có bị filter loại hay không
        status = "❌ LOẠI" if is_anomaly else "✅ GIỮ"
        
        print(f"Top {i+1:02d} | Frames: [{start:06d} -> {end:06d}] | Score: {sim:.4f} | Trạng thái: {status}")

if __name__ == "__main__":
    main()