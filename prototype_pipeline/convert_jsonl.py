import json

input_file = "training_ready_rank_151.jsonl"
output_file = "output_pretty.jsonl"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        # Bỏ qua dòng đầu tiên (video thứ 1)
        f.readline()
        f.readline()
        f.readline()
        
        # Đọc dòng tiếp theo (chính là video thứ 2)
        second_line = f.readline()

        if not second_line:
            print("⚠️ File không có đủ 2 video.")
        else:
            # Chuyển chuỗi JSON (dạng 1 dòng) thành Python Dictionary
            video_data = json.loads(second_line)

            # Ghi vào file mới với định dạng thụt lề (indent=4)
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(video_data, f_out, indent=4, ensure_ascii=False)
            
            print(f"✅ Đã trích xuất thành công video thứ 2 vào file: {output_file}")

except FileNotFoundError:
    print(f"❌ Không tìm thấy file {input_file}. Hãy kiểm tra lại đường dẫn.")
except json.JSONDecodeError:
    print("❌ Lỗi định dạng JSON ở dòng thứ 2.")