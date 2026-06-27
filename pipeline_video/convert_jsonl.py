import json

input_file = "training_ready_rank_151.jsonl"
output_file = "output_pretty.jsonl"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip the first three lines (videos 1-3)
        f.readline()
        f.readline()
        f.readline()

        # Read the next line (video 4)
        second_line = f.readline()

        if not second_line:
            print("⚠️ File does not contain enough videos.")
        else:
            # Parse the single-line JSON string into a Python dict
            video_data = json.loads(second_line)

            # Write to new file with pretty-print formatting (indent=4)
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(video_data, f_out, indent=4, ensure_ascii=False)

            print(f"✅ Successfully extracted video to: {output_file}")

except FileNotFoundError:
    print(f"❌ File not found: {input_file}")
except json.JSONDecodeError:
    print("❌ JSON decode error on the target line.")
