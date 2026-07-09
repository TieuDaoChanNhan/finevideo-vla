import json
import glob
import re
import os

def check_flattened_directory(input_dir="/p/data1/mmlaion/shared/vla/vla_25b"):
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not files:
        print(f"❌ No .jsonl files found in {input_dir}")
        return

    print(f"🔍 Sanity-checking {len(files)} data files...")
    print("=" * 60)

    vocab_limits = {
        "agent": 256,
        "avclm": 8192,
        "seed2": 8192,
        "cosmos": 64000
    }

    total_errors = 0

    for file_idx, file_path in enumerate(files, start=1):
        file_name = os.path.basename(file_path)
        print(f"▶️ Checking [{file_idx}/{len(files)}]: {file_name}...")

        corrupted_json = 0
        missing_text = 0
        tag_errors = 0
        vocab_errors = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip(): continue

                # 1. Parse JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    corrupted_json += 1
                    if corrupted_json <= 3: print(f"  ❌ JSON error at line {line_idx}")
                    continue

                # 2. Check key "text"
                if "text" not in data:
                    missing_text += 1
                    if missing_text <= 3: print(f"  ❌ Line {line_idx} missing 'text' field")
                    continue

                text_content = data["text"]

                # 3. Check matching open/close tags
                for tag in ["seed2", "cosmos", "avc_lm", "agent"]:
                    open_count = len(re.findall(f"<{tag}>", text_content))
                    close_count = len(re.findall(f"</{tag}>", text_content))
                    if open_count != close_count:
                        tag_errors += 1
                        if tag_errors <= 3: print(f"  ❌ Mismatched <{tag}> tags at line {line_idx} (open: {open_count}, close: {close_count})")

                # 4. Check vocab bounds
                token_matches = re.findall(r"<([a-zA-Z0-9_]+)_(\d+)>", text_content)
                for prefix, num_str in token_matches:
                    if prefix in vocab_limits:
                        num_val = int(num_str)
                        if num_val >= vocab_limits[prefix]:
                            vocab_errors += 1
                            if vocab_errors <= 3: print(f"  ❌ Vocab overflow: <{prefix}_{num_val}> at line {line_idx}")

        file_errors = corrupted_json + missing_text + tag_errors + vocab_errors
        total_errors += file_errors

        if file_errors == 0:
            print(f"   ✅ {file_name}: clean")
        else:
            print(f"   ⚠️ {file_name}: {file_errors} error(s) found")

    print("=" * 60)
    if total_errors == 0:
        print("🎉 All files passed. Ready to submit to Megatron.")
    else:
        print(f"🚨 Total errors found: {total_errors}. Fix the flatten code and re-run.")

if __name__ == "__main__":
    check_flattened_directory()
