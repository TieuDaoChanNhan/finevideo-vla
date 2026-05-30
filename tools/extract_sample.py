import json
import os

def extract_video_with_agent():
    input_file = "../prototype/FineVideo-VLA/final_dataset/final_vla_rank_0.jsonl"
    output_file = "sample_with_agent.json"

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    print(f"🔍 Searching for a video containing <agent> in: {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):

            # Fast text pre-check: skip lines without <agent> immediately
            if "<agent>" in line:
                print(f"✅ Found <agent> at line {line_num}.")

                data = json.loads(line)
                video_id = data.get("video_id", "unknown")

                # Save the full JSON record to file for inspection
                with open(output_file, "w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, ensure_ascii=False, indent=4)

                print(f"🎉 Saved video '{video_id}' to '{output_file}'.")

                # Print where exactly the <agent> tag is located
                for s_idx, scene in enumerate(data.get("scenes", [])):
                    for a_idx, act in enumerate(scene.get("activities", [])):
                        tokens = act.get("video_tokens", "")

                        if "<agent>" in tokens:
                            print(f"\n📍 Token location: Scene {s_idx + 1}, Activity {a_idx + 1}")

                            # Show a snippet around <agent>
                            start_idx = tokens.find("<agent>")
                            snippet_start = max(0, start_idx - 60)  # back 60 chars to show the preceding <avc_lm> tag

                            print("👀 Snippet (interleaved tokens):")
                            print(f"... {tokens[snippet_start:start_idx + 150]} ...\n")
                            return  # stop after first match

    print("❌ Scanned the entire file but found no <agent> tokens.")

if __name__ == "__main__":
    extract_video_with_agent()
