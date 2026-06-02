import os
SCRATCH_DIR = "/p/scratch/laionize/nguyen38"
SCRATCH_CACHE_DIR = os.path.join(SCRATCH_DIR, "hf_cache")

os.environ["HF_HOME"] = SCRATCH_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = SCRATCH_CACHE_DIR

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from tokenizers import Tokenizer
import subprocess
import imageio_ffmpeg
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import json
from datasets import load_dataset
from tqdm import tqdm
import shutil

# Global constants for PyTorch hardware acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATASET_NAME = "HuggingFaceFV/finevideo"
MAX_VIDEOS = 100

# =====================================================================
# TOKENIZER CLASSES (Preserved exactly as requested)
# =====================================================================

class Seed2Tokenizer:
    """
    Handles Semantic Vision. 
    Can process raw images or extract frames from MP4 videos.
    """
    def __init__(self, target_size=512):
        self.folder = "./seed2" # Ensure this folder exists
        self.target_size = target_size
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self):        
        if os.path.exists(self.folder):
            sys.path.append(self.folder)
        
        # Try importing the local Seed2 model
        try:
            from seed2_tokenizer import Seed2Tokenizer as LocalSeed2Tokenizer
            print(f"📦 [Seed2] Initializing Tokenizer on {DEVICE}...")
            self.tokenizer = LocalSeed2Tokenizer.from_pretrained(
                self.folder, 
                torch_dtype=DTYPE
            ).to(DEVICE)
            print("✅ [Seed2] Encoder loaded successfully.")
        except Exception as e:
            print(f"⚠️ [Seed2] Error loading real tokenizer: {e}")

    def encode_image(self, image_input):
        """
        Core function to process a single image.
        Accepts either a file path (str) or a PIL Image object.
        """
        try:
            with torch.no_grad():
                tokens = self.tokenizer.encode_image(image_input) 
                
                if torch.is_tensor(tokens):
                    return tokens.flatten().detach().cpu().numpy().tolist()
                return tokens
                
        except Exception as e:
            print(f"❌ [Seed2] Tokenize error: {e}")
            return []

class CosmosVideoTokenizer:
    """
    Handles Spatio-Temporal Vision (Cosmos DV8x16x16).
    Compresses chunks of video frames into 3D Spatio-Temporal Tubelets.
    Enforces a strict 8-frame sliding window with causal padding for leftovers.
    """
    def __init__(self):
        # We target the Discrete Video (DV) tokenizer with 8x16x16 compression
        self.model_name = "Cosmos-Tokenizer-DV8x16x16"
        self.tokenizer_path = f"pretrained_ckpts/{self.model_name}/encoder.jit"
        self.decoder_path = f"pretrained_ckpts/{self.model_name}/decoder.jit"
        
        self.encoder = None
        self.decoder = None
        
        self.load_tokenizer()

    def load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            print(f"⚠️ [Cosmos] Checkpoint not found at {self.tokenizer_path}.")
            print(f"   -> Please run the HuggingFace download script as per NVIDIA docs.")
            return
        
        print(f"📦 [Cosmos] Loading Video Encoder on {DEVICE}...")
        try:
            self.encoder = CausalVideoTokenizer(checkpoint_enc=self.tokenizer_path)
            print("✅ [Cosmos] Encoder loaded successfully.")
        except Exception as e:
            print(f"⚠️ [Cosmos] Error loading encoder: {e}")

    def encode_video_chunk(self, frame_list, target_size=160):
        """
        Takes an exact list of 8 PIL Images and encodes them.
        Returns a flat list of token IDs.
        """
        if self.encoder is None: 
            return []
            
        try:
            # 1. Prepare individual frames (Resize + Normalize)
            processed_frames = []
            transform = T.Compose([
                T.Resize((target_size, target_size)),
                T.ToTensor(), # Converts to [0, 1], shape [C, H, W]
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]
            ])

            for img in frame_list:
                img_rgb = img.convert("RGB")
                tensor_frame = transform(img_rgb)
                processed_frames.append(tensor_frame)

            # 2. Stack into 4D tensor: [Time(8), Channels(3), Height(256), Width(256)]
            temporal_tensor = torch.stack(processed_frames)

            # 3. Permute and Add Batch dimension
            # Required Shape: [Batch(1), Channels(3), Time(8), Height(256), Width(256)]
            temporal_tensor = temporal_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            
            # Move to target device and precision
            temporal_tensor = temporal_tensor.to(DEVICE).to(DTYPE)

            # 4. Inference
            with torch.no_grad():
                indices, _ = self.encoder.encode(temporal_tensor)
            
            # 5. Flatten indices to a 1D list
            return indices.flatten().detach().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"❌ [Cosmos] Encode error: {e}")
            return []

class AVCLMTokenizer:
    """
    Handles Physical/Motion Vision (AVC-LM).
    Tokenizes raw H.264 bitstreams. Can extract bitstreams directly from MP4 containers.
    """
    def __init__(self, vocab_dir="avc_lm_v2"):
        self.vocab_dir = vocab_dir
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self):
        json_path = os.path.join(self.vocab_dir, "tokenizer.json")
        if not os.path.exists(json_path):
            print(f"⚠️ [AVC-LM] Tokenizer file not found at {json_path}")
            return
        
        print(f"📦 [AVC-LM] Loading BPE Tokenizer from {self.vocab_dir}...")
        try:
            self.tokenizer = Tokenizer.from_file(json_path)
            print("✅ [AVC-LM] Tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ [AVC-LM] Load error: {e}")

    def encode_h264_bitstream(self, h264_file_path):
        """
        Core function: Reads raw .h264 file and converts it to tokens using the Latin-1 trick.
        """
        if self.tokenizer is None: return []
        
        try:
            with open(h264_file_path, "rb") as f:
                raw_bytes = f.read()
            
            # THE LATIN-1 TRICK: Crucial for binary-to-text mapping required by HuggingFace Tokenizers
            text_data = raw_bytes.decode("latin-1")
            
            # Tokenize the binary 'text'
            encoding = self.tokenizer.encode(text_data)
            return encoding.ids
            
        except Exception as e:
            print(f"❌ [AVC-LM] Bitstream Encoding error: {e}")
            return []

    def encode_mp4_segment(self, mp4_file_path, start_sec, duration_sec):
        """
        Extracts a specific segment of the MP4 into an H.264 stream and tokenizes it.
        """
        if not os.path.exists(mp4_file_path):
            return []

        temp_h264_path = f"temp_segment_{start_sec}.h264"

        try:
            # -ss: start time, -t: duration
            command = [
                FFMPEG_BIN, "-y", 
                "-ss", str(start_sec),
                "-i", mp4_file_path, 
                "-t", str(duration_sec),
                "-vf", "scale=256:256,fps=30", # Force 30 FPS here as well 
                "-c:v", "libx264",       
                "-crf", "40",            
                "-preset", "ultrafast",  
                "-an",                   
                temp_h264_path
            ]
            
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            tokens = self.encode_h264_bitstream(temp_h264_path)
            
            if os.path.exists(temp_h264_path):
                os.remove(temp_h264_path)
                
            return tokens
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [AVC-LM] Segment extraction failed. Error: {e}")
            return []


# =====================================================================
# VLA DATA PIPELINE (The Interleaver)
# =====================================================================

class VLADatasetBuilder:
    """
    Streams raw FineVideo data, parses it into the target hierarchical structure,
    extracts 30FPS frames per activity, generates multimodal video tokens, 
    and saves the final JSONL output.
    """
    def __init__(self, video_folder="./videos", jsonl_folder = "./metadata", overlap_threshold=0.2):
        print("\n🏗️  Initializing VLA Dataset Builder & Metadata Parser...")
        self.seed2 = Seed2Tokenizer()
        self.cosmos = CosmosVideoTokenizer()
        self.avc_lm = AVCLMTokenizer()
        self.video_folder = video_folder
        self.jsonl_folder = jsonl_folder
        self.target_fps = 30
        self.overlap_threshold = overlap_threshold
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.jsonl_folder, exist_ok=True)

    # --- Cleaning temp data

    def cleanup_temp_data(self):
        """
        Clears all contents inside the videos, metadata, and temp_frames directories
        to ensure no residual files consume disk space after processing a video.
        """
        folders_to_clean = [self.video_folder, self.jsonl_folder, "temp_frames"]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path) # Remove standard files
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path) # Remove subdirectories if any
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to delete {file_path}. Reason: {e}")

    # --- METADATA PARSING METHODS ---

    @staticmethod
    def time_to_seconds(time_str):
        """Converts 'HH:MM:SS.mmm' to float seconds."""
        if not time_str: return 0.0
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0

    @staticmethod
    def calculate_overlap(start1, end1, start2, end2):
        """Calculates overlap duration in seconds between two intervals."""
        return max(0.0, min(end1, end2) - max(start1, start2))

    def parse_video_metadata(self, raw_data):
        """
        Converts raw FineVideo JSON into the design.jsonl hierarchical format.
        Applies multi-assignment logic for props, editing, and speech.
        """
        # Generate an ID or use the original filename
        video_id = raw_data.get("original_video_filename", "unknown").replace(".mp4", "")
        if video_id == "unknown":
            video_id = raw_data.get("youtube_title", "video").replace(" ", "_").lower()

        content_meta = raw_data.get("content_metadata", {})
        
        # Build the foundational target structure
        design_obj = {
            "video_id": video_id,
            "metadata": {
                "youtube_title": raw_data.get("youtube_title", ""),
                "category": raw_data.get("content_fine_category", ""),
                "resolution": raw_data.get("resolution", ""),
                "fps": content_meta.get("fps", 30.0),
                "duration_sec": raw_data.get("duration_seconds", 0)
            },
            "global_context": content_meta.get("description", ""),
            "scenes": []
        }

        raw_scenes = content_meta.get("scenes", [])
        timecoded_speech = raw_data.get("timecoded_text_to_speech", [])

        # Process Scenes
        for scene in raw_scenes:
            scene_start = self.time_to_seconds(scene.get("timestamps", {}).get("start_timestamp", ""))
            scene_end = self.time_to_seconds(scene.get("timestamps", {}).get("end_timestamp", ""))
            
            parsed_scene = {
                "scene_id": scene.get("sceneId", 0),
                "scene_title": scene.get("title", ""),
                "scene_time_range_sec": [round(scene_start, 3), round(scene_end, 3)],
                "scene_thematic": scene.get("thematicElements", ""),
                "scene_mood": scene.get("mood", {}).get("description", ""),
                "activities": []
            }

            # Process Activities
            for act_idx, activity in enumerate(scene.get("activities", [])):
                act_start = self.time_to_seconds(activity.get("timestamp", {}).get("start_timestamp", ""))
                act_end = self.time_to_seconds(activity.get("timestamp", {}).get("end_timestamp", ""))
                
                parsed_act = {
                    "activity_id": f"scene_{parsed_scene['scene_id']}_act_{act_idx + 1}",
                    "time_range_sec": [round(act_start, 3), round(act_end, 3)],
                    "text_prompt": activity.get("description", ""),
                    "speech_transcript": "", 
                    "props_present": [],
                    "video_editing": [],
                    "video_tokens": "" # Placeholder for the tokenizer pipeline
                }
                parsed_scene["activities"].append(parsed_act)

            # Map Props and Editing via Temporal Overlap
            for parsed_act in parsed_scene["activities"]:
                a_start, a_end = parsed_act["time_range_sec"]
                
                for prop in scene.get("props", []):
                    p_start = self.time_to_seconds(prop.get("timestamp", {}).get("start_timestamp", ""))
                    p_end = self.time_to_seconds(prop.get("timestamp", {}).get("end_timestamp", ""))
                    if self.calculate_overlap(a_start, a_end, p_start, p_end) >= self.overlap_threshold:
                        parsed_act["props_present"].append(prop.get("name", ""))
                
                for edit in scene.get("videoEditingDetails", []):
                    e_start = self.time_to_seconds(edit.get("timestamps", {}).get("start_timestamp", ""))
                    e_end = self.time_to_seconds(edit.get("timestamps", {}).get("end_timestamp", ""))
                    if self.calculate_overlap(a_start, a_end, e_start, e_end) >= self.overlap_threshold:
                        parsed_act["video_editing"].append(edit.get("description", ""))

            design_obj["scenes"].append(parsed_scene)

        # Map Speech globally across all scenes
        for speech in timecoded_speech:
            s_start = self.time_to_seconds(speech.get("start", ""))
            s_end = self.time_to_seconds(speech.get("end", ""))
            for parsed_scene in design_obj["scenes"]:
                for parsed_act in parsed_scene["activities"]:
                    a_start, a_end = parsed_act["time_range_sec"]
                    if self.calculate_overlap(s_start, s_end, a_start, a_end) >= self.overlap_threshold:
                        parsed_act["speech_transcript"] += speech.get("text", "") + " "

        # Cleanup trailing spaces
        for parsed_scene in design_obj["scenes"]:
            for parsed_act in parsed_scene["activities"]:
                parsed_act["speech_transcript"] = parsed_act["speech_transcript"].strip()

        return design_obj


    # --- VIDEO PROCESSING METHODS ---

    def extract_30fps_frames(self, video_path, start_sec, end_sec):
        """Extracts exactly 30 frames per second using FFmpeg."""
        duration = end_sec - start_sec
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))

        try:
            command = [
                FFMPEG_BIN, "-y", "-ss", str(start_sec), "-i", video_path,
                "-t", str(duration), "-r", str(self.target_fps), "-f", "image2",
                os.path.join(temp_dir, "frame_%04d.png")
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            frames = []
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".png")])
            for file in frame_files:
                img_path = os.path.join(temp_dir, file)
                img = Image.open(img_path).convert("RGB")
                img_copy = img.copy() 
                frames.append(img_copy)
                img.close()
            return frames
        except Exception as e:
            print(f"❌ Frame extraction failed for {start_sec}-{end_sec}: {e}")
            return []

    def tokenize_activity_frames(self, frames, video_path, start_sec, duration):
        """Interleaves Seed2, Cosmos, and AVC-LM tokens."""
        all_formatted_tokens = []
        activity_token_count = 0
        
        # 1. SEED2
        if len(frames) > 0:
            temp_path = "temp_seed2_frame.jpg"
            frames[0].resize((self.seed2.target_size, self.seed2.target_size)).save(temp_path)
            seed2_ids = self.seed2.encode_image(temp_path)
            if seed2_ids:
                all_formatted_tokens.append(f"<seed2> {' '.join(map(str, seed2_ids))} </seed2>")
                activity_token_count += len(seed2_ids)
            if os.path.exists(temp_path): os.remove(temp_path)

        # 2. COSMOS
        buffer = []
        for frame in frames:
            buffer.append(frame)
            if len(buffer) == 8:
                cosmos_ids = self.cosmos.encode_video_chunk(buffer)
                if cosmos_ids:
                    all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, cosmos_ids))} </cosmos>")
                    activity_token_count += len(cosmos_ids)
                buffer.clear()
        
        # Cosmos Padding
        if len(buffer) > 0:
            missing_frames = 8 - len(buffer)
            last_frame = buffer[-1]
            for _ in range(missing_frames): buffer.append(last_frame)
            cosmos_ids = self.cosmos.encode_video_chunk(buffer)
            if cosmos_ids:
                all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, cosmos_ids))} </cosmos>")
                activity_token_count += len(cosmos_ids)

        # 3. AVC-LM
        avc_ids = self.avc_lm.encode_mp4_segment(video_path, start_sec, duration)
        if avc_ids:
            all_formatted_tokens.append(f"<avc_lm> {' '.join(map(str, avc_ids))} </avc_lm>")
            activity_token_count += len(avc_ids)

        return " ".join(all_formatted_tokens), activity_token_count

    # --- ORCHESTRATION ---

    def process_pipeline(self, output_jsonl):
        """
        Streams from HuggingFace, parses metadata, processes video, and writes output.
        """
        print(f"\n🚀 Starting End-to-End Pipeline. Output will be saved to: {output_jsonl}")

        try:
            dataset = load_dataset(
                DATASET_NAME, 
                split="train", 
                cache_dir=SCRATCH_CACHE_DIR,
                token=HF_TOKEN
            )
            dataset_iter = iter(dataset)
        except Exception as e:
            print(f"❌ Failed to load dataset from HuggingFace: {e}")
            return

        total_processed = 0
        pbar = tqdm(total=MAX_VIDEOS, desc="Processing Videos")
        
        while total_processed < MAX_VIDEOS:
            try:
                # 1. Fetch raw data from stream
                item = next(dataset_iter)
                raw_metadata = item['json']
                video_bytes = item['mp4']

                # 2. Parse metadata into design.jsonl format
                design_obj = self.parse_video_metadata(raw_metadata)
                video_id = design_obj["video_id"]
                
                # 3. Temporarily save video bytes to disk for FFmpeg processing
                temp_video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)

                temp_jsonl_path = os.path.join(self.jsonl_folder, f"{video_id}.jsonl")
                with open(temp_jsonl_path, "w", encoding="utf-8") as f:
                    json.dump(raw_metadata, f, ensure_ascii=False)

                # print(f"\n🎬 Tokenizing video: {video_id}...")
                video_total_tokens = 0
                video_total_duration = 0.0

                # 4. Tokenize each activity
                for scene in design_obj.get("scenes", []):
                    for activity in scene.get("activities", []):
                        act_id = activity.get("activity_id", "unknown")
                        time_range = activity.get("time_range_sec", [0.0, 0.0])
                        start_sec = time_range[0]
                        end_sec = time_range[1]
                        duration = end_sec - start_sec
                        
                        if duration > 0: video_total_duration += duration

                        # print(f"  -> Activity {act_id} ({start_sec}s - {end_sec}s)")
                        
                        frames = self.extract_30fps_frames(temp_video_path, start_sec, end_sec)
                        if frames:
                            interleaved_tokens, act_token_count = self.tokenize_activity_frames(
                                frames, temp_video_path, start_sec, duration
                            )
                            # Inject tokens into the parsed design object
                            activity["video_tokens"] = interleaved_tokens
                            video_total_tokens += act_token_count
                        else:
                            activity["video_tokens"] = ""

                # 5. Print Stats
                tokens_per_sec = video_total_tokens / video_total_duration if video_total_duration > 0 else 0
                print(f"🎬 Tokenized {total_processed + 1}/{MAX_VIDEOS} videos...")
                print(f"📊 --- STATS: {video_id} ---")
                print(f"✅ Total Duration  : {video_total_duration:.2f} seconds")
                print(f"✅ Total Tokens    : {video_total_tokens:,} tokens")
                print(f"✅ Density         : {tokens_per_sec:.2f} tokens/second")

                # 6. Save to JSONL (Flattened to 1 line)
                with open(output_jsonl, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(design_obj, ensure_ascii=False, separators=(',', ':')) + "\n")

                # 7. Cleanup temp video file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

                total_processed += 1
                pbar.update(1)

            except StopIteration:
                print("End of dataset reached.")
                break
            except Exception as e:
                print(f"❌ Error processing item: {e}")
                # The continue statement will still trigger the finally block before looping
                continue 
            finally:
                # 7. Comprehensive Cleanup
                # This executes GUARANTEED after every video, success or fail
                self.cleanup_temp_data()
                
        pbar.close()
        print(f"\n✨ End-to-End processing complete! Data saved to: {output_jsonl}")


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    builder = VLADatasetBuilder(video_folder="./videos", jsonl_folder = "./metadata", overlap_threshold=0.2)
    builder.process_pipeline(output_jsonl="training_ready.jsonl")