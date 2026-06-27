import os
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
import imageio

# Global constants for PyTorch hardware acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

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

    def encode_mp4(self, video_path, fps_rate=1):
        """
        Video processing router.
        Extracts frames at a specific rate (default 1 FPS) and tokenizes them.
        """
        print(f"🎬 [Seed2] Extracting semantic frames from: '{video_path}' at {fps_rate} FPS")
        if not os.path.exists(video_path):
            print(f"❌ [Seed2] Video file not found: {video_path}")
            return ""

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Failsafe for corrupted video metadata
        if original_fps <= 0: 
            print(f"❌ Fail to get video metadata!")
            original_fps = 30 

        frame_skip_interval = int(round(original_fps / fps_rate))
        current_frame_idx = 0
        all_formatted_tokens = []
        total_token_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break # End of video
            
            # Process exactly on the specified interval (e.g., every 30th frame)
            if current_frame_idx % frame_skip_interval == 0:
                # 1. Convert OpenCV BGR to standard RGB PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((self.target_size, self.target_size))
                
                # 2. Call the core image tokenizer
                temp_path = "temp_frame.jpg"
                pil_img.save(temp_path)
                token_ids = self.encode_image(temp_path)
                total_token_count += len(token_ids)
                
                # 3. Format into string with structural tags (<BOI> = Begin Of Image)
                token_string = f"<BOI> <seed2> {' '.join(map(str, token_ids))} </seed2> <EOI>"
                all_formatted_tokens.append(token_string)
                
            current_frame_idx += 1

        cap.release()
        print(f"✅ [Seed2] Extracted and tokenized {len(all_formatted_tokens)} frames with {total_token_count} tokens")
        
        # Return a single concatenated string ready for the LLM JSONL file
        return " ".join(all_formatted_tokens)

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
        self.load_decoder()

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

    def load_decoder(self):
        if not os.path.exists(self.decoder_path): return
        print(f"📦 [Cosmos] Loading Video Decoder on {DEVICE}...")
        try:
            self.decoder = CausalVideoTokenizer(checkpoint_dec=self.decoder_path)
            print("✅ [Cosmos] Decoder loaded successfully.")
        except Exception as e:
            print(f"⚠️ [Cosmos] Error loading decoder: {e}")

    def encode_video_chunk(self, frame_list, target_size=256):
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

    def encode_mp4(self, video_path, target_size=256):
        """
        Reads an MP4 file with STRICT 1-second (30 frames) boundaries.
        Processes in chunks of 8 frames. 
        At exactly the 30th frame, it forces padding on the leftover frames 
        (usually 6 frames) to complete the 4th chunk of that second.
        """
        print(f"🎬 [Cosmos] Extracting with STRICT 30-frame boundaries: {video_path}")
        if not os.path.exists(video_path):
            return ""

        cap = cv2.VideoCapture(video_path)
        
        buffer = []
        all_formatted_tokens = []
        global_frame_count = 0
        chunk_count = 0
        cnt_token = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video stream
                
            global_frame_count += 1
            
            # Convert OpenCV BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            buffer.append(pil_img)

            # 1. Standard processing: Encode whenever we collect a full 8-frame chunk
            if len(buffer) == 8:
                token_ids = self.encode_video_chunk(buffer, target_size)
                cnt_token += len(token_ids)
                all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, token_ids))} </cosmos>")
                chunk_count += 1
                buffer.clear()

            # 2. ENFORCE 1-SECOND BOUNDARY: Handle the 30th frame mark
            if global_frame_count % 30 == 0:
                # At this stage, the buffer usually contains the remaining 6 frames (frames 25-30)
                if len(buffer) > 0:
                    missing_frames = 8 - len(buffer)
                    last_frame = buffer[-1] # This is the 30th frame
                    
                    # Duplicate the last frame to complete the 8-frame requirement
                    for _ in range(missing_frames):
                        buffer.append(last_frame)
                    
                    # Process the 4th chunk of this specific second
                    token_ids = self.encode_video_chunk(buffer, target_size)
                    cnt_token += len(token_ids)
                    all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, token_ids))} </cosmos>")
                    chunk_count += 1
                    buffer.clear()

        # ==========================================================
        # FINAL LEFTOVER PADDING
        # Handles cases where the video duration is not a multiple of 30
        # ==========================================================
        if len(buffer) > 0:
            missing_frames = 8 - len(buffer)
            last_frame = buffer[-1]
            for _ in range(missing_frames):
                buffer.append(last_frame)
                
            token_ids = self.encode_video_chunk(buffer, target_size)
            cnt_token += len(token_ids)
            all_formatted_tokens.append(f"<cosmos> {' '.join(map(str, token_ids))} </cosmos>")
            chunk_count += 1

        cap.release()
        print(f"✅ [Cosmos] Extracted {global_frame_count} frames into {chunk_count} strictly aligned chunks with total of {cnt_token}.")
        
        return " ".join(all_formatted_tokens)

class AVCLMTokenizer:
    """
    Handles Physical/Motion Vision (AVC-LM).
    Tokenizes raw H.264 bitstreams. Can extract bitstreams directly from MP4 containers.
    """
    def __init__(self, vocab_dir="avc-lm"):
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

    def encode_mp4(self, mp4_file_path):
        """
        Wrapper function: Extracts the H.264 stream from an MP4 file, tokenizes it, 
        and cleans up the temporary files.
        """
        print(f"🎬 [AVC-LM] Processing MP4: {mp4_file_path}")
        if not os.path.exists(mp4_file_path):
            print(f"❌ [AVC-LM] File not found: {mp4_file_path}")
            return []

        # Define a temporary file path
        temp_h264_path = "temp_extracted_stream.h264"

        try:
            # Command to extract raw H.264 stream from MP4 without re-encoding
            # -c:v copy: Just copy the video stream (super fast)
            # -bsf:v h264_mp4toannexb: Convert MP4 format to raw Annex B format
            # -an: Strip audio
            command = [
                FFMPEG_BIN, "-y", "-i", mp4_file_path, 
                "-vf", "scale=256:256",  
                "-c:v", "libx264",       
                "-crf", "40",            
                "-preset", "ultrafast",  
                "-an",                   
                temp_h264_path
            ]
            
            # Suppress ffmpeg terminal output for cleaner logs
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Tokenize the newly extracted temporary file
            tokens = self.encode_h264_bitstream(temp_h264_path)
            
            # Clean up the temporary file
            if os.path.exists(temp_h264_path):
                os.remove(temp_h264_path)
                
            print(f"✅ [AVC-LM] Extracted and tokenized {len(tokens)} tokens from MP4.")
            return tokens
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [AVC-LM] FFmpeg extraction failed. Is ffmpeg installed? Error: {e}")
            return []
        except Exception as e:
            print(f"❌ [AVC-LM] MP4 Processing error: {e}")
            return []

class FullVideoPipeline:
    """
    Orchestrates the synchronization of Seed2 (1Hz), Cosmos (~4Hz), 
    and AVC-LM (30Hz) into a single, interleaved timeline for the LLM.
    """
    def __init__(self):
        print("\n🏗️  Initializing Full VLA Video Pipeline...")
        self.seed2 = Seed2Tokenizer()
        self.cosmos = CosmosVideoTokenizer()
        self.avc_lm = AVCLMTokenizer()
        print("✅ All 3 Tokenizers loaded and armed.")

    def process_video(self, video_path, output_jsonl="vla_dataset.jsonl"):
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return

        print(f"\n🚀 Starting Grand Merger Pipeline for: '{video_path}'")
        
        # 1. Pre-calculate AVC-LM tokens for the entire video bitstream
        print("⏳ Pre-processing AVC-LM bitstream (Physical Motion)...")
        all_avc_tokens = self.avc_lm.encode_mp4(video_path)
        
        # 2. Open video for frame-by-frame processing
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"❌ Error: No frame!")
            total_frames = 1 # Failsafe
            
        # Calculate roughly how many AVC tokens belong to each physical frame (~33ms)
        avc_tokens_per_frame = len(all_avc_tokens) // total_frames
        
        buffer_cosmos = []
        vision_tokens_stream = ["<BOV>"]
        
        global_frame_count = 0
        chunk_count_cosmos = 0
        total_seed2_calls = 0

        print(f"🎬 Processing {total_frames} frames in a single synchronized loop...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            global_frame_count += 1
            
            # --- A. SEED2 LOGIC (1Hz) ---
            # Trigger exactly on the first frame of every second (frame 1, 31, 61...)
            if (global_frame_count - 1) % 30 == 0:
                temp_path = "temp_pipeline_frame.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((self.seed2.target_size, self.seed2.target_size))
                pil_img.save(temp_path)
                
                token_ids_seed2 = self.seed2.encode_image(temp_path)
                total_seed2_calls += 1
                vision_tokens_stream.append(f"<seed2> {' '.join(map(str, token_ids_seed2))} </seed2>")

            # --- B. COSMOS LOGIC (~4Hz) ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            buffer_cosmos.append(pil_img)

            # Standard 8-frame chunk
            if len(buffer_cosmos) == 8:
                token_ids_cosmos = self.cosmos.encode_video_chunk(buffer_cosmos)
                vision_tokens_stream.append(f"<cosmos> {' '.join(map(str, token_ids_cosmos))} </cosmos>")
                chunk_count_cosmos += 1
                buffer_cosmos.clear()

            # Enforce strict 1-second boundary padding
            if global_frame_count % 30 == 0:
                if len(buffer_cosmos) > 0:
                    missing_frames = 8 - len(buffer_cosmos)
                    last_frame = buffer_cosmos[-1]
                    for _ in range(missing_frames):
                        buffer_cosmos.append(last_frame)
                    
                    token_ids_cosmos = self.cosmos.encode_video_chunk(buffer_cosmos)
                    vision_tokens_stream.append(f"<cosmos> {' '.join(map(str, token_ids_cosmos))} </cosmos>")
                    chunk_count_cosmos += 1
                    buffer_cosmos.clear()

            # --- C. AVC-LM LOGIC (30Hz) ---
            # Extract the slice of tokens corresponding to this specific frame
            start_idx = (global_frame_count - 1) * avc_tokens_per_frame
            # For the very last frame, take all remaining tokens to avoid dropping any
            if global_frame_count == total_frames:
                end_idx = len(all_avc_tokens)
            else:
                end_idx = start_idx + avc_tokens_per_frame
                
            frame_avc_tokens = all_avc_tokens[start_idx:end_idx]
            if frame_avc_tokens:
                vision_tokens_stream.append(f"<avc_lm> {' '.join(map(str, frame_avc_tokens))} </avc_lm>")

            # Log progress every second
            if global_frame_count % 30 == 0:
                print(f"  [Progress] Synchronized {global_frame_count // 30} second(s) of video...")

        # --- FINAL LEFTOVER PADDING FOR COSMOS ---
        if len(buffer_cosmos) > 0:
            missing_frames = 8 - len(buffer_cosmos)
            last_frame = buffer_cosmos[-1]
            for _ in range(missing_frames):
                buffer_cosmos.append(last_frame)
                
            token_ids_cosmos = self.cosmos.encode_video_chunk(buffer_cosmos)
            vision_tokens_stream.append(f"<cosmos> {' '.join(map(str, token_ids_cosmos))} </cosmos>")
            chunk_count_cosmos += 1

        cap.release()
        vision_tokens_stream.append("</EOV>")
        
        # Cleanup temp file
        if os.path.exists("temp_pipeline_frame.jpg"):
            os.remove("temp_pipeline_frame.jpg")

        # --- JSONL ASSEMBLY ---
        final_vision_string = " ".join(vision_tokens_stream)
        
        result_entry = {
            "id": os.path.basename(video_path),
            "source": "Local_Prototype",
            "vision": final_vision_string,
            "text": "Action description placeholder.",
            "state": None,
            "action": None
        }

        with open(output_jsonl, 'a') as f:
            f.write(json.dumps(result_entry) + "\n")
            
        print("\n📊 --- FINAL PIPELINE STATS ---")
        print(f"✅ Total Frames Processed : {global_frame_count}")
        print(f"✅ Seed2 (1Hz) Calls      : {total_seed2_calls} (Semantic frames)")
        print(f"✅ Cosmos (~4Hz) Chunks   : {chunk_count_cosmos} (Spatio-temporal tubelets)")
        print(f"✅ AVC-LM (30Hz) Tokens   : {len(all_avc_tokens)} (Physical bitstream)")
        print(f"✨ Successfully saved synchronized metadata to '{output_jsonl}'")

class CosmosCompressionTester:
    """
    Test script to compress video using Cosmos and decompress it back to video 
    to visually verify if the quality degradation is acceptable.
    """
    def __init__(self):
        self.model_name = "Cosmos-Tokenizer-DV8x16x16"
        self.tokenizer_path = f"pretrained_ckpts/{self.model_name}/encoder.jit"
        self.decoder_path = f"pretrained_ckpts/{self.model_name}/decoder.jit"
        
        print(f"📦 [Cosmos] Loading Models on {DEVICE}...")
        self.encoder = CausalVideoTokenizer(checkpoint_enc=self.tokenizer_path)
        self.decoder = CausalVideoTokenizer(checkpoint_dec=self.decoder_path)
        print("✅ [Cosmos] Encoder & Decoder Ready.")

    def encode_decode_chunk(self, frame_list, target_size=160):
        """
        Encodes exactly 8 frames to tokens, then immediately decodes 
        them back to a pixel-space tensor.
        """
        # 1. PREPARE INPUT (Resize to 160x160)
        processed_frames = []
        transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        for img in frame_list:
            img_rgb = img.convert("RGB")
            processed_frames.append(transform(img_rgb))

        # Shape: [Batch(1), Channels(3), Time(8), Height(160), Width(160)]
        temporal_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0)
        temporal_tensor = temporal_tensor.to(DEVICE).to(DTYPE)

        with torch.no_grad():
            # 2. ENCODE: Get indices
            # indices shape will be: [1, 2, 10, 10] for target_size=160
            indices, _ = self.encoder.encode(temporal_tensor)
            token_count = indices.numel()

            # 3. DECODE: Reconstruct video directly from indices
            # reconstructed_tensor shape: [1, 3, 8, 160, 160]
            reconstructed_tensor = self.decoder.decode(indices)

        return token_count, reconstructed_tensor

    def process_and_visualize(self, video_path, output_mp4="reconstructed.mp4", target_size=160):
        print(f"🎬 Processing video for Visual Test: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        buffer = []
        total_tokens = 0
        all_reconstructed_frames = [] # To save the final video

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(Image.fromarray(frame_rgb))

            if len(buffer) == 8:
                # Encode and instantly Decode
                tokens_in_chunk, recon_tensor = self.encode_decode_chunk(buffer, target_size)
                total_tokens += tokens_in_chunk
                
                # Un-normalize back to [0, 255] for saving
                # recon_tensor shape: [1, 3, 8, 160, 160]
                recon_tensor = recon_tensor.squeeze(0).permute(1, 2, 3, 0) # -> [8, 160, 160, 3]
                recon_tensor = (recon_tensor + 1.0) / 2.0 # from [-1, 1] to [0, 1]
                recon_tensor = (recon_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                
                # Append each frame in the chunk to our final video list
                for i in range(8):
                    all_reconstructed_frames.append(recon_tensor[i])
                
                buffer.clear()

        cap.release()
        
        # Save the reconstructed frames to an MP4 file
        if all_reconstructed_frames:
            print(f"💾 Saving reconstructed video to {output_mp4}...")
            # imageio is great for saving a list of numpy arrays to video
            imageio.mimwrite(output_mp4, all_reconstructed_frames, fps=30, quality=8)
            
            # Print the stats for your report
            frames = len(all_reconstructed_frames)
            seconds = frames / 30.0
            print(f"\n📊 --- COMPRESSION STATS ({target_size}x{target_size}) ---")
            print(f"✅ Total Frames    : {frames}")
            print(f"✅ Total Tokens    : {total_tokens}")
            print(f"✅ Tokens/Second   : {total_tokens / seconds:.2f} (Target: < 1000)")
            print(f"✨ Please open '{output_mp4}' to check the visual quality!")

# =====================================================================
# USAGE EXAMPLE FOR THE MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":
    tester = CosmosCompressionTester()
    # Test with target_size=160 (which yields exactly 800 tokens/sec)
    tester.process_and_visualize("test_sample.mp4", "reconstructed_160.mp4", target_size=160)