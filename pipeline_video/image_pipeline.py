import io
import os
import sys
import json
import re
import tarfile
import math
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from tokenizers import ByteLevelBPETokenizer
import torchvision.transforms as T
from cosmos_tokenizer.image_lib import ImageTokenizer
from bs4 import BeautifulSoup

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to your downloaded stackexchange tar.gz file
TAR_PATH = "askubuntu.com.tar.gz" 
OUTPUT_JSONL = "askubuntu_prototype_unified.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Limit for prototype (stop after processing this many QA pairs)
PROTOTYPE_LIMIT = 50

# ==============================================================================
# TOKENIZER WRAPPERS
# ==============================================================================

class Seed2Tokenizer:
    def __init__(self):
        self.folder = "./seed2" # Ensure this folder exists
        self.tokenizer = None
        self.load_tokenizer()

    def load_tokenizer(self):        
        if os.path.exists(self.folder):
            sys.path.append(self.folder)
        
        # Try importing Seed2
        try:
            from seed2_tokenizer import Seed2Tokenizer
            print("📦 [Seed2] Initializing Tokenizer...")
            self.tokenizer = Seed2Tokenizer.from_pretrained(
                self.folder, 
                torch_dtype=DTYPE
            ).to(DEVICE)
        except Exception as e:
            print(f"⚠️ [Seed2] Error loading tokenizer: {e}")

    def tokenize(self, image_path):
        if self.tokenizer is None: return []
        try:
            with torch.no_grad():
                tokens = self.tokenizer.encode_image(image_path) 
                if torch.is_tensor(tokens):
                    return tokens.flatten().detach().cpu().numpy().tolist()
                return tokens
        except Exception as e:
            print(f"❌ [Seed2] Tokenize error: {e}")
            return []


class JPEGLMTokenizer:
    def __init__(self):
        self.folder = 'jpeg_tokenizer' # Ensure you have vocab.json/merges.txt here
        self.tokenizer = None
        self.img_size = (128, 128) 
        self.quality = 15          
        self.sos_marker = b'\xff\xda'
        self.load_tokenizer()

    def load_tokenizer(self):
        vocab_path = os.path.join(self.folder, "vocab.json")
        merges_path = os.path.join(self.folder, "merges.txt")
        
        if os.path.exists(vocab_path):
            print(f"📦 [JPEG-LM] Loading Tokenizer...")
            self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        else:
            print(f"⚠️ [JPEG-LM] Config not found at {vocab_path}")

    def _get_jpeg_body(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.img_size)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=self.quality, subsampling=2)
            raw_bytes = buf.getvalue()
            idx = raw_bytes.find(self.sos_marker)
            return raw_bytes[idx:] if idx != -1 else raw_bytes
        except Exception:
            return None

    def tokenize(self, image_path):
        if not self.tokenizer: return []
        jpeg_body = self._get_jpeg_body(image_path)
        if jpeg_body:
            # Decode to latin-1 to preserve byte values
            jpeg_str = jpeg_body.decode('latin-1')
            encoded = self.tokenizer.encode(jpeg_str)
            return encoded.ids
        return []

    def decode_and_save(self, token_ids, output_filename="jpeg_reconstructed.jpg"):
        """
        Decodes BPE tokens back to bytes. Since the original encoder stripped
        the JPEG header (keeping only the SOS marker), we inject a dummy header.
        """
        print(f"[JPEG-LM] Decoding {len(token_ids)} tokens...")
        
        # 1. Decode tokens back to a string, preventing space removal
        decoded_str = self.tokenizer.decode(token_ids)
        
        # 2. Convert string back to raw bytes (latin-1 preserves 0-255 values)
        jpeg_entropy_bytes = decoded_str.encode('latin-1')
        
        # 3. Dummy Header Trick
        # Create a blank 256x256 image with the exact same compression settings
        # to steal its valid JPEG header (everything before the SOS marker).
        dummy_img = Image.new("RGB", self.img_size, color="black")
        buf = io.BytesIO()
        dummy_img.save(buf, format='JPEG', quality=self.quality, subsampling=2)
        dummy_bytes = buf.getvalue()
        
        # Find SOS (Start of Scan) marker
        sos_marker = b'\xff\xda'
        idx = dummy_bytes.find(sos_marker)
        
        if idx == -1:
            print("❌ [JPEG-LM] Could not generate dummy header.")
            return
            
        valid_header = dummy_bytes[:idx]
        
        # Combine the valid dummy header with our decoded entropy data
        full_jpeg_bytes = valid_header + jpeg_entropy_bytes
        
        try:
            # 4. Save the reconstructed bytes as an image
            img = Image.open(io.BytesIO(full_jpeg_bytes))
            img.save(output_filename)
            print(f"✅ [JPEG-LM] Image saved successfully to {output_filename}")
        except Exception as e:
            print(f"⚠️ [JPEG-LM] PIL could not parse the image. It might have visual artifacts: {e}")
            # Save raw bytes anyway so you can inspect it with an image viewer
            fallback_path = output_filename.replace(".jpg", "_raw.jpg")
            with open(fallback_path, "wb") as f:
                f.write(full_jpeg_bytes)
            print(f"✅ [JPEG-LM] Saved raw bytes to {fallback_path}")

class CosmosImageTokenizer:
    def __init__(self):
        # Ensure this path matches your downloaded checkpoint
        self.tokenizer_path = "pretrained_ckpts/Cosmos-Tokenizer-DI16x16/encoder.jit"
        self.decoder_path = "pretrained_ckpts/Cosmos-Tokenizer-DI16x16/decoder.jit"
        self.encoder = None
        self.load_tokenizer()
        self.decoder = None
        self.load_decoder()

    def load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            print(f"⚠️ [Cosmos] Checkpoint not found: {self.tokenizer_path}")
            return
        
        print(f"📦 [Cosmos] Loading Tokenizer...")
        try:
            self.encoder = ImageTokenizer(checkpoint_enc=self.tokenizer_path, device=DEVICE)
            if hasattr(self.encoder, 'to'):
                self.encoder.to(DEVICE).to(DTYPE).eval()
        except Exception as e:
            print(f"⚠️ [Cosmos] Error loading: {e}")

    def tokenize(self, image_path):
        if self.encoder is None: return []
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((256, 256)) # Cosmos standard input
            tensor = T.ToTensor()(img).unsqueeze(0) # [1, 3, H, W]
            tensor = (tensor * 2.0) - 1.0 # Normalize [-1, 1]
            tensor = tensor.to(DEVICE).to(DTYPE)

            with torch.no_grad():
                # Cosmos DI returns (indices, codes). We use indices.
                indices, _ = self.encoder.encode(tensor)
            
            return indices.flatten().detach().cpu().numpy().tolist()
        except Exception as e:
            print(f"❌ [Cosmos] Tokenize error: {e}")
            return []
    
    def load_decoder(self):
        self.decoder = ImageTokenizer(checkpoint_dec=self.decoder_path, device=DEVICE)
        if hasattr(self.decoder, 'to'):
            self.decoder.to(DEVICE).to(DTYPE).eval()
    
    def decode_and_save(self, indices_list, output_filename="cosmos_reconstructed.jpg"):
        """
        Takes a flat list of tokens (e.g., 1024 tokens), reshapes them,
        runs the decoder, denormalizes the tensor, and saves the image.
        """
        print(f"[Cosmos] Decoding {len(indices_list)} tokens...")
        
        # Calculate grid size (e.g., sqrt(1024) = 32 -> 32x32 grid)
        grid_size = int(math.sqrt(len(indices_list)))
        
        # Convert list to tensor and reshape to (Batch, Height, Width)
        indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=DEVICE)
        indices_tensor = indices_tensor.view(1, grid_size, grid_size)
        
        try:
            with torch.no_grad():
                # The decoder outputs a tensor in range [-1, 1]
                reconstructed_tensor = self.decoder.decode(indices_tensor)
            
            # Denormalize: Convert from [-1, 1] back to [0, 1] range for image saving
            reconstructed_tensor = (reconstructed_tensor + 1.0) / 2.0
            reconstructed_tensor = reconstructed_tensor.clamp(0, 1)
            
            # Convert to PIL Image and save
            img = T.ToPILImage()(reconstructed_tensor.squeeze(0).cpu().float())
            img.save(output_filename)
            print(f"✅ [Cosmos] Image saved successfully to {output_filename}")
            
        except Exception as e:
            print(f"❌ [Cosmos] Decoding failed: {e}")



# ==============================================================================
# MAIN PROCESSOR
# ==============================================================================

class DataProcessor:
    def __init__(self):
        print("🚀 Initializing Models...")
        self.seed2 = Seed2Tokenizer() 
        self.jpeg_lm = JPEGLMTokenizer()
        self.cosmos = CosmosImageTokenizer()
        
        print(f"📂 Opening Tar File: {TAR_PATH}")
        self.tar = tarfile.open(TAR_PATH, "r:gz")
        self.question_cache = {}
        self.root_prefix = self._find_root_prefix()
        print(f"🔍 Detected Root Prefix: '{self.root_prefix}'")

    def _find_root_prefix(self):
        try:
            for member in self.tar:
                if member.isdir(): return member.name 
                if '/' in member.name: return member.name.split('/')[0]
            return "" 
        except Exception: return ""

    def get_image_bytes(self, filename):
        full_path = os.path.join(self.root_prefix, filename) if self.root_prefix else filename
        try:
            member = self.tar.getmember(full_path)
            f = self.tar.extractfile(member)
            return f.read()
        except KeyError:
            try:
                member = self.tar.getmember(filename)
                f = self.tar.extractfile(member)
                return f.read()
            except KeyError: return None
        except Exception: return None

    def clean_html_text(self, html_content):
        """Strip HTML tags and return clean plain text for body_cleaned field."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            return soup.get_text(separator="\n").strip()
        except:
            return html_content

    def process_content_with_images(self, text_html, source_type="unknown"):
        """
        Returns two values:
        1. Text with image tokens interleaved (for the 'text' field)
        2. List of image metadata dicts (for the 'images_meta' field)
        """
        if not text_html: return "", []

        html_pattern = r'<img[^>]+src="([^">]+)"[^>]*>'
        md_pattern = r'!\[.*?\]\((.*?)\)'
        combined_pattern = f'({html_pattern})|({md_pattern})'
        
        matches = list(re.finditer(combined_pattern, text_html))
        new_text = text_html
        processed_urls = set()
        
        images_meta_list = []

        for match in matches:
            full_tag = match.group(0)
            img_url = match.group(2) if match.group(2) else match.group(4)
            
            if not img_url: continue
            filename = os.path.basename(img_url)
            
            if full_tag in processed_urls: continue
            processed_urls.add(full_tag)

            images_meta_list.append({
                "filename": filename,
                "source": source_type,  # "question_body" or "answer_body"
                "original_url": img_url
            })

            # Tokenize image
            img_bytes = self.get_image_bytes(filename)
            if img_bytes:
                temp_path = f"temp_{filename}"
                with open(temp_path, "wb") as f: f.write(img_bytes)
                
                try:
                    t_jpeg = self.jpeg_lm.tokenize(temp_path)
                    t_seed2 = self.seed2.tokenize(temp_path)
                    t_cosmos = self.cosmos.tokenize(temp_path)
                    
                    if len(t_seed2) > 0 or len(t_cosmos) > 0:
                        token_block = (
                            f" <BOI> "
                            f"<jpeg_start> {json.dumps(t_jpeg)} <jpeg_end> "
                            f"<seed2_start> {json.dumps(t_seed2)} <seed2_end> "
                            f"<cosmos_start> {json.dumps(t_cosmos)} <cosmos_end> "
                            f"<EOI> "
                        )
                        new_text = new_text.replace(full_tag, token_block)
                except Exception as e:
                    print(f"❌ Tokenize Fail: {filename} - {e}")
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)
            
        return new_text, images_meta_list

    def run_pipeline(self):
        print("🔄 Finding Posts.xml...")
        posts_member = None
        target_name = os.path.join(self.root_prefix, "Posts.xml") if self.root_prefix else "Posts.xml"
        try: posts_member = self.tar.getmember(target_name)
        except KeyError:
            for member in self.tar.getmembers():
                if member.name.endswith("Posts.xml"):
                    posts_member = member; break
        
        if not posts_member: return

        print(f"📄 Processing: {posts_member.name}")
        f_xml = self.tar.extractfile(posts_member)
        
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
            context = ET.iterparse(f_xml, events=("end",))
            count = 0
            
            for event, elem in context:
                if elem.tag == "row":
                    attr = elem.attrib
                    post_type = attr.get("PostTypeId")
                    
                    # --- QUESTION ---
                    if post_type == "1":
                        post_id = attr.get("Id")
                        body = attr.get("Body", "")
                        
                        proc_body, imgs_meta = self.process_content_with_images(body, source_type="question_body")

                        self.question_cache[post_id] = {
                            "title": attr.get("Title", ""),
                            "body_interleaved": proc_body,
                            "body_original": body,
                            "images_meta": imgs_meta,
                            "tags": attr.get("Tags"),
                            "original_attr": attr
                        }
                        
                    # --- ANSWER ---
                    elif post_type == "2":
                        parent_id = attr.get("ParentId")
                        
                        if parent_id in self.question_cache:
                            q_data = self.question_cache[parent_id]
                            
                            a_body = attr.get("Body", "")
                            proc_a_body, a_imgs_meta = self.process_content_with_images(a_body, source_type="answer_body")
                            
                            unified_record = {
                                "id": f"qa_{attr.get('Id')}",
                                "text": (
                                    f"Question: {q_data['title']}\n"
                                    f"{q_data['body_interleaved']}\n"
                                    f"Answer:\n"
                                    f"{proc_a_body}"
                                ),
                                "metadata": {
                                    "domain": TAR_PATH.replace(".tar.gz", ""),
                                    "question": {
                                        "post_id": parent_id,
                                        "title": q_data['title'],
                                        "body_cleaned": self.clean_html_text(q_data['body_original']),
                                        "tags": q_data['tags'],
                                        "original_xml_attributes": q_data['original_attr']
                                    },
                                    "answer": {
                                        "post_id": attr.get("Id"),
                                        "body_cleaned": self.clean_html_text(a_body),
                                        "original_xml_attributes": attr
                                    },
                                    "images_meta": q_data['images_meta'] + a_imgs_meta
                                },
                                "interleaved_format_description": "text contains interleaved: question || answer || image_tokens (jpeg_bpe + seed2 + cosmos)"
                            }
                            
                            json.dump(unified_record, out_f)
                            out_f.write("\n")
                            out_f.flush()
                            
                            count += 1
                            if count % 10 == 0: print(f"✅ Processed {count} pairs...")
                            if count >= PROTOTYPE_LIMIT: break

                    elem.clear()

if __name__ == "__main__":
    try:
        processor = DataProcessor()
        processor.run_pipeline()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    # cosmos = CosmosImageTokenizer()
    # jpeg_lm = JPEGLMTokenizer()
    # tokens = cosmos.tokenize('cat.jpg')
    # print(f'{len(tokens)} Cosmos Tokens: {tokens}')
    # cosmos.decode_and_save(tokens)
    # tokens = jpeg_lm.tokenize('cat.jpg')
    # print(f'{len(tokens)} JPEG-LM Tokens: {tokens}')
    # jpeg_lm.decode_and_save(tokens)