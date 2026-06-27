---
{}
---
```
import os
import torch
if not os.path.exists("seed2"):
    os.system("git clone https://huggingface.co/ontocord/seed2")
from diffusers import StableUnCLIPImg2ImgPipeline
from seed2.seed2_tokenizer import Seed2Tokenizer
tokenizer = None 
pipe = None
torch.cuda.empty_cache()
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-unclip', torch_dtype=torch.float16)
pipe = pipe.to('cuda')
tokenizer = Seed2Tokenizer.from_pretrained("seed2", torch_dtype=torch.float16).to('cuda')
if not os.path.exists("cat.jpg"):
    os.system("wget https://images.unsplash.com/photo-1574158622682-e40e69881006?w=300 -O cat.jpg")
tokens = tokenizer.encode_image("cat.jpg")
print (tokens)
image_embeds = tokenizer.model.get_codebook_entry(tokens)
print (image_embeds) 
import time
t = time.time()
image = tokenizer.decode(pipe, tokens)[0] # this is using a fixed latent initalized by the model, so a call with the same tokens will produce the same image
print (time.time()-t)
image
```
