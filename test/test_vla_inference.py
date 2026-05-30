import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla/vla_25b_test/hf/iter_0006000"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "USER: Athlete sits on a bench and talks about his goals. ASSISTANT: <seed2> <seed2_"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("🚀 Probing the model's output distribution for cutting tokens...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10, 
        do_sample=False
    )

print("\n📊 MODEL GENERATION:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))