from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "OpenMeditron/Medtrion3-Gemma2-2B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"   # safe for your laptop
)

print("Model loaded successfully")

# 🔥 Test generation
prompt = "What is diabetes?"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n Response:")
print(response)