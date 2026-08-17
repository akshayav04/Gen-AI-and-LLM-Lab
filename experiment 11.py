import torch
import matplotlib.pyplot as plt

from transformers import pipeline
from diffusers import StableDiffusionPipeline

# ============================================================
# 3. Check GPU
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("AI CONTENT GENERATION")
print("=" * 60)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not available. Image generation will be very slow.")

# ============================================================
# 4. Load FLAN-T5 Text Generation Model
# ============================================================

print("\nLoading text generation model...")

text_generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

print("Text model loaded successfully!")

# ============================================================
# 5. Load Stable Diffusion
# ============================================================

print("\nLoading Stable Diffusion model...")
print("This may take several minutes during the first run.")

image_generator = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

image_generator = image_generator.to(device)

# Reduce GPU memory usage
if device == "cuda":
    image_generator.enable_attention_slicing()

print("Stable Diffusion model loaded successfully!")

# ============================================================
# 6. Get Topic from User
# ============================================================

topic = input("\nEnter a content topic: ")

# ============================================================
# 7. Generate Text
# ============================================================

text_prompt = f"""
Write a short article of about 120 words on the topic:
{topic}

Include:
1. Introduction
2. Importance
3. Applications
"""

print("\nGenerating text...")

text_result = text_generator(
    text_prompt,
    max_new_tokens=180,
    do_sample=False
)

generated_text = text_result[0]["generated_text"]

# ============================================================
# 8. Generate Image
# ============================================================

image_prompt = f"""
A realistic high-quality illustration representing {topic},
professional digital art, highly detailed, realistic,
cinematic lighting, 4K quality
"""

print("Generating image...")
print("Please wait...")

generated_image = image_generator(
    image_prompt,
    height=512,
    width=512,
    num_inference_steps=20
).images[0]

# ============================================================
# 9. Save Image
# ============================================================

generated_image.save("generated_content_image.png")

# ============================================================
# 10. Display Generated Text
# ============================================================

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(generated_text)

# ============================================================
# 11. Display Generated Image
# ============================================================

plt.figure(figsize=(8, 8))
plt.imshow(generated_image)
plt.axis("off")
plt.title("AI Generated Image")
plt.show()

print("\nImage saved as: generated_content_image.png")
print("=" * 60)
