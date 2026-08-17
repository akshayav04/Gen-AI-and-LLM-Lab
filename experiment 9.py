import torch
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipe.to(device)

prompt = """
A futuristic smart city with flying cars,
green buildings,
robots assisting people,
highly detailed,
realistic,
4K quality.
"""

image = pipe(prompt).images[0]

image.save("generated_image.png")

plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

print("Image successfully generated and saved as generated_image.png")
