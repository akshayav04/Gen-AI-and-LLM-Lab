import torch
import matplotlib.pyplot as plt

from PIL import Image
from google.colab import files

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)

# -------------------------------------
# Device
# -------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# -------------------------------------
# Load BLIP Model
# -------------------------------------

model_name = "Salesforce/blip-vqa-base"

processor = BlipProcessor.from_pretrained(model_name)

model = BlipForQuestionAnswering.from_pretrained(
    model_name
).to(device)

# -------------------------------------
# Upload Image
# -------------------------------------

print("\nUpload an image for analysis:")

uploaded_files = files.upload()

image_path = next(iter(uploaded_files))

image = Image.open(image_path).convert("RGB")

# -------------------------------------
# Ask Question
# -------------------------------------

question = input(
    "\nEnter a question about the uploaded image: "
)

# -------------------------------------
# Process Inputs
# -------------------------------------

inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# -------------------------------------
# Generate Answer
# -------------------------------------

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30
    )

answer = processor.decode(
    generated_ids[0],
    skip_special_tokens=True
)

# -------------------------------------
# Display Image
# -------------------------------------

plt.figure(figsize=(8,6))
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# -------------------------------------
# Result
# -------------------------------------

print("\nMULTIMODAL AI RESULT")
print("-" * 50)
print("Question:", question)
print("Answer:", answer)
