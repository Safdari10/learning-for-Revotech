# What is Visual Question Answering (VQA)?
# VQA models can answer questoions about an image. Instead of just generating captions, they allow you to ask questions and get relevant answers.
# Example Use Cases: 
# "What is the color of the car?" "Red"
# "How many people are in the image?" "Three"
# "What is the person doing?" "Playing tennis"

# Step 1: Install the Required Libraries
# Use the BLIP model for VQA, which is a multimodal model that can process both images and text.
# pip install transformers torch torchvision pillow

# Step 2: Load the BLIP Model for VQA

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Load the BLIP processor and model for VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# load the image
image_path = "image1.jpg"
image = Image.open(image_path).convert("RGB")

# Define a question
question = "What is in the image?"

# Preprocess the image and question
inputs = processor(image, question, return_tensors="pt")

# Generate an answer
with torch.no_grad():
    answer = model.generate(**inputs)
    
    
# Decode and print the answer
generated_answer = processor.decode(answer[0], skip_special_tokens=True)
print("Generated Answer:", generated_answer)

