# What are Vision-Language Models?
# Vision-Language Models (VLMs) are AI models that can process both images and text together to perform tasks like: 
# Image Captioning which is generating descriptions for iamges.
# Visual Question Answering (VQA) which is answering questions about images.
# Image-Text Retrieval which is finding images based on text queries and vice versa.
# Multimodal Reasoning which is understanding relationships between visual and textual content.

# Examles of VLMS:
# 1. CLIP (OpenAI) which matches images with text descriptions.
# 2. BLIP (Salesforce) used for image captioning and VQA.
# 3. LLaVA (Large Language and Vision Assistant) which is GPT-4-like model for images.


# Step 1: Install Pre-Trained Vision-Language Models

# For first hands-on experiment, install Hugging Face Transformers library which provides pre-trained VLMs and Pillow library for image processing.

# pip install transformers torch torchvision pillow
# we will use BLIP(Bootstrapped Language-Image Pre-training) model for image captioning.


# Step 2: Image Captioning with BLIP
# build a simple program that takes an image and generates a caption.
# an image was saved in the directory

# Step 3: Write the code

# import libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BlIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load the image
image_path = "image1.jpg" 
image = Image.open(image_path).convert("RGB")

# Preprocess the image and generate a caption 
#inputs = processor(images=image, return_tensors="pt") # preprocess the image using the processor and convert it to PyTorch tensors. PyTorch tensors are used as input to the model. PyTorch is a popular deep learning library.
# with torch.no_grad(): # here we are using torch.no_grad() to disable gradient calculation which is not needed for inference. gradient calculation is used for training. 
    #caption = model.generate(**inputs) # generate the caption using the model and inputs generated from the image.
    
# Decode and print the caption
#generated_caption = processor.decode(caption[0], skip_special_tokens=True)
#print("Generated Caption:", generated_caption)

# Step 4: Run the code
# run the code and it will generate a caption for the image.

# Step 5: Understand the code
# The code uses the BlipProcessor and BlipForConditionalGeneration classes from the transformers library to load the BLIP model for image captioning.
# It then loads an image using the PIL library and preprocesses it using the processor.
# The model generates a caption for the image, which is then decoded and printed.

# Step 6: Experiment with the code
# modfiy the model parameters to optimize the caption generation.

input = processor(images=image, return_tensors="pt")
with torch.no_grad():
    caption = model.generate(
        **input,
        max_length=30, # set the maximum length of the generated caption to 30 tokens.
        num_return_sequences=3, # generate multiple captions for the same image.
        temperature=0.7, # adjust the temperature parameter to control the randomness of the generated captions. here we set it to 0.7 which is a common value and makes it more creative.
        top_k=50, # adjust the top_k parameter to control the diversity of the generated captions. here we set it to 50 to reduce unlikely words.
        top_p=0.9, # adjust the top_p parameter to control the diversity of the generated captions. here we set it to 0.9 to focus on high probability words.
        repetition_penalty=1.5, # Avoid repeating phrases in the generated captions by setting the repetition_penalty parameter to 1.5.
        do_sample=True # Enable sampling-based generation to support multiple return sequences.
    )
    
for i, cap in enumerate(caption):  # loop through the generated captions, i is the index and cap is the caption.
    generated_caption = processor.decode(cap, skip_special_tokens=True) # decode and print the generated caption.
    print(f"Generated Caption {i+1}: {generated_caption}")
    
    
# output: 
# Generated Caption 1: beautiful tiger lying down on the ground with his eyes closed
# Generated Caption 2: leopard laying on dirt in front of grass and bushes
# Generated Caption 3: wildlife park in india has a number of exotic animals, including this tiger
