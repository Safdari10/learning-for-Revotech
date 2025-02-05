# What is LLaVA?
# LLaVA is a true multimodel AI model that combines:
# a) A vision encoder (CLIP like image model)
# b) A large languaged model (LLaMA / Vicuna, Mistral) LLaMA, Vicuna, Mistral are large language models that can handle both text and images and are other examples of multimodal AI models.

# this one model can handle both captioning and VQA. 

# Seting up LLaVA for multimodal AI
# !1. Install the necessary libraries
# below is the code to install torch, sentencepiece, and accelerate which are the necessary libraries for LLaVA
# pip install torch sentencepiece accelerate 
# below is the code to install the Hugging Face Transformers library which is the latest version of the Transformers library but not the official one
# pip install git+https://github.com/huggingface/transformers.git
# below is the code to install the LLaVA library
# pip install git+https://github.com/haotian-liu/LLaVA.git


#! 2. Load the LLaVA model

import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

model_name = "liuhaotian/llava-v1.6-mistral-7b"

# Load the processor and model with trust_remote_code enabled
processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True) # trust_remote_code=True means that we trust the model code to not execute arbitrary code
model = LlavaForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


#! 3. Generate Image Caption

def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    
    prompt = "Describe the image"
    
    # Encode inputs
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    # Generate the caption
    with torch.no_grad():
        caption = model.generate_image_caption(**inputs, max_length=50)

    return processor.decode_batch(caption[0], skip_special_tokens=True)


# testing captioning
image_path = "image.jpg"
caption = generate_caption(image_path)
print("Caption:", caption)



#! 4. Generate Visual Question Answering

def answer_vqa(image_path, question):
    # Load and preprocess the image
    image = Image.open(image_path)
    
    # Encode inputs
    inputs = processor(image, question, return_tensors="pt").to(model.device)

    # Generate the answer
    with torch.no_grad():
        answer = model.generate_vqa(**inputs, max_length=50)

    return processor.decode_batch(answer[0], skip_special_tokens=True)

# test VQA
image_path = "image.jpg"
question = "What color is the car?"
answer = answer_vqa(image_path, question)
print("Answer:", answer)

#!!! unable to execute the code as getting missing file error preprocessor_config.json not found for LLaVA model. 
#! have submitted the issue in the repository.