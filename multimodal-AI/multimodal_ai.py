# What is multimodal AI?
# Multimodal AI can process and understand multiple types of data at the same time such as:
# - Text(e.g. ChatGPT, Gemini AI)
# - Images(e.g. CLIP, DALL-E)
# - Audio(e.g. Whisper, Speach AI, Deep Voice)
# - Video(e.g. NVIDIA DeepStream, Flamingo)

# example:
# Gemini AI, GPT-4V can understand images and respond in text.
# Flamingo can answer questions about videos.
# DeepSteam can detect objects in real-time video streams.

# How does multimodal AI work?
# 1. Feature Extraction: AI converts inputs (text, image, video, etc.) into a numerical format.
# 2. Fusion: AI models combine different data types to make predictions.
# 3. Output Generation: AI model produces text, images, captions, etc.

# Examples of use cases:
# self-driving cars(processing video + sensor data)
# AI assistants(voice + text + images)
# AI for healthcare(X-ray images + patient records)
# AI for social media(text + images + videos)


# Hands-on: Image + Text Model (BLIP2)
# BLIP2 is a multimodal AI model that can generates image captions and answers questions about images.
# BLIP2 uses a transformer-based architecture to process both images and text.
# transformers are a type of deep learning model that can process sequences of data like text, images, and audio.
# deep learning models are trained on large datasets to learn patterns and make predictions.

# step 1: install dependencies
# pip install torch torchvision transformers pillow accelerate protobuf
# protobuff is a data serialization format used by the BLIP2 model.
# accelerate is a library that helps speed up deep learning training on GPUs.

# step 2: import libraries
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import torch

# step 3: load the BLIP2 model
# ! BLIP2 is a large model, and we are having tokenization issues with the model, otherwise BLIP2 is a good model for image captioning and question answering.
# ! other examples of multimodal AI models are CLIP, DALL-E, and VQ-VAE-2 which can handle different types of data like text and images.
# Processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# so we are using a smaller model blip1 instead of blip2
# also since blip1 is only good for image captioning, we also have to use the vqa version of the model to answer questions about images

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


# Step 4: Function to generate image captions
def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)

# Function to generate an answer
def answer_vqa(image_path, question):
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    inputs = vqa_processor(image, question, return_tensors="pt")

    # Generate answer
    with torch.no_grad():
        output = vqa_model.generate(**inputs, max_length=20)

    # Decode and return the result
    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer.strip() # Remove leading/trailing spaces


# Step 6: test the model
image_path = "image.jpg"
caption = generate_caption(image_path)
print("Image Caption:", caption)

question = "What is in the image?"
answer = answer_vqa(image_path, question)
print("Answer:", answer)


# ! While we didnt actaully use a multimodal AI model in this case because BLIP2 was having tokenization issues, we learnt that multimodal AI can process and understand multiple types of data at the same time such as text, images, audio, and video.

# Why is multimodal AI important?
# 1. Multimodal AI is important because it gives more human like understanding and AI can process multiple types of data at the same time like humans do.
# 2. Better Decision-making: AI will be useful in making better decisions by combining different data types in real-time for example in self-driving cars, healthcare, and robotics.
# 3. More Engaging AI Applications: Enables intelligent chatbots that understand text + images.

