# Hands-On: Using CLIP (A Foundation Model for Image Understanding)
# use CLIP to classify images based on text descriptions.

# Step 1: Install the required libraries
# !pip install torch torchvision transformers pillow clip

# Step 2: Load the CLIP model

import torch
import clip
from PIL import Image

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
model, preprocess = clip.load("ViT-B/32", device=device) # Load CLIP model

# Load the image
image_path = "image.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device) # Load image and preprocess it for CLIP

# Define text labels
text_labels = ["a tiger", "a car", "a person", "a tree" ] # Define text labels
text_inputs = clip.tokenize(text_labels).to(device) # Tokenize text labels for CLIP. tokenize() is a utility function provided by CLIP to convert text to token IDs. 
# token ids are used as input to the model. tokens are the basic units of text, like words or subwords.

# Get CLIP predictions
with torch.no_grad():
    image_features = model.encode_image(image) # Encode image features
    text_features = model.encode_text(text_inputs) # Encode text features
 
    # Compute similarity between image and text features
    similarity = (image_features @ text_features.T).softmax(dim=-1) # Compute similarity between image and text features using dot product and softmax
    best_match = text_labels[similarity.argmax().item()] # Get the best matching text label
    
print(f"Predicted Label: {best_match}") # Print the predicted label based on the image