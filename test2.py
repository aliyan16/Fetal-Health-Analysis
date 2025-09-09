import json
import torch
import open_clip
from PIL import Image

# Define paths for model configuration and weights
PATH_FETALCLIP_CONFIG = "FetalCLIP_config.json"
PATH_FETALCLIP_WEIGHT = "FetalCLIP_weights.pt"

# Set device to GPU (CUDA) for faster computation
device = torch.device("cuda")

# Load and register model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

# Load the FetalCLIP model and preprocessing transforms as well as tokenizer
model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
tokenizer = open_clip.get_tokenizer("FetalCLIP")
model.eval()
model.to(device)

# List of input image file paths
images = ["Image5.jpeg", "Image4.jpeg"] 
# Preprocess images and stack them into a single tensor
images = torch.stack([preprocess_test(Image.open(img_path)) for img_path in images]).to(device)

# Define text prompts for classification
text_prompts = [
    "Ultrasound image focusing on the fetal abdominal area, highlighting structural development.",
    "Fetal ultrasound image focusing on the heart, highlighting detailed cardiac structures.",
] # Please refer to the text prompts in zero_shot_planes_db/
text_tokens = tokenizer(text_prompts).to(device) # Tokenize the text prompts

# Perform model inference
with torch.no_grad(), torch.cuda.amp.autocast():
    # Encode text and images into feature vectors
    text_features = model.encode_text(text_tokens)
    image_features = model.encode_image(images)

    # Normalize feature vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores (probabilities) between image and text features
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)