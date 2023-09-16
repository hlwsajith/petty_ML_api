import torch
from torchvision import transforms
from PIL import Image

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),         # Resize the input image to 256x256 pixels
    transforms.CenterCrop(224),     # Crop the center 224x224 pixels
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor (0-1 range)
    transforms.Normalize(           # Normalize with precomputed mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply preprocessing
    return preprocess(image).unsqueeze(0)  # Add a batch dimension
