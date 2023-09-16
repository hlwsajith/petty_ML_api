import torch
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from io import BytesIO
from starlette.responses import JSONResponse
from app.models.dog_breed_model import DogBreedClassifier
from app.utils.image_preprocessing import preprocess_image
from app.utils.data_loader import load_class_names
import pickle

predict_router = APIRouter()

model = DogBreedClassifier(num_classes=len(load_class_names("data/dog_breeds_dataset.pkl")))
model.eval()

class_names = load_class_names("data/dog_breeds_dataset.pkl")
def load_class_names(dataset_path):
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
            class_names = dataset['Breed'].tolist()
            return class_names
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {dataset_path}")


@predict_router.post("/predict/")
async def predict_dog_breed(file: UploadFile):
    try:
        image = Image.open(BytesIO(await file.read()))
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = torch.argmax(output).item()
        predicted_breed = load_class_names("data/dog_breeds_dataset.pkl")[predicted_class]

        return JSONResponse(content={"predicted_breed": predicted_breed}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
