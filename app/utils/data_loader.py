import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class DogBreedsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        breed = self.data.iloc[idx, 0]
        country = self.data.iloc[idx, 1]
        fur_color = self.data.iloc[idx, 2]
        height = self.data.iloc[idx, 3]
        eye_color = self.data.iloc[idx, 4]
        longevity = self.data.iloc[idx, 5]
        character_traits = self.data.iloc[idx, 6]
        health_problems = self.data.iloc[idx, 7]

        # You can process and encode the data as needed here

        sample = {
            "breed": breed,
            "country": country,
            "fur_color": fur_color,
            "height": height,
            "eye_color": eye_color,
            "longevity": longevity,
            "character_traits": character_traits,
            "health_problems": health_problems,
        }

        return sample

def load_class_names(dataset_path):
    # Load the class names from the dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
        return dataset.classes
