from torch.utils.data import Dataset
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None # NB: for TIF images

class PretrainingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.image_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ][:1] # TODO: remove after verification of local training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"img_path {img_path}")
        image = Image.open(img_path).convert('RGB')  # TODO: is "RGB" correct 3 color channels for feature?
        if self.transform:
            image = self.transform(image)
        return image
