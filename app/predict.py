import torch
import torchvision.transforms as transforms
from PIL import Image
from app.ViT_model import ViT

class Predictor:
    def __init__(self, model_path="breast_cancer_vit.pth", img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT(img_size=img_size, num_classes=2)  # Binary classification
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
        return {"Benign": probabilities[0], "Malignant": probabilities[1]}
