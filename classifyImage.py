import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tkinter import filedialog
from model import SimpleCNN

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 2, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128) 
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path, device="cpu"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def classify_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_val(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

if __name__ == "__main__":
    model_path = "../models/image_classifier.pth"
    class_names = ["class_1", "class_2"]

    model = load_model(model_path)

    image_path = filedialog.askopenfilename(title="Select an image")
    if image_path:
        prediction = classify_image(image_path, model, class_names)
        print(f"Prediction: {prediction}")