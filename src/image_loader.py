import os

import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)  # Add batch dimension
            images.append(image)
    return images
