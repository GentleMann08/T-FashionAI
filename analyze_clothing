import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Загружаем предобученную модель ResNet для распознавания одежды
model = models.resnet50(pretrained=True)
model.eval()

# Преобразования для входного изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def analyze_clothing(image_path):
    # Загрузка и предобработка изображения
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Добавляем размер батча

    # Прогоняем через модель и получаем предсказание
    with torch.no_grad():
        outputs = model(image_tensor)

    # Предполагаем, что у нас есть словарь классов одежды
    _, predicted = outputs.max(1)
    clothing_category = predicted.item()

    return clothing_category
