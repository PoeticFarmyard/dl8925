# week5

import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Load pretrained GoogleNet model

model = models.googlenet(pretrained=True)
model.eval()

# Define preprocessing transforms

preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406], # ImageNet mean
    std=[0.229, 0.224, 0.225] # ImageNet std
  )
])

# Download an example image of a Labrador

url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# Preprocess the image and add batch dimension

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0) # Shape: [1, 3, 224, 224]

# Run inference without computing gradients

with torch.no_grad():
  output = model(input_batch)

# Load ImageNet class labels

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# Compute probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("Top 5 Predictions:")

for i in range(top5_prob.size(0)):
  print(f"{labels[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
