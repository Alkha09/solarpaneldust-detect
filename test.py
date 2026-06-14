import torch
import cv2
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import datetime
import torch.nn.functional as F

# -------------------------------
# Load Model (MobileNetV2)
# -------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# -------------------------------
# Transform (must match training)
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Load Image
# -------------------------------
img = cv2.imread("test_image.png")
if img is None:
    print("❌ Image not found! Make sure test_image.png is in the same folder.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(img_rgb).unsqueeze(0)

# -------------------------------
# Prediction
# -------------------------------
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

confidence = confidence.item() * 100
predicted_class = predicted.item()
classes = ["Clean", "Dusty"]
result = classes[predicted_class]

# -------------------------------
# Efficiency Estimation
# -------------------------------
if result == "Clean":
    efficiency = 95 + (confidence / 100) * 5   # 95–100%
else:
    efficiency = 60 - (confidence / 100) * 20  # 40–60%

efficiency = round(efficiency, 2)

# -------------------------------
# Maintenance Decision
# -------------------------------
if result == "Dusty" and confidence > 70:
    maintenance = "IMMEDIATE CLEANING REQUIRED"
elif result == "Dusty":
    maintenance = "Cleaning Recommended Soon"
else:
    maintenance = "No Maintenance Required"

# -------------------------------
# Next Cleaning Schedule
# -------------------------------
today = datetime.date.today()
if result == "Dusty":
    next_cleaning = today + datetime.timedelta(days=2)
else:
    next_cleaning = today + datetime.timedelta(days=15)

# -------------------------------
# Display on Image
# -------------------------------
cv2.putText(img, f"Prediction: {result}",          (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
cv2.putText(img, f"Confidence: {confidence:.2f}%", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),   2)
cv2.putText(img, f"Efficiency: {efficiency}%",     (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),   2)
cv2.putText(img, f"Maintenance: {maintenance}",    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(img, f"Next Cleaning: {next_cleaning}",(10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow("Solar Panel Dust Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------
# Print Results in Console
# -------------------------------
print("=" * 40)
print(f"  Prediction  : {result}")
print(f"  Confidence  : {confidence:.2f}%")
print(f"  Efficiency  : {efficiency}%")
print(f"  Maintenance : {maintenance}")
print(f"  Next Cleaning: {next_cleaning}")
print("=" * 40)