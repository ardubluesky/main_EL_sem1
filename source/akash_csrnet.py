import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------
#  CSRNet Model
# --------------------------------------------------------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features.children())[0:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return torch.relu(x)  # ensure non-negative density

# --------------------------------------------------------------
#  Label Loader (YOLO -> Points -> Density Map)
# --------------------------------------------------------------
def load_yolo_labels(label_path, image_shape):
    h, w = image_shape[:2]
    points = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, _, _ = map(float, parts)
                x_pixel = int(x_center * w)
                y_pixel = int(y_center * h)
                points.append((x_pixel, y_pixel))
    return points

def generate_density_map(image_shape, points, sigma=4):
    h, w = image_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1
    return gaussian_filter(density, sigma=sigma)

# --------------------------------------------------------------
#  Training Setup
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train_single_image(img_path, label_path, device, epochs=50, lr=1e-5):
    # Load image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load labels -> density map
    points = load_yolo_labels(label_path, img.shape)
    density_gt = generate_density_map(img.shape, points, sigma=4)

    # Convert to tensors
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    density_tensor = torch.tensor(density_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Model + optimizer
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor)

        # Resize ground truth to match output size
        out_h, out_w = output.shape[2], output.shape[3]
        density_resized = F.interpolate(
            density_tensor, size=(out_h, out_w), mode='bilinear', align_corners=False
        )

        loss = criterion(output, density_resized)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, density_gt, output.detach().cpu().numpy().squeeze()

# --------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    img_path = Path("C:/Users/akash/Desktop/main EL/main_EL_sem1/source/test/images/0002_jpg.rf.8d3cbff2fb01586a011856692afa0017.jpg")
    label_path = Path("C:/Users/akash/Desktop/main EL/main_EL_sem1/source/test/labels/0002_jpg.rf.8d3cbff2fb01586a011856692afa0017.txt")

    # Train
    model, density_gt, density_pred = train_single_image(img_path, label_path, device, epochs=50)

    # Visualize GT density
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(density_gt, cmap='jet')
    plt.title("Ground Truth Density Map")
    plt.colorbar()

    # Visualize predicted density
    plt.subplot(1,2,2)
    plt.imshow(density_pred, cmap='jet')
    plt.title("Predicted Density Map")
    plt.colorbar()
    plt.show()