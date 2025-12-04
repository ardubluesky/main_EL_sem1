import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
#  CSRNet Model
# --------------------------------------------------------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # Front-end (VGG16 until conv4_3)
        vgg = models.vgg16(pretrained=True)
        self.frontend = nn.Sequential(*list(vgg.features.children())[0:23])

        # Back-end dilated CNN
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        # Output 1-channel density map
        self.output_layer = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# --------------------------------------------------------------
#  Preprocessing + Inference
# --------------------------------------------------------------
def load_model(device):
    model = CSRNet().to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def estimate_crowd_percentage(model, img_path, device, max_capacity=200):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert for model
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        density_map = model(input_tensor)

    density_np = density_map.cpu().numpy().squeeze()

    # Crowd count = sum of density map
    count = float(density_np.sum())

    # Cap percentage to 100
    percentage = min(100.0, (count / max_capacity) * 100)

    return count, percentage, density_np


# --------------------------------------------------------------
#  Visualization Helper
# --------------------------------------------------------------
def show_density_map(density_map):
    plt.imshow(density_map, cmap='jet')
    plt.title("Crowd Density Map")
    plt.colorbar()
    plt.show()


# --------------------------------------------------------------
#  MAIN EXECUTION
# --------------------------------------------------------------
if __name__ == "__main__":
    # Select GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}\n")

    model = load_model(device)

    # ----------------------------------------
    #  CHANGE THIS TO YOUR INPUT IMAGE PATH
    # ----------------------------------------
    img_path = "..\train\images\0008_jpg.rf.936a6a47c944714f0d9d35b65168112b.jpg"

    # Adjust max_capacity for your scene
    max_capacity = 300

    count, percentage, density = estimate_crowd_percentage(
        model, img_path, device, max_capacity=max_capacity
    )

    print(f"Estimated Crowd Count: {count:.2f}")
    print(f"Estimated Density Percentage: {percentage:.2f}%")

    # Optional visualization
    show_density_map(density)
