import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# CSRNet Model for ShanghaiTech Part B
# --------------------------------------------------------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # Front-end (VGG16 until conv4_3)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[0:23])

        # Back-end dilated CNN (matches SHB pretrained weights)
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
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        # Output 1-channel density map
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# --------------------------------------------------------------
# Preprocessing for CSRNet
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------------------
# Load pretrained model
# --------------------------------------------------------------
def load_pretrained_model(weights_path, device):
    model = CSRNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Some pretrained checkpoints may store the state dict directly or under 'state_dict'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------------------
# Crowd Estimation Function
# --------------------------------------------------------------
def estimate_crowd(model, img_path, device):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(input_tensor)

    density_np = density_map.cpu().numpy().squeeze()
    count = float(density_np.sum())  # Sum of density map = estimated crowd count
    return count, density_np


# --------------------------------------------------------------
# Visualization
# --------------------------------------------------------------
def show_density_map(density_map):
    plt.imshow(density_map, cmap='jet')
    plt.title("Crowd Density Map")
    plt.colorbar()
    plt.show()


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Path to pretrained CSRNet ShanghaiTech Part B weights
    pretrained_weights = r"./partBmodel_best.pth"  # <-- CHANGE to your path

    model = load_pretrained_model(pretrained_weights, device)

    # Test image
    img_path = r"images\image_3.jpg"  # <-- CHANGE to your image path

    count, density_map = estimate_crowd(model, img_path, device)

    print(f"Estimated Crowd Count: {count:.2f}")

    show_density_map(density_map)
