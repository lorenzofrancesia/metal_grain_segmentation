import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation


img_path = "C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\gts\\0_0.png"
image = Image.open(img_path).convert('L')

transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
    ])
image = transform(image)

gt_array = image.numpy().astype(np.float32)
if np.mean(gt_array) > 0.5:
    binary_img = (gt_array < 0.5)
else:
    binary_img = (gt_array > 0.5)

skeleton = skeletonize(binary_img)

dilated_skeleton = dilation(skeleton)

ground_truth = torch.from_numpy(dilated_skeleton.astype(np.float32)).unsqueeze(0) 
# print(f"groundtruth has shape {ground_truth.shape}")

input_skeleton = skeleton.copy()
input_skeleton = input_skeleton[0,:,:]

row_indices, col_indices = np.where(input_skeleton == True)
coordinates = list(zip(row_indices, col_indices))

num_erase_points = np.random.randint(9, 16)
erosion_radius = np.random.randint(5, 10) 
selected_points = np.random.choice(len(coordinates), min(num_erase_points, len(coordinates)), replace=False)

for i in selected_points:
    row, col = coordinates[i]
    rr, cc = np.ogrid[:input_skeleton.shape[0], :input_skeleton.shape[1]]
    mask = (rr - row)**2 + (cc - col)**2 <= erosion_radius**2
    input_skeleton[mask] = 0

dilated_input = dilation(input_skeleton)
input_image = torch.from_numpy(dilated_input.astype(np.float32)).unsqueeze(0).unsqueeze(0)

