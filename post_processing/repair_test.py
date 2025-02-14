import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
from scipy.spatial import distance
import matplotlib.pyplot as plt

def skeleton_extraction(image_path):
    """Extracts the skeleton of a binary grain boundary image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    
    if np.mean(img) > 0.5:
        binary_img = (img < 0.5)
    else:
        binary_img = (img > 0.5)
        
    skeleton = skeletonize(binary_img)  # Convert to binary and skeletonize
    return (skeleton * 255).astype(np.uint16)

# def detect_breakpoints(skeleton):
#     """Detects breakpoints in the skeleton image."""
#     breakpoints = []
#     kernel = np.ones((3, 3), np.uint16)
#     neighbors = cv2.filter2D(skeleton, -1, kernel)
    
#     for y in range(1, skeleton.shape[0] - 1):
#         for x in range(1, skeleton.shape[1] - 1):
#             if skeleton[y, x] == 1 and neighbors[y, x] in [2, 3]:
#                 breakpoints.append((x, y))
                
#     return breakpoints

def detect_breakpoints(skeleton):
    """Detects breakpoints in the skeleton image."""
    breakpoints = []
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint16)

    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    
    endpoints = (neighbor_count/255 == 11) & (skeleton > 0)
    
    # plt.imshow(endpoints, cmap='gray')
    # plt.axis('off')
    # plt.show()
    
    print(endpoints.shape)
    
    y_coords, x_coords = np.nonzero(endpoints)
    breakpoints = list(zip(x_coords, y_coords))
                
    return breakpoints, endpoints

# def connect_breakpoints(image, breakpoints, max_distance=50, angle_threshold=120):
#     """Connects breakpoints within a defined distance and angle threshold."""
#     for i, p1 in enumerate(breakpoints):
#         for j, p2 in enumerate(breakpoints):
#             if i >= j:
#                 continue
            
#             dist = distance.euclidean(p1, p2)
#             if dist < max_distance:
#                 angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
#                 if abs(angle) < angle_threshold or abs(angle) > (180 - angle_threshold):
#                     cv2.line(image, p1, p2, 255, 1)
                    
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()
    
def connect_breakpoints(image, breakpoints, endpoints, max_distance=30, angle_threshold=120):
    points_set = set(tuple(p) for p in breakpoints)
    og = image.copy()

    for p1_tuple in list(points_set): # Iterate over a copy to avoid changing set size during iteration
        p1 = np.array(p1_tuple) # Convert back to numpy array for distance calculation
        if p1_tuple not in points_set: # if the point has already been connected skip it
            continue

        best_p2 = None
        min_dist = float('inf')

        for p2_tuple in set(points_set): # Iterate over a copy to avoid changing set size during iteration
            p2 = np.array(p2_tuple)
            if tuple(p1) == tuple(p2): # avoid comparing same points
                continue
            dist = distance.euclidean(p1, p2)
            if dist < max_distance:
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
                if abs(angle) < angle_threshold or abs(angle) > (180 - angle_threshold):
                    if dist < min_dist:
                        min_dist = dist
                        best_p2 = p2_tuple

        if best_p2:
            cv2.line(image, tuple(p1), best_p2, 255, 1)
            points_set.remove(p1_tuple) # Remove p1 as well now that it is connected
            points_set.remove(best_p2)
    

    # Final Result
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(og, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(endpoints, cmap='gray')
    axes[1].set_title("Endpoints")
    axes[1].axis('off')

    axes[2].imshow(image, cmap='gray')
    axes[2].set_title("Connected")
    axes[2].axis('off')
    
    return image

def detect_fork_points(skeleton):
    """Identifies fork points where three or more branches meet."""
    fork_points = []
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1 and neighbors[y, x] >= 4:
                fork_points.append((x, y))
    return fork_points

def extend_missing_boundaries(image, breakpoints, search_distance=50):
    """Extends grain boundaries from breakpoints if no connection is found."""
    for p in breakpoints:
        for d in range(1, search_distance):
            x_new, y_new = int(p[0] + d * np.cos(0)), int(p[1] + d * np.sin(0))
            if 0 <= x_new < image.shape[1] and 0 <= y_new < image.shape[0]:
                if image[y_new, x_new] == 1:
                    cv2.line(image, p, (x_new, y_new), 1, 1)
                    break
    return image

def pseudo_label_repair(image):
    """Main function to repair pseudo-labels in grain boundary images."""
    skeleton = skeleton_extraction(image)
    print("Skeletonized")
    breakpoints, endpoints = detect_breakpoints(skeleton)
    print("Breakpoints found")
    image = connect_breakpoints(skeleton, breakpoints, endpoints)
    print("Connected")
    # fork_points = detect_fork_points(skeleton)
    image = extend_missing_boundaries(image, breakpoints)
    print("extended")
    return image


image_path = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\preds\\0_1.png'
repaired_image = pseudo_label_repair(image_path)

plt.imshow(repaired_image, cmap='gray')
plt.title("Result")
plt.axis('off')
plt.show()


# cv2.imwrite('repaired_boundary.png', repaired_image * 255)
