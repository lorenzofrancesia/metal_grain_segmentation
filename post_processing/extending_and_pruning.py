import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
from scipy.spatial import distance


def skeletonize_image(image_path):
    """
    Reads, binarizes, and skeletonizes an image, displaying each step.
    """
    # 1. Read Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.title("1. Original Grayscale Image")
    # plt.axis('off')
    # plt.show()

    # 2. Normalize
    img = img / 255.0
    # plt.imshow(img, cmap='gray')
    # plt.title("2. Normalized Image (0-1)")
    # plt.axis('off')
    # plt.show()
    # print(np.unique(img))

    # 3. Binarize
    if np.mean(img) > 0.5:
        binary_img = (img < 0.5)
    else:
        binary_img = (img > 0.5)
    # plt.imshow(binary_img, cmap='gray')
    # plt.title("3. Binary Image (Thresholded)")
    # plt.axis('off')
    # plt.show()
    # print(np.unique(binary_img))

    # 4. Skeletonize
    skeleton = skeletonize(binary_img)
    # plt.imshow(skeleton, cmap='gray')
    # plt.title("4. Skeletonized Image")
    # plt.axis('off')
    # plt.show()

    return (skeleton * 255).astype(np.uint16)

def get_endpoints_and_branch_points(skeleton):
    """
    Detects and visualizes endpoints and branch points.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint16)

    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    
    # print(np.unique(neighbor_count))
    # unique_values = np.unique(neighbor_count)
    # cmap = {0: (0, 0, 0), 255: (1, 1, 1)}
    # other_colors = plt.colormaps['tab10'].colors
    # color_index = 0
    # for value in unique_values:
    #     if value not in cmap:
    #         cmap[value] = other_colors[color_index % len(other_colors)]
    #         color_index += 1
    # rgb_image = np.zeros((neighbor_count.shape[0], neighbor_count.shape[1], 3), dtype=float)
    # for value, color in cmap.items():
    #     rgb_image[neighbor_count == value] = color
    # fig, ax = plt.subplots()
    # ax.imshow(rgb_image)
    # legend_elements = [plt.Line2D([0], [0], marker='o', color=cmap[value], label=f"{value/255}", markersize=8) for value in sorted(cmap.keys()) if value != 0 and value != 255]
    # ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title("Neighbor Counts")
    # plt.axis('off')
    # plt.show()

    endpoints = (neighbor_count/255 == 11) & (skeleton > 0)
    branch_points = (neighbor_count/255 >= 13) & (skeleton > 0)

    # Visualize Endpoints
    # plt.imshow(endpoints, cmap='gray')
    # plt.title("Endpoints")
    # plt.axis('off')
    # plt.show()
    
    # # Visualize Branch Points
    # plt.imshow(branch_points, cmap='gray')
    # plt.title("Branch Points")
    # plt.axis('off')
    # plt.show()

    # Visualize Endpoints and Branchpoints OVER the skeleton
    # display_image = np.zeros((*skeleton.shape, 3), dtype=np.uint8) # Create a 3 channel all black image.
    # display_image[skeleton > 0, :] = [255, 255, 255]  # Make skeleton white
    # display_image[endpoints, :] = [255, 0, 0]       # Make endpoints red
    # display_image[branch_points, :] = [0, 255, 0]    # Make branch points green
    # plt.imshow(display_image)
    # plt.title("Skeleton with Endpoints (Red) and Branch Points (Green)")
    # plt.axis('off')
    # plt.show()

    return endpoints, branch_points

def prune_branches(skeleton, branch_length_ratio=0.05):
    """
    Iteratively prunes
    """
    pruned_skeleton = skeleton.copy().astype(np.uint16)
    iteration = 0

    while True:
        iteration += 1
        endpoints, _ = get_endpoints_and_branch_points(pruned_skeleton)
        num_endpoints = np.sum(endpoints)

        if num_endpoints == 0:
            print("No endpoints found. Breaking loop.")
            break
    
        # Label the CURRENT skeleton, BEFORE any removal
        labeled_skeleton = label(pruned_skeleton, connectivity=2)  #Initial labeling
        regions = regionprops(labeled_skeleton)
        removed_any = False

        # print(f"Iteration: {iteration}")
        # print(f"Number of endpoints: {num_endpoints}")
        # print(f"Number of regions: {len(regions)}")
        
        total_branch_length = 0
        
        for i, region in enumerate(regions):
            if region.major_axis_length < branch_length_ratio:
                # print(f"    Region {i+1}: Major Axis Length {region.major_axis_length:.2f}, Marked for removal")
                for coord in region.coords:
                    pruned_skeleton[coord[0], coord[1]] = 0
                removed_any = True
            # else:
                # print(f"    Region {i+1}: Major Axis Length {region.major_axis_length:.2f}")
        
        # print(f"Any removed: {removed_any}")

        if not removed_any:
            print("No branches removed in this iteration. Breaking loop.")
            break

        pruned_skeleton = (pruned_skeleton > 0).astype(np.uint16)

        # print(f"Pixels remaining (simplified): {np.sum(pruned_skeleton > 0)}")

        # plt.imshow(pruned_skeleton, cmap='gray')
        # plt.title(f"Pruned Skeleton (Iteration {iteration})")
        # plt.axis('off')
        # plt.show()

    return (pruned_skeleton * 255).astype(np.uint16)

def detect_breakpoints(skeleton):
    """Detects breakpoints in the skeleton image."""
    breakpoints = []
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint16)

    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    
    endpoints = (neighbor_count/255 == 11) & (skeleton > 0)
    endpoints = endpoints.astype(np.uint8)
    
    # plt.imshow(endpoints, cmap='gray')
    # plt.axis('off')
    # plt.show()
    
    y_coords, x_coords = np.nonzero(endpoints)
    breakpoints = list(zip(x_coords, y_coords))
                
    return breakpoints, endpoints

# def connect_breakpoints(image, breakpoints, max_distance=50, angle_threshold=60):  
#     try: 
#         connected_points = set()
#         points_array = np.array(breakpoints)
#         for i in range(len(points_array)):
#             p1 = points_array[i]
#             if i  in connected_points:
#                 continue
#             best_match = None
#             min_dist = float('inf')
#             for j in range(len(points_array)):
#                 p2 = points_array[j]
#                 if i == j or j in connected_points:
#                     continue
#                 dist = distance.euclidean(p1, p2)
#                 if dist > max_distance:
#                     continue
#                 angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
#                 if abs(angle) < angle_threshold or abs(angle) > (180 - angle_threshold):
#                     if dist < min_dist:
#                         min_dist = dist
#                         best_match = j
#             if best_match is not None:
#                 cv2.line(image, tuple(p1), tuple(points_array[best_match]), 255, 1)
#                 connected_points.add(i)
#                 connected_points.add(best_match)
#         return image
#     except Exception as e:
#         print(e)

def connect_breakpoints(image, breakpoints, max_distance=40, angle_threshold=120):
    points_set = set(tuple(p) for p in breakpoints)
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
    return image

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

def process_image(image_path, min_branch_length=20):
    """
    Main function to orchestrate the process.
    """
    skeleton = skeletonize_image(image_path) 
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    plt.show()   
    pruned_skeleton = prune_branches(skeleton, min_branch_length)
    
    plt.imshow(pruned_skeleton, cmap='gray')
    plt.axis('off')
    plt.show()
    breakpoints, endpoints = detect_breakpoints(pruned_skeleton) 
    
    plt.imshow(endpoints, cmap='gray')
    plt.axis('off')
    plt.show()
    
    connected_skeleton = connect_breakpoints(pruned_skeleton, breakpoints)
    
    
    plt.imshow(connected_skeleton, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # extended_skeleton = extend_missing_boundaries(connected_skeleton, breakpoints)

    # # Final Result
    # image = cv2.imread(image_path)
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(image)
    # axes[0].set_title("Original Image")
    # axes[0].axis('off')

    # axes[1].imshow(pruned_skeleton, cmap='gray')
    # axes[1].set_title("Pruned Skeletonized Image")
    # axes[1].axis('off')

    # axes[2].imshow(extended_skeleton, cmap='gray')
    # axes[2].set_title("Extended Skeleton Image")
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()

    # return image


# # # Example usage
image_path = 'C:\\Users\\lorenzo.francesia\\OneDrive - Swerim\\Desktop\\preds\\0_2.png'
pruned_skeleton = process_image(image_path, min_branch_length=30)