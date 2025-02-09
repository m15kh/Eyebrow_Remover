import cv2
import numpy as np

# Input files
image_path = "/home/faryad/projects/eyebrow-project/img-readme/IMG_20250208_153309.jpg"
checkerboard_txt_path = "output_coordinates_and_homography.txt"
output_image_path = "unwrapped_cylindrical_checkerboard.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load the image from {image_path}")
    exit(1)

# Load the checkerboard points
try:
    with open(checkerboard_txt_path, "r") as f:
        lines = f.readlines()
        image_points = []
        reading_points = False
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if "Selected Points" in line:
                reading_points = True
                continue
            if "Homography Matrix" in line:
                break
            
            if reading_points:
                x, y = map(float, line.split())
                image_points.append([x, y])
        
        image_points = np.array(image_points, dtype=np.float32)

except Exception as e:
    print(f"Error reading checkerboard points: {e}")
    exit(1)

# Sort points from left to right and separate into two rows
sorted_indices = np.argsort(image_points[:, 0])
sorted_points = image_points[sorted_indices]

# Split into top and bottom rows based on y-coordinates
median_y = np.median(sorted_points[:, 1])
top_row = sorted_points[sorted_points[:, 1] < median_y]
bottom_row = sorted_points[sorted_points[:, 1] > median_y]

# Sort each row by x-coordinate
top_row = top_row[np.argsort(top_row[:, 0])]
bottom_row = bottom_row[np.argsort(bottom_row[:, 0])]

# Parameters for the output image
TARGET_WIDTH = 2400
TARGET_HEIGHT = 400
MARGIN = 50

# Create target points for a flat grid
target_points = []
spacing_x = (TARGET_WIDTH - 2 * MARGIN) / 11  # 12 points = 11 spaces
spacing_y = TARGET_HEIGHT - 2 * MARGIN

# Generate target points for top and bottom rows
for row in range(2):
    y = MARGIN + row * spacing_y
    for col in range(12):
        # Apply non-linear spacing for x coordinates to compensate for cylindrical distortion
        # Use an arc-length based mapping
        angle = (col / 11) * np.pi * 0.5  # Adjust the multiplier to control the amount of correction
        x = MARGIN + (TARGET_WIDTH - 2 * MARGIN) * (angle / (np.pi * 0.5))
        target_points.append([x, y])

target_points = np.array(target_points, dtype=np.float32)

# Combine the rows in the correct order
source_points = np.vstack((top_row, bottom_row))

# Calculate the transformation matrix
tps = cv2.createThinPlateSplineShapeTransformer()
source_points = source_points.reshape(1, -1, 2)
target_points = target_points.reshape(1, -1, 2)
matches = [cv2.DMatch(i, i, 0) for i in range(len(source_points[0]))]

tps.estimateTransformation(target_points, source_points, matches)

# Create output image
output = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

# Create a mesh grid of points to transform
rows, cols = TARGET_HEIGHT, TARGET_WIDTH
src_points = np.float32([(x, y) for y in range(rows) for x in range(cols)])
src_points = src_points.reshape(1, -1, 2)

# Transform all points
dst_points = tps.applyTransformation(src_points)[1]
dst_points = dst_points.reshape(-1, 2)

# Map each point from the destination to the source
for i, (x, y) in enumerate(dst_points):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        src_x = i % TARGET_WIDTH
        src_y = i // TARGET_WIDTH
        output[src_y, src_x] = image[int(y), int(x)]

# Fill any holes using interpolation
mask = np.all(output == 0, axis=2)
for y in range(1, TARGET_HEIGHT-1):
    for x in range(1, TARGET_WIDTH-1):
        if mask[y, x]:
            # Simple interpolation from neighboring pixels
            output[y, x] = np.mean(output[y-1:y+2, x-1:x+2][~mask[y-1:y+2, x-1:x+2]].reshape(-1, 3), axis=0)

# Save and display the result
cv2.imwrite(output_image_path, output)
print(f"Unwrapped cylindrical checkerboard saved to {output_image_path}")

# Display results
cv2.imshow("Original Image", cv2.resize(image, (1200, 800)))
cv2.imshow("Unwrapped Checkerboard", output)
cv2.waitKey(0)
cv2.destroyAllWindows()