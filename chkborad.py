import cv2
import numpy as np

# Input image path
image_path = "/home/faryad/projects/eyebrow-project/img-readme/IMG_20250208_153309.jpg"
output_txt_path = "output_coordinates_and_homography.txt"

# Variables to store points
selected_points = []

# Mouse callback function
def select_point(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Point selected: ({x}, {y})")

        # Draw the point on the resized image
        cv2.circle(temp_resized_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Checkerboard Corners", temp_resized_image)

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load the image from {image_path}")
    exit(1)

# Resize the image to fit the screen
screen_height, screen_width = 800, 1200  # Define maximum display size
h, w = image.shape[:2]
resize_ratio = min(screen_width / w, screen_height / h)
resized_image = cv2.resize(image, (int(w * resize_ratio), int(h * resize_ratio)))

# Create a copy for displaying updates
temp_resized_image = resized_image.copy()

# Show the resized image and set the mouse callback
cv2.imshow("Select Checkerboard Corners", temp_resized_image)
cv2.setMouseCallback("Select Checkerboard Corners", select_point)

print("Select points on the checkerboard. Press 's' to save and calculate the homography matrix, or 'q' to save and quit.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('q'):
        # Scale points back to the original image size
        original_points = [(int(x / resize_ratio), int(y / resize_ratio)) for x, y in selected_points]

        # Save points to a text file
        with open(output_txt_path, "w") as f:
            f.write("Selected Points (x, y):\n")
            for point in original_points:
                f.write(f"{point[0]}, {point[1]}\n")

        print(f"Points saved to {output_txt_path}")

        # If 's' is pressed, calculate the homography matrix
        if key == ord('s') and len(original_points) >= 4:
            # Define destination points for the rectangle
            width = 500
            height = 500
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # Convert selected points to NumPy array (use the first 4 points)
            src_points = np.array(original_points[:4], dtype=np.float32)

            # Calculate the homography matrix
            homography_matrix, _ = cv2.findHomography(src_points, dst_points)
            print("Homography Matrix:")
            print(homography_matrix)

            # Append homography matrix to the text file
            with open(output_txt_path, "a") as f:
                f.write("\nHomography Matrix:\n")
                for row in homography_matrix:
                    f.write(" ".join(map(str, row)) + "\n")

            print(f"Homography matrix saved to {output_txt_path}")

        break

cv2.destroyAllWindows()
