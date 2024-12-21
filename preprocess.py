import numpy as np
import cv2




def apply_canny_edge_detection(img, low_threshold=50, high_threshold=150, blur_kernel_size=(5, 5)):
    """
    Applies Canny edge detection to the input image.
    Parameters:
        img: Input image.
        low_threshold: Lower threshold for the hysteresis procedure.
        high_threshold: Upper threshold for the hysteresis procedure.
        blur_kernel_size: Kernel size for Gaussian blur to reduce noise.
    Returns:
        Original image and the edges detected.
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return img, edges


def remove_eyebrows(image, eyebrow, expansion=10):
    """
    Removes the eyebrows from the input image using inpainting.
    Parameters:
        image: Input image.
        eyebrow: List of points defining the eyebrow.
        expansion: Expansion size for the eyebrow mask.
    Returns:
        Image with eyebrows removed.
    """
    eyebrow = [point for point in eyebrow if point is not None]

    if not eyebrow:  # Check if the list is empty
        raise ValueError("No valid eyebrow points provided.")

    eyebrow_array = np.array(eyebrow, dtype=np.int32)

    # Create a convex hull around the eyebrow points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(eyebrow_array)

    # Expand the mask to include surrounding pixels
    mask_hull = np.zeros_like(mask)
    cv2.fillPoly(mask_hull, [hull], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
    expanded_mask = cv2.dilate(mask_hull, kernel, iterations=1)

    # Inpaint to remove the eyebrow
    inpainted_image = cv2.inpaint(image, expanded_mask, inpaintRadius=100, flags=cv2.INPAINT_TELEA)
    return inpainted_image

import numpy as np


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points in 2D (x, y).
    Parameters:
        p1: First point (common point).
        p2: Second point.
        p3: Third point.
    Returns:
        Angle in degrees between the vectors p1p2 and p1p3.
    """
    # Convert points to numpy arrays (using only x and y)
    a = np.array(p1[:2])  # Use only x and y
    b = np.array(p2[:2])  # Use only x and y
    c = np.array(p3[:2])  # Use only x and y

    # Vectors from p1 to p2 (ba) and from p1 to p3 (bc)
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle between vectors ba and bc
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Use arccos to find the angle in radians
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle)
    print(angle_degrees, "angle_degrees")
    return angle_degrees




def draw_upward_lines(image, x_averg, y_averg, angle_degrees, length=500, color=(0, 255, 0), thickness=1):

    # Convert angle to radians
    half_angle = np.radians(angle_degrees / 2)
    # Calculate the direction vectors for the two lines, making sure they point upwards
    direction1 = (np.sin(half_angle), -np.cos(half_angle))  # Upward direction
    direction2 = (np.sin(-half_angle), -np.cos(-half_angle))  # Upward direction

    # Calculate the endpoints of the lines
    endpoint1 = (int(x_averg + length * direction1[0]), int(y_averg + length * direction1[1]))
    endpoint2 = (int(x_averg + length * direction2[0]), int(y_averg + length * direction2[1]))

    # Ensure points are integers before drawing the lines
    endpoint1 = tuple(map(int, endpoint1))
    endpoint2 = tuple(map(int, endpoint2))

    # Draw the lines
    cv2.line(image, (int(x_averg), int(y_averg)), endpoint1, color, thickness)  # First line
    cv2.line(image, (int(x_averg), int(y_averg)), endpoint2, color, thickness)  # Second line

    return image


def draw_tangent_line(tim, x_point, y_point):
    # Generate some curve data (for demonstration, a sine wave in this case)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * 20 + 50  # A sine wave curve for illustration

    # Find the closest point on the curve to (x_point, y_point)
    closest_idx = np.argmin(np.abs(x - x_point))
    x_curve = x[closest_idx]
    y_curve = y[closest_idx]

    # Calculate the slope (derivative) at the point using finite differences
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]
    slope = delta_y / delta_x  # Approximate derivative

    # Calculate the tangent line at the point (x_point, y_point)
    # Equation of the tangent line: y = slope * (x - x_point) + y_point

    # Increase the length of the tangent line
    x1 = x_point - 50
    x2 = x_point + 50
    y1_tangent = slope * (x1 - x_point) + y_point
    y2_tangent = slope * (x2 - x_point) + y_point

    # Draw the tangent line
    cv2.line(tim, (int(x1), int(y1_tangent)), (int(x2), int(y2_tangent)), (255, 0, 0), 3)  # Blue tangent line

    # Draw the point on the image (red dot)
    cv2.circle(tim, (int(x_point), int(y_point)), 2, (0, 0, 255), -1)  # Red circle

    return tim


import matplotlib.pyplot as plt

def draw_vertical_line(x):
    """
    Draw a vertical line passing through the point (x, y) where y can vary.
    Parameters:
        x: The x-coordinate of the point through which the line passes.
    """
    # Define the range of y values for plotting
    y_values = [-10, 10]  # You can adjust this range based on your plot limits
    
    # Plot the vertical line
    plt.plot([x, x], y_values, label=f"Vertical line at x = {x}", color="blue")


def extended_line(p_l, P_r, length=150):

    direction = np.array(P_r) - np.array(p_l)
    line_length = np.linalg.norm(direction)
    direction = direction / line_length  # Normalize the direction vector

    # Extend the line by the given length
    p_l_extended = tuple((np.array(p_l) - direction * length).astype(int))
    P_r_extended = tuple((np.array(P_r) + direction * length).astype(int))

    return p_l_extended, P_r_extended




def draw_vertical_line(image, x, length = 100, color=(255, 0, 0), thickness=1):
    cv2.line(image, (x[0], x[1]- length),  (x[0], x[1] + length), color, thickness)

    return image

def create_mask(image, start_point, end_point):
    """
    Create a rectangular mask in a specific area of an image using OpenCV.

    :param image: Input image as a numpy array.
    :param start_point: Top-left corner of the mask area (x, y).
    :param end_point: Bottom-right corner of the mask area (x, y).
    :return: Masked image (numpy array).
    """
    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype="uint8")  # Black mask

    # Draw the rectangle
    cv2.rectangle(mask, start_point, end_point, 255, -1)  # White rectangle mask

    # Apply the mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    white_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return white_mask, masked_image