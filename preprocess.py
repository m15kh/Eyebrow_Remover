import numpy as np
import cv2
import matplotlib.pyplot as plt




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





def draw_tangent_line(tim, point, p_ref_u, p_ref_d, color, thickness):
    x_p_ref_u, y_p_ref_u = p_ref_u
    x_point, y_point = point
    x_p_ref_d, y_p_ref_d = p_ref_d

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
    # Define the length of the tangent line segment
    length_down = y_p_ref_d - y_point
    length_up = y_point - y_p_ref_u

    # Calculate the end points of the tangent line segment (downward direction)
    x2_down = x_point + length_down / np.sqrt(1 + slope**2)
    y2_down_tangent = y_point + slope * (x2_down - x_point)

    # Calculate the end points of the tangent line segment (upward direction)
    x2_up = x_point - length_up / np.sqrt(1 + slope**2)
    y2_up_tangent = y_point - slope * (x_point - x2_up)

    # Ensure the tangent line is constrained within `p_ref_u` and `p_ref_d`
    if y2_down_tangent > y_p_ref_d:
        y2_down_tangent = y_p_ref_d
        x2_down = x_point + (y2_down_tangent - y_point) / slope

    if y2_up_tangent < y_p_ref_u:
        y2_up_tangent = y_p_ref_u
        x2_up = x_point + (y2_up_tangent - y_point) / slope

    # Draw the tangent line
    cv2.line(
        tim,
        (int(x2_up), int(y2_up_tangent)),
        (int(x2_down), int(y2_down_tangent)),
        color,
        thickness,
    )  # Blue tangent line

    # Draw the point on the image (red dot)
    # cv2.circle(tim, (int(x_point), int(y_point)), 2, color, thickness)  # Red circle

    return tim





def extend_lines(p_left, p_right, p_xleft_ref, p_xright_ref):
    # Extract x and y coordinates
    x_left, y_left = p_left
    x_right, y_right = p_right
    x_left_ref, y_left_ref = p_xleft_ref
    x_right_ref, y_right_ref = p_xright_ref

    # Calculate the slope and y-intercept of the line between p_left and p_right
    if x_right != x_left:
        slope = (y_right - y_left) / (x_right - x_left)
        intercept_left = y_left - slope * x_left  # y = mx + b -> b = y - mx
    else:
        # Special case: vertical line, no slope, just use x value
        slope = None
        intercept_left = x_left  # The line is vertical at x = x_left

    # Extend the line until it reaches x_left_ref and x_right_ref
    if slope is not None:
        y_left_extended = int(slope * x_left_ref + intercept_left)
        p_left_extended = (x_left_ref, y_left_extended)
    else:
        p_left_extended = (x_left_ref, y_left)

    if slope is not None:
        y_right_extended = int(slope * x_right_ref + intercept_left)
        p_right_extended = (x_right_ref, y_right_extended)
    else:
        p_right_extended = (x_right_ref, y_right)

    # Return the extended points
    return p_left_extended, p_right_extended



def draw_vertical_line(image, x, p_ref_d, p_ref_u , color=(255, 0, 0), thickness=1):
    # Extract the x, y coordinates of the reference point (p_refrence)
    # Extract the x, y coordinates of the starting point (x)
    xx, xy = x

    # Draw a vertical line from x to the y-coordinate of p_refrence
    # The x-coordinate stays the same, only y changes.
    cv2.line(image, (xx, p_ref_u[1]), (xx, p_ref_d[1]), color, thickness)

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


def connect_noise_points(tim, point_1, point_2, line_color, line_thickness):
    cv2.line(tim, point_1, point_2, line_color, line_thickness)
    return tim