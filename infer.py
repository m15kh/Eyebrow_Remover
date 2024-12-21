import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from angle import calculate_angle


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


if __name__ == '__main__':
    # Initialize the face analysis model
    app = FaceAnalysis(allowed_modules=['detection', 'genderage', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load the input image
    input_image_path = '/home/ubuntu/m15kh/eyebrow/insightface/alignment/coordinate_reg/test/tst2.png'
    img = cv2.imread(input_image_path)

    # Extract the folder name from the input image name
    folder_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_folder = os.path.join('./output', folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Analyze faces in the image
    faces = app.get(img)
    tim = img.copy()  # Copy of the original image for visualization
    color = (0, 0, 255)  # Color for landmarks

    # Initialize landmark variables
    p43, p44, p45, p46, p47, p48, p49, p50, p51 = [None] * 9
    p97, p98, p99, p100, p101, p102, p103, p104, p105 = [None] * 9

    for face in faces:
        # Get landmarks and round to integer values
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(np.int64)

        # Draw landmarks and label them
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 3, color, 1, cv2.LINE_AA)
            label = str(i)

            # Assign specific landmarks to variables
            if i == 43: p43 = p
            elif i == 44: p44 = p
            elif i == 45: p45 = p
            elif i == 46: p46 = p
            elif i == 47: p47 = p
            elif i == 48: p48 = p
            elif i == 49: p49 = p
            elif i == 50: p50 = p
            elif i == 51: p51 = p
            elif i == 97: p97 = p
            elif i == 98: p98 = p
            elif i == 99: p99 = p
            elif i == 100: p100 = p
            elif i == 101: p101 = p
            elif i == 102: p102 = p
            elif i == 103: p103 = p
            elif i == 104: p104 = p
            elif i == 105: p105 = p
            elif i == 71: p71 = p
            elif i == 76: p76 = p
            elif i == 82: p82 = p

            cv2.putText(tim, label, (p[0] + 5, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        print(f"Gender: {face.gender}, Age: {face.age}")

        cv2.line(tim, p50, p51, (255, 0, 0), 1)  # Line between left eyebrow points
        cv2.line(tim, p102, p103, (255, 0, 0), 1)  # Line between right eyebrow points
        cv2.line(tim, p49, p104, (255, 0, 0), 1)  # Line connecting upper points of both eyebrows
        cv2.line(tim, p76, p50, (255, 0, 0), 1)  
        cv2.line(tim, p82, p102, (255, 0, 0), 1) 
        cv2.line(tim, p71, p43, (255, 0, 0), 1)  #libe between left eye and nose
        cv2.line(tim, p71, p101, (255, 0, 0), 1)  #line between right eye and nose
 
        angle_degrees = calculate_angle(p71, p43, p101)

    # Save the image with landmarks and lines
    cv2.imwrite(os.path.join(output_folder, 'tst2.jpg'), tim)

    # Apply Canny edge detection
    original, edges = apply_canny_edge_detection(tim, low_threshold=50, high_threshold=150)
    cv2.imwrite(os.path.join(output_folder, 'canny-tst2.png'), edges)

    # Remove eyebrows using landmarks
    l_eyebrow = [p43, p44, p45, p47, p46, p50, p51, p49, p48, p43]
    r_eyebrow = [p97, p98, p99, p100, p101, p105, p104, p103, p102, p97]

    tim_without_eyebrows = remove_eyebrows(img, l_eyebrow)
    tim_without_eyebrows = remove_eyebrows(tim_without_eyebrows, r_eyebrow)

    # Save the image with eyebrows removed
    cv2.imwrite(os.path.join(output_folder, 'tst2-hide.jpg'), tim_without_eyebrows)

    print(f"Processing complete. Results saved in folder: {output_folder}")
