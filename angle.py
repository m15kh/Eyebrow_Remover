import numpy as np

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between two lines defined by three points.
    
    Parameters:
        p1 (tuple): The shared point (x1, y1) where the two lines meet.
        p2 (tuple): The endpoint of the first line (x2, y2).
        p3 (tuple): The endpoint of the second line (x3, y3).
    
    Returns:
        float: The angle between the two lines in degrees.
    """
    # Vectors for the lines
    vector1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector2 = (p3[0] - p1[0], p3[1] - p1[1])
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    # Clamp the value to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians and convert to degrees
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    print(f"The angle between the lines is {angle_degrees:.2f} degrees")

    return angle_degrees

