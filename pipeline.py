import os
import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis
from preprocess import *
# from df_remover import remover


# Read the configuration from JSON
with open('config.json', 'r') as f:
    config = json.load(f)

input_image_path = config.get('input_image_path')  
output_folder = config.get('output_folder')
line_thickness = config.get('line_thickness') 
line_color = tuple(config.get('line_color'))  

# Read the image
img = cv2.imread(input_image_path)
folder_name = os.path.splitext(os.path.basename(input_image_path))[0]
output_folder = os.path.join(output_folder, folder_name)
os.makedirs(output_folder, exist_ok=True)

def eyebrow_finder():
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(img)
    tim = img.copy()  

    p43, p44, p45, p46, p47, p48, p49, p50, p51 = [None] * 9
    p97, p98, p86, p100, p101, p102, p103, p104, p105 = [None] * 9

    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(np.int64)

        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 3, (0,0,0), 1, cv2.LINE_AA)  # Use line_color from the config
            label = str(i)

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
            elif i == 80: p80 = p
            elif i == 43: p43 = p
            elif i ==  86 : p86 = p

            cv2.putText(tim, label, (p[0] + 5, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,200,200), 1, cv2.LINE_AA)

        # Drawing lines with thickness from the config
        #cv2.line(tim, p50, p51, line_color, line_thickness)  # Line between left eyebrow points
        #cv2.line(tim, p102, p103, line_color, line_thickness)  # Line between right eyebrow points

        cv2.line(tim, p71, p43, line_color, line_thickness)  # Line between left eye and nose
        cv2.line(tim, p71, p101, line_color, line_thickness)  # Line between right eye and nose
        
        p51_extended, p103_extended = extend_lines(p51, p103,p43,p101)
        cv2.line(tim, p51_extended, p103_extended, line_color, line_thickness)  # Extended line between eyebrows
            
        p49_extended, p104_extended = extend_lines(p49, p104, p43, p101)  
        cv2.line(tim, p49_extended, p104_extended, line_color, line_thickness)  # Upper points of both eyebrows

        p43_extended, p101_extended = extend_lines(p43, p101,p43,p101)
        cv2.line(tim, p43_extended, p101_extended, line_color, line_thickness)  # Line connecting left eye and right eye
        
        p46_extended, p97_extended = extend_lines(p46, p97,p43,p101)
        cv2.line(tim, p46_extended, p97_extended, line_color, line_thickness)  # Line connecting lower points of both eyebrows
        
        angle_degrees = calculate_angle(p71, p43, p101)
        
        y_averg = (p71[1] + p80[1]) / 2
        x_averg = p80[0]

        # Draw vertical lines and tangent lines
        draw_vertical_line(tim, p48, p49, p86,line_color, line_thickness)
        draw_vertical_line(tim, p49, p49, p86, line_color, line_thickness)
        draw_vertical_line(tim, p104, p104, p86, line_color, line_thickness)
        draw_vertical_line(tim, p105, p104, p86, line_color, line_thickness)
        
   
        draw_tangent_line(tim, p50, p49, p86, line_color, line_thickness)
        draw_tangent_line(tim, p102, p104, p86, line_color, line_thickness)
        
        # connect_noise_points(tim, p76, p50, line_color, line_thickness)
        # connect_noise_points(tim, p82, p102, line_color, line_thickness)

    # Mask and output
    points = [p43, p101, p104, p48]
    white_mask, masked_image = create_mask(tim, p45, p100)

    cv2.imwrite(os.path.join(output_folder, 'tst2-landmarks.jpg'), tim)
    original, edges = apply_canny_edge_detection(tim, low_threshold=50, high_threshold=150)
    cv2.imwrite(os.path.join(output_folder, 'tst2-canny.png'), edges)
    cv2.imwrite(os.path.join(output_folder, 'tst2-white_mask.png'), white_mask)
    cv2.imwrite(os.path.join(output_folder, 'tst2-masked_rgb_mask.png'), masked_image)

    print(f"Processing complete. Results saved in folder: {output_folder}")
    
    return tim, white_mask



def main():
    img, mask = eyebrow_finder()
    # Remove eyebrows using mask if needed
    # remove_eyebrow = remover(img, mask, (os.path.join(output_folder, 'tst2-masked_rgb_mask.png')))
    # cv2.imwrite(os.path.join(output_folder, 'tst2-remove_eyebrows.jpg'), remove_eyebrow)

main()
