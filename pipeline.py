import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from preprocess import *


app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))

input_image_path = 'test/tst4.png'  #NOTE
img = cv2.imread(input_image_path)

folder_name = os.path.splitext(os.path.basename(input_image_path))[0]
output_folder = os.path.join('./output', folder_name)
os.makedirs(output_folder, exist_ok=True)

faces = app.get(img)
tim = img.copy()  
color = (0, 0, 255)  

p43, p44, p45, p46, p47, p48, p49, p50, p51 = [None] * 9
p97, p98, p99, p100, p101, p102, p103, p104, p105 = [None] * 9

for face in faces:
    lmk = face.landmark_2d_106
    lmk = np.round(lmk).astype(np.int64)

    for i in range(lmk.shape[0]):
        p = tuple(lmk[i])
        cv2.circle(tim, p, 3, color, 1, cv2.LINE_AA)
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

        cv2.putText(tim, label, (p[0] + 5, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


    cv2.line(tim, p50, p51, (255, 0, 0), 1)  # Line between left eyebrow points
    cv2.line(tim, p102, p103, (255, 0, 0), 1)  # Line between right eyebrow points
    cv2.line(tim, p76, p50, (255, 0, 0), 1)  
    cv2.line(tim, p82, p102, (255, 0, 0), 1) 
    cv2.line(tim, p71, p43, (255, 0, 0), 1)  #libe between left eye and nose
    cv2.line(tim, p71, p101, (255, 0, 0), 1)  #line between right eye and nose
    # cv2.line(tim, p51, p103, (255, 0, 0), 1)  #line between left eye and right eye
    
    
    p51_extended, p103_extended = extended_line(p51, p103)
    cv2.line(tim, p51_extended, p103_extended, (255, 0, 0), 1)  
        
    p49_extended, p104_extended = extended_line(p49, p104)  
    cv2.line(tim, p49_extended, p104_extended, (255, 0, 0), 1)  # Line connecting upper points of both eyebrows

    p43_extended, p101_extended = extended_line(p43, p101)
    cv2.line(tim, p43_extended, p101_extended, (255, 0, 0), 1)  # Line connecting left eye and right eye
    
    p46_extended, p97_extended = extended_line(p46, p97, 250)
    cv2.line(tim, p46_extended, p97_extended, (255, 0, 0), 1)  # Line connecting lower points of both eyebrows
    
    angle_degrees = calculate_angle(p71, p43, p101)
    
    y_averg = (p71[1] + p80[1]) / 2
    x_averg = p80[0]
    # x_averg = p71[0]
    # y_averg = p71[1]
    # draw_upward_lines(tim, x_averg, y_averg, angle_degrees)
    
    

    #vertical line
    draw_vertical_line(tim, p48)
    draw_vertical_line(tim, p105)
    draw_vertical_line(tim, p49)
    draw_vertical_line(tim, p104)

    # tangent line
    draw_tangent_line(tim, p50[0], p50[1])
    draw_tangent_line(tim, p102[0], p102[1])
    


#mask 
points = [p43, p101, p104, p48]
white_mask, masked_image = create_mask(tim, p45, p100)

# Remove eyebrows using landmarks shadow mode
# l_eyebrow = [p43, p44, p45, p47, p46, p50, p51, p49, p48, p43]
# r_eyebrow = [p97, p98, p99, p100, p101, p105, p104, p103, p102, p97]

# tim_without_eyebrows = remove_eyebrows(img, l_eyebrow)
# tim_without_eyebrows = remove_eyebrows(tim_without_eyebrows, r_eyebrow)
# cv2.imwrite(os.path.join(output_folder, 'tst4-hide.jpg'), tim_without_eyebrows)

cv2.imwrite(os.path.join(output_folder, 'tst4-landmarks.jpg'), tim)
original, edges = apply_canny_edge_detection(tim, low_threshold=50, high_threshold=150)
cv2.imwrite(os.path.join(output_folder, 'tst4-canny.png'), edges)
cv2.imwrite(os.path.join(output_folder, 'tst4-white_mask.png'), white_mask)
cv2.imwrite(os.path.join(output_folder, 'tst4-masked_rgb_mask.png'), masked_image)

print(f"Processing complete. Results saved in folder: {output_folder}")
