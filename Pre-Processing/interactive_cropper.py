import os
import pydicom
import cv2
import json
import numpy as np

# --- Configuration ---
DICOM_FOLDER = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/LODOX_Scans' 
COORDINATES_FILE = 'cropping_coordinates.json'
MAX_DISPLAY_WIDTH = 1920  # Max width of the display window in pixels
MAX_DISPLAY_HEIGHT = 1080 # Max height of the display window in pixels
# --------------------

ref_point = []
cropping_complete = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping_complete
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping_complete = True
        cv2.rectangle(image_display_resized, ref_point[0], ref_point[1], (255, 255, 255), 2)
        cv2.imshow("image", image_display_resized)

all_files = [f for f in os.listdir(DICOM_FOLDER) if f.lower().endswith('.dcm')]
cropping_coordinates = {}

for filename in all_files:
    path = os.path.join(DICOM_FOLDER, filename)
    dicom_file = pydicom.dcmread(path)
    image = dicom_file.pixel_array.astype(np.float32)

    # Normalize original image
    if image.max() > image.min():
        image_normalized = (image - image.min()) / (image.max() - image.min())
    else:
        image_normalized = np.zeros_like(image)
    
    # --- Create a resized version for display ---
    original_height, original_width = image_normalized.shape
    scale_w = MAX_DISPLAY_WIDTH / original_width
    scale_h = MAX_DISPLAY_HEIGHT / original_height
    scale = min(scale_w, scale_h) # Use the smaller scale factor to fit within bounds
    
    display_width = int(original_width * scale)
    display_height = int(original_height * scale)
    
    image_for_display = (image_normalized * 255).astype(np.uint8)
    image_display_resized = cv2.resize(image_for_display, (display_width, display_height))
    image_display_resized = cv2.cvtColor(image_display_resized, cv2.COLOR_GRAY2BGR)
    # ---------------------------------------------
    
    clone = image_display_resized.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    print(f"\nProcessing: {filename}")
    print("Please draw a rectangle around the object of interest.")
    print("Press 'r' to reset the crop. Press 'c' to confirm and move to the next image.")
    
    cropping_complete = False
    while not cropping_complete:
        cv2.imshow("image", image_display_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image_display_resized = clone.copy()
        elif key == ord("c"):
            break
            
    if len(ref_point) == 2:
        # --- Scale the coordinates back to the original image size ---
        x1_scaled, y1_scaled = ref_point[0]
        x2_scaled, y2_scaled = ref_point[1]
        
        x1_orig = int(x1_scaled / scale)
        y1_orig = int(y1_scaled / scale)
        x2_orig = int(x2_scaled / scale)
        y2_orig = int(y2_scaled / scale)
        # -----------------------------------------------------------
        
        start_x, end_x = min(x1_orig, x2_orig), max(x1_orig, x2_orig)
        start_y, end_y = min(y1_orig, y2_orig), max(y1_orig, y2_orig)
        
        cropping_coordinates[filename] = {
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y
        }
        print(f"Coordinates for {filename} saved.")

    cv2.destroyAllWindows()

# Save all coordinates to a JSON file
with open(COORDINATES_FILE, 'w') as f:
    json.dump(cropping_coordinates, f, indent=4)

print(f"\nAll cropping coordinates have been saved to {COORDINATES_FILE}")