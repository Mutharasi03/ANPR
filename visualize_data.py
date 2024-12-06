import ast
import cv2 as cv
import numpy as np
import pandas as pd

# Read interpolated data
try:
    res = pd.read_csv('final.csv')
except FileNotFoundError:
    print("Error: The file 'test_interpolated.csv' was not found.")
    exit()

# Open the video file
cap = cv.VideoCapture('./sample.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file 'sample.mp4'.")
    exit()

# Initialize the video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
output = cv.VideoWriter('./output1.mp4', fourcc, fps, (width, height))

license_plate = {}

# Identify the number plate with the highest confidence score for each car
for car_id in np.unique(res['car_id']):
    try:
        idx = res[res['car_id'] == car_id]['license_number_score'].idxmax()
        license_number = str(res.loc[idx, 'license_number'])
        frame_number = int(res.loc[idx, 'frame_nmr'])
        license_plate_bbox = res.loc[idx, 'license_plate_bbox']

        license_number = license_number.replace(' ', '')

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Unable to read frame {frame_number}.")
            continue

        x1, y1, x2, y2 = map(int, ast.literal_eval(license_plate_bbox.replace(' ', ',')))

        plate_crop = frame[y1:y2, x1:x2]
        plate_crop = cv.resize(plate_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id] = {
            'license_crop': plate_crop,
            'license_plate_nmb': license_number
        }
    except Exception as e:
        print(f"Error processing car_id {car_id}: {e}")
        continue

# Reset to the first frame
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_nmr = 0

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    df = res[res['frame_nmr'] == frame_nmr]
    for row_idx in range(len(df)):
        try:
            # Draw car bounding box
            x1car, y1car, x2car, y2car = map(int, ast.literal_eval(df.iloc[row_idx]['car_bbox'].replace(' ', ',')))
            cv.rectangle(frame, (x1car, y1car), (x2car, y2car), (0,  255, 0), 3)

            # Add license plate text
            car_id = df.iloc[row_idx]['car_id']
            if car_id in license_plate:
                license_crop = license_plate[car_id]['license_crop']
                text = license_plate[car_id]['license_plate_nmb']

                if text not in ['0', '-1']:
                    # Calculate text size
                    (text_width, text_height), baseline = cv.getTextSize(
                        text,
                        cv.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        3
                    )

                    # Add white background for text
                    frame[y1car - text_height - 10:y1car, x1car:x1car + text_width] = (255, 255, 255)

                    # Overlay text
                    cv.putText(frame, text, (x1car, y1car - 5), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        except Exception as e:
            print(f"Error processing frame {frame_nmr}, car_id {df.iloc[row_idx]['car_id']}: {e}")
            continue

    # Write frame to the output video
    output.write(frame)

    frame_nmr += 1

# Release resources
cap.release()
output.release()