import streamlit as st
import cv2
import numpy as np
import pandas as pd
import ast
import csv
import sqlite3
from ultralytics import YOLO
from sort.sort import *
import util
import add_missing_data
import os
import cv2 as cv


st.title("Vehicle and License Plate Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    mot_tracker = Sort()

 
    cap = cv2.VideoCapture(uploaded_file.name)
    results = {}
    frame_nmr = -1
    vehicles = [2, 3, 5, 7]  

    # Process video frames
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

                if car_id != -1: 
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    license_plate_text, license_plate_text_score = util.read_license_plate(
                        license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

    # Write results to CSV
    output_csv = './test.csv'
    util.write_csv(results, output_csv)

    # Interpolate missing data
    with open(output_csv, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    interpolated_data = add_missing_data.interpolate_bounding_boxes(data)

    final_csv = 'final.csv'
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
              'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open(final_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

    res = pd.read_csv(final_csv)
    cap = cv2.VideoCapture(uploaded_file.name)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video_path = './output1.mp4'
    output = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(res['car_id']):
        car_id = int(car_id)
        idx = res[res['car_id'] == car_id]['license_number_score'].idxmax()
        license_number = str(res.loc[idx, 'license_number'])
        frame_number = int(res.loc[idx, 'frame_nmr'])
        license_plate_bbox = res.loc[idx, 'license_plate_bbox']

        license_number = license_number.replace(' ', '')

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning(f"Warning: Unable to read frame {frame_number}.")
            continue

        x1, y1, x2, y2 = map(int, ast.literal_eval(
            license_plate_bbox.replace(' ', ',')))

        plate_crop = frame[y1:y2, x1:x2]
        plate_crop = cv.resize(
            plate_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id] = {
            'license_crop': plate_crop,
            'license_plate_nmb': license_number
        }

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = 0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    frame_window = st.empty()
    with st.spinner("Processing video frames..."):
        for frame_nmr in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            df = res[res['frame_nmr'] == frame_nmr]
            for row_idx in range(len(df)):
                try:
                    
                    x1car, y1car, x2car, y2car = map(int, ast.literal_eval(
                        df.iloc[row_idx]['car_bbox'].replace(' ', ',')))
                    cv.rectangle(frame, (x1car, y1car),
                                 (x2car, y2car), (0, 255, 0), 3)


                    car_id = df.iloc[row_idx]['car_id']
                    if car_id in license_plate:
                        text = license_plate[car_id]['license_plate_nmb']

                        if text not in ['0', '-1']:
                            (text_width, text_height), baseline = cv.getTextSize(
                                text,
                                cv.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                3
                            )

                            # Add white background for text
                            frame[y1car - text_height - 10:y1car,
                                  x1car:x1car + text_width] = (255, 255, 255)

                            # Overlay text
                            cv.putText(frame, text, (x1car, y1car - 5),
                                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                except Exception as e:
                    st.error(
                        f"Error processing frame {frame_nmr}, car_id {df.iloc[row_idx]['car_id']}: {e}")
                    continue

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_window.image(rgb_frame, channels="RGB")

            output.write(frame)


            progress_bar.progress(min(frame_nmr / total_frames, 1.0))

    cap.release()
    output.release()

    conn = sqlite3.connect('vehicle_info.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_info (
        car_id INTEGER,
        license_number TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()

    # Insert data into the database
    for car_id in np.unique(res['car_id']):
        car_id = int(car_id) 
        idx = res[res['car_id'] == car_id]['license_number_score'].idxmax()
        license_number = str(res.loc[idx, 'license_number'])

        cursor.execute('''
        INSERT INTO vehicle_info (car_id, license_number)
        VALUES (?, ?)
        ''', (car_id, license_number))
    conn.commit()

    # Query the database to display the results
    cursor.execute('SELECT car_id, license_number, timestamp FROM vehicle_info')
    vehicle_data = cursor.fetchall()
    conn.close()

   
    st.subheader("Detected License Plate Information")
    vehicle_df = pd.DataFrame(
        vehicle_data, columns=['Car ID', 'License Number', 'Timestamp'])
    st.dataframe(vehicle_df)

    st.success("Processing complete!")
    # st.video(output_video_path)

