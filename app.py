import os
import time
import csv
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import cv2
from predictWithOCR import DetectionPredictor
import torch
import easyocr

# Initialize Flask App
app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the necessary folders
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# CSV file path
RESULTS_CSV = 'results.csv'

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Initialize YOLO model
cfg = {'model': 'yolov8n.pt', 'imgsz': 640, 'conf': 0.5, 'iou': 0.4, 'max_det': 1000}
predictor = DetectionPredictor(cfg)

# Helper function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to save results to CSV
def save_to_csv(data):
    # Create the CSV file if it doesn't exist
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(['Timestamp', 'Source', 'Detected Number Plate', 'Confidence'])
    
    # Append new data
    with open(RESULTS_CSV, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Home route to render HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading image and running detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform detection with OCR on the uploaded image
        img = cv2.imread(filepath)
        results = predictor.predict(img)  # Assuming it returns a list of detected plates
        if results:
            for plate, confidence in results:
                # Save to CSV (source: upload)
                save_to_csv([time.strftime("%Y-%m-%d %H:%M:%S"), 'Upload', plate, confidence])
        
        return jsonify({'results': results}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# Route for live stream processing
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection on each frame
            results = predictor.predict(frame)  # Assuming it returns detected plates
            
            if results:
                for plate, confidence in results:
                    # Save to CSV (source: live)
                    save_to_csv([time.strftime("%Y-%m-%d %H:%M:%S"), 'Live Stream', plate, confidence])
            
            # Encode frame for streaming
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
