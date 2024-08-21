from flask import Flask, request, send_file, render_template
from detection.DocScan import stack_img_generator
from RFCalc import RFValueCalc
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return {"error": "No image part in the request"}, 400
    
    file = request.files['image']
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    stacked_img, warped_img = stack_img_generator(img)

    stacked_img_path = os.path.join(OUTPUT_FOLDER, 'stacked.jpeg')
    cv2.imwrite(stacked_img_path, stacked_img)

    if warped_img is None:
        rf_valued_img = RFValueCalc(img)
    else:
        rf_valued_img = RFValueCalc(warped_img)

    rf_img_path = os.path.join(OUTPUT_FOLDER, 'rf_valued_img.jpeg')
    cv2.imwrite(rf_img_path, rf_valued_img)

    return {
        "stacked_image_url": f"/download/stacked.jpeg",
        "rf_calculated_image_url": f"/download/rf_valued_img.jpeg"
    }

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)
