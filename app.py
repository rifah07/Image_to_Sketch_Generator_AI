import logging

from flask import Flask, request, redirect, url_for, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
SKETCH_FOLDER = 'static/sketches'

# Ensure the upload and sketch folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SKETCH_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SKETCH_FOLDER'] = SKETCH_FOLDER

def image_to_sketch(image_path, output_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Error: Could not read the image at {image_path}.")
            return

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert the gray image
        inverted_gray = 255 - gray

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

        # Invert the blurred image
        inverted_blurred = 255 - blurred

        # Create the pencil sketch by combining the gray image with the inverted blurred image
        sketch = cv2.divide(gray, inverted_blurred, scale=256.0)

        # Save the sketch
        cv2.imwrite(output_path, sketch)
        logger.debug(f"Sketch saved to {output_path}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                logger.error("No file part in the request.")
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an empty file without a filename
            if file.filename == '':
                logger.error("No selected file.")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                output_path = os.path.join(app.config['SKETCH_FOLDER'], filename)

                # Save the uploaded file
                file.save(input_path)
                logger.debug(f"File saved to {input_path}")

                # Convert the image to a sketch
                image_to_sketch(input_path, output_path)

                # Return the sketch image in the response
                sketch_url = url_for('static', filename=f'sketches/{filename}')
                return render_template('index.html', sketch_url=sketch_url)
        except Exception as e:
            logger.error(f"Error handling the upload: {e}")
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)