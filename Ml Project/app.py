from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import os
import tensorflow as tf

from backend.tf_inference import inference  # Inference function for testing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)

# Load model (or create it if training)
def create_model():
    model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Adjust for object detection tasks
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()  # Instantiate model

# Placeholder training data
def load_training_data():
    # Replace with actual data loading code
    images = np.random.rand(100, 224, 224, 3)  # Dummy data
    labels = np.random.randint(0, 2, 100)      # Dummy labels (binary)
    return images, labels

# Training function
@app.route('/api/train', methods=["POST"])
def train_model():
    images, labels = load_training_data()
    history = model.fit(images, labels, epochs=10, validation_split=0.2)

    return jsonify({
        'status': 'Training complete',
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    })

# Testing function using existing inference code
@app.route('/api/test', methods=["POST"])
def test_model():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # Remove unused part

    image = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image))

    if img.mode != 'RGB':
        img = img.convert("RGB")
    
    # Convert to numpy array
    img_arr = np.array(img)

    # Perform object detection (testing)
    results = inference(sess, detection_graph, img_arr, conf_thresh=0.5)
    
    # Plot detected objects with bounding boxes
    plot_detections(img_arr, results)

    return jsonify(results)

def plot_detections(img_arr, results):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_arr)
    for obj in results['objects']:
        ymin, xmin, ymax, xmax = obj['box']
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='red', facecolor='none', linewidth=2)
        )
        plt.text(xmin, ymin, obj['name'], color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
