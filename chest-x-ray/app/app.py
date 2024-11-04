# -*- coding: utf-8 -*-
"""
@author: MateoRueda
"""

from flask import (Flask, request, jsonify)
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torch.hub import load_state_dict_from_url
from torchvision.models._api import WeightsEnum


app = Flask(__name__)
CORS(app)

# Parameters
num_classes = 2
classes = ["Normal", "Pneumonia"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# Load the model
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    
    # Modify WeightsEnum to remove "check_hash"
    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash", None)  # Remove "check_hash" argument if present
        return load_state_dict_from_url(self.url, *args, **kwargs)
    
    WeightsEnum.get_state_dict = get_state_dict  # Apply the modification

    # Initialize the EfficientNet B5 model
    model = efficientnet_b5()
    
    # Modify the classifier to match the number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Load the model weights
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the appropriate device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model

# Preprocess the input data
def input_fn(request_body, content_type):
    """Deserialize and preprocess the input data."""
    if content_type == 'application/x-image':
        # Load the image from the request body
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        
        # Apply transformations
        test_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = test_transforms(image).unsqueeze(0)  # Add batch dimension
        return image.to(device)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Make predictions
def predict_fn(input_data, model):
    """Make predictions using the model and the input data."""
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs.data, 1)
        return classes[predicted.item()]

@app.route('/health', methods=['GET'])
def health():
  return jsonify(status='healthy'), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the content type is correct
        content_type = request.content_type
        if content_type != 'application/x-image':
            return jsonify({"error": "Content type must be 'application/x-image'"}), 400

        # Read the image data from the request body
        image_data = request.data

        # Preprocess the image
        input_data = input_fn(image_data, content_type)

        # Load the model (you can specify the model directory if needed)
        model = model_fn('./')

        # Make the prediction
        prediction = predict_fn(input_data, model)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        # Handle any errors and return a JSON response
        return jsonify({"error": str(e)}), 500
   
    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)