# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:14:40 2024

@author: mates
"""

import requests

# Path to your image file
image_path = "./test/pneumonia/example.png"

# URL of your Flask API
url = "http://localhost:8080/predict"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Make a POST request with the image data
    response = requests.post(url, headers={"Content-Type": "application/x-image"}, data=image_file)

# Print the response from the API
print(response.json())
