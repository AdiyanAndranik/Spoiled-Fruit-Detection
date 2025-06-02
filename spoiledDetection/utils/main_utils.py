import numpy as np
from PIL import Image
import base64
from io import BytesIO

def preprocess_image(image, scaler):
    image = image.resize((64, 64))
    image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    img_flattened = img_array.flatten()
    img_scaled = scaler.transform([img_flattened])[0]
    img_cnn = img_array.reshape(1, 64, 64, 3)
    return img_scaled, img_cnn

def decodeImage(base64_string, filename):
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img.save(filename, 'PNG')
    except Exception as e:
        raise Exception(f"Error decoding image: {str(e)}")

def encodeImageIntoBase64(filename):
    try:
        with open(filename, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")