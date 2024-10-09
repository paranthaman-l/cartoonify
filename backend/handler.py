import json
import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from network.Transformer import Transformer  # Ensure this path is correct

app = Flask(__name__)

# Initialize your S3 client and models here
gpu = -1
bucket = "cartoongan"
mapping_id_to_style = {0: "Hosoda", 1: "Hayao", 2: "Shinkai", 3: "Paprika"}

# Function to load models (you'll need to implement this as needed)
def load_models():
    models = {}
    for style in mapping_id_to_style.values():
        model = Transformer()
        # Add logic to load the model from S3 or local path
        model.load_state_dict(torch.load(os.path.join("./pretrained_models", style + '_net_G_float.pth')))

        model.eval()
        models[style] = model
    return models

models = load_models()

def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

@app.route('/transform', methods=['POST'])
def transform_image():
    """
    Endpoint to handle image transformation requests.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    image = data["image"]
    image = image[image.find(",") + 1:]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec)).convert("RGB")

    model_id = int(data["model_id"])
    load_size = int(data["load_size"])
    style = mapping_id_to_style[model_id]
    model = models[style]

    if gpu > -1:
        model.cuda()  # for GPU
    else:
        model.float() # for CPU
        
    # Get the original image size
    orig_h, orig_w = image.size

    # Resize the image to the specified load_size while maintaining the aspect ratio
    # ratio = orig_h * 1.0 / orig_w
    # if ratio > 1:
    #     new_h = load_size
    #     new_w = int(new_h * 1.0 / ratio)
    # else:
    #     new_w = load_size
    #     new_h = int(new_w * ratio)

    image = image.resize((orig_w, orig_h), Image.BICUBIC)
    image = np.asarray(image)[:, :, [2, 1, 0]]  # RGB to BGR
    image = transforms.ToTensor()(image).unsqueeze(0)

    # Preprocess, (-1, 1)
    image = -1 + 2 * image
    if gpu > -1:
        image = Variable(image).cuda()
    else:
        image = Variable(image).float()

    with torch.no_grad():
        output_image = model(image)[0]

    output_image = output_image[[2, 1, 0], :, :]  # BGR to RGB
    output_image = output_image.cpu().float() * 0.5 + 0.5
    output_image = np.uint8(output_image.numpy().transpose(1, 2, 0) * 255)

    # Resize the output image to match the original input size
    output_image = Image.fromarray(output_image)
    output_image = output_image.resize((orig_w, orig_h), Image.BICUBIC)

    result = {
        "output": img_to_base64_str(output_image)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2004)