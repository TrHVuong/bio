import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("CUDA disabled manually")

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import requests

MODEL_PATH = "model/best_model.keras"
MODEL_URL = os.getenv("MODEL_URL")

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}")
    os.makedirs("model", exist_ok=True)
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")
else:
    print("Model already exists.")

from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    'ape', 'beans', 'bee', 'bird', 'buffalo', 'butterfly', 'cabbage', 'cat',
    'cauliflower', 'chicken', 'chipmunk', 'coffee', 'cow', 'cucumber', 'deer',
    'dog', 'dragon_fruits', 'elephant', 'fish', 'fox', 'ginger', 'goat',
    'horse', 'jackfruit', 'jelly_fish', 'lion', 'litchi', 'longan', 'lotus',
    'maize', 'mouse', 'orchid', 'panther', 'papaya', 'peacock', 'pig',
    'potato', 'rabbit', 'rice_plant', 'seal', 'snake', 'spider', 'tiger',
    'tomato', 'turtle', 'zebra'
]
CLASS_NAMES_VI = {
    'ape': 'Khỉ',
    'beans': 'Đậu',
    'bee': 'Ong',
    'bird': 'Chim',
    'buffalo': 'Trâu',
    'butterfly': 'Bướm',
    'cabbage': 'Bắp cải',
    'cat': 'Mèo',
    'cauliflower': 'Súp lơ',
    'chicken': 'Gà',
    'chipmunk': 'Sóc chuột',
    'coffee': 'Cà phê',
    'cow': 'Bò',
    'cucumber': 'Dưa chuột',
    'deer': 'Hươu',
    'dog': 'Chó',
    'dragon_fruits': 'Thanh long',
    'elephant': 'Voi',
    'fish': 'Cá',
    'fox': 'Cáo',
    'ginger': 'Gừng',
    'goat': 'Dê',
    'horse': 'Ngựa',
    'jackfruit': 'Mít',
    'jelly_fish': 'Sứa',
    'lion': 'Sư tử',
    'litchi': 'Vải',
    'longan': 'Nhãn',
    'lotus': 'Hoa sen',
    'maize': 'Ngô',
    'mouse': 'Chuột',
    'orchid': 'Hoa lan',
    'panther': 'Báo đen',
    'papaya': 'Đu đủ',
    'peacock': 'Công',
    'pig': 'Lợn',
    'potato': 'Khoai tây',
    'rabbit': 'Thỏ',
    'rice_plant': 'Cây lúa',
    'seal': 'Hải cẩu',
    'snake': 'Rắn',
    'spider': 'Nhện',
    'tiger': 'Hổ',
    'tomato': 'Cà chua',
    'turtle': 'Rùa',
    'zebra': 'Ngựa vằn'
}


def recognize_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    class_en = CLASS_NAMES[class_idx]
    class_vi = CLASS_NAMES_VI[class_en]

    
    return class_vi, float(confidence)
