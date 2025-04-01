# -- coding: utf-8 --
import argparse
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Process an image with model')
parser.add_argument('--image', type=str, help='Path to the input image', required=True)
args = parser.parse_args()

# Проверка существования файла
if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image file not found: {args.image}")

path_to_image = args.image

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {
    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    "device_map": "auto" if device == "cuda" else None
}

# Загрузка модели с учетом доступности GPU
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "ARTART123/OCRRP", 
        load_in_8bit=True if device == "cuda" else False,
        **kwargs
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to CPU with float32 precision")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "ARTART123/OCRRP", 
        torch_dtype=torch.float32,
        device_map=None
    )

if device == "cpu":
    model = model.to('cpu')

# Загрузка процессора
processor = AutoProcessor.from_pretrained("ARTART123/OCRRP")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": path_to_image},
            {"type": "text", "text": "Please read text from image."},
        ],
    }
]

# Подготовка входных данных
try:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
except Exception as e:
    print(f"Error processing inputs: {e}")
    exit()

# Генерация ответа
try:
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    print("Result:", output_text)
except Exception as e:
    print(f"Error during generation: {e}")