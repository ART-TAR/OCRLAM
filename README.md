# Инструкция по запуску

## Рекомендуемые требования к ПК

- 16 ГБ оперативной памяти или 8гб видеопамяти

- Windows или Linux

## Необходимо установить:

### Вариант 1. (в систему)

- Python 3.10

- Pytorch:
Выбрать соотвествующую ОС и устройство (CUDA/CPU)
https://pytorch.org/get-started/locally/

- pip install git+https://github.com/huggingface/transformers accelerate
- pip install qwen-vl-utils==0.0.8 
- pip install huggingface_hub

### Вариант 2. (в conda)

- conda create -n qwenocr python==3.10

Далее как о варианте 1

### Вариант 3. (в venv)
Создать venv и как во варианте 1


## Использование

1. python qwenocr.py --image путь_к_изображению

