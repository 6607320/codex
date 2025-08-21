# === quest_14_2.py (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ) ===
# Квест: 14.2 - Экспорт в "Свиток ONNX"
# Цель: Научиться конвертировать модель из формата PyTorch в универсальный
# формат ONNX, а затем запустить предсказание с помощью ONNX Runtime.

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np

# --- Акт 1: Призыв Обученного Голема ---
# Мы используем ИСПРАВЛЕННЫЙ чертеж нашей MiniCNN.
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # было padding=1, исправил ошибку
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # было padding=1, исправил ошибку
        # После двух max_pool(2) картинка 28x28 станет 7x7.
        # Поэтому правильный размер 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Сплющиваем в вектор правильного размера.
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# Сотворяем "чистого" Голема и переводим в режим "экзамена".
model = MiniCNN()
model.eval()

# --- Акт 2: Ритуал "Перевода на Латынь" (Экспорт в ONNX) ---
print("Начинаю ритуал 'Перевода на Латынь' (Экспорт в ONNX)...")

dummy_input = torch.randn(1, 1, 28, 28)
onnx_model_path = "mnist_cnn.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=10,
    input_names=['input'],
    output_names=['output']
)
print(f"Перевод завершен! Универсальный 'Свиток' сохранен как '{onnx_model_path}'")

# --- Акт 3: Экзамен с помощью "Чтеца Свитков" ---
print("\nПровожу экзамен с помощью 'Универсального Чтеца Свитков'...")

ort_session = ort.InferenceSession(onnx_model_path)
input_data_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
input_name = ort_session.get_inputs()[0].name
ort_outputs = ort_session.run(None, {input_name: input_data_np})

# --- Акт 4: Вердикт ---
prediction = np.argmax(ort_outputs[0])

print("Ритуал с 'Чтецом' прошел успешно!")
print(f"  'Чтец' увидел в случайном шуме цифру: {prediction}")
print("\nМагия универсальности освоена!")