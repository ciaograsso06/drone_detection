# 📡 Drone Detection with YOLOv8

Este projeto utiliza **YOLOv8** para detectar drones em imagens e disponibiliza uma **API Flask** para fazer previsões em tempo real.

---

## 📁 Estrutura do Projeto

```plaintext
drone_detection/
│── dataset/
│   │── images/             # Imagens do dataset
│   │── labels/             # Labels no formato YOLO
│   └── dataset.yaml        # Configuração do dataset
│── model/
│   │── train.py            # Script de treinamento
│   │── validate.py         # Script de validação
│   └── best.pt             # Modelo treinado
│── api/
│   │── app.py              # API Flask
│   └── requirements.txt    # Dependências
└── README.md               # Documentação do projeto
```

---

## 🚀 Instalação

### 1️⃣ Instalar dependências

```bash
pip install ultralytics opencv-python flask
```

### 2️⃣ Baixar e preparar o dataset
Caso o dataset não esteja no formato YOLO, converta as anotações para:
```plaintext
<classe> <x_center> <y_center> <width> <height>
```

Exemplo:
```plaintext
0 0.5 0.5 0.3 0.3
```

---

## 🎯 Treinar o Modelo

Execute o script para treinar o modelo YOLOv8:
```bash
python model/train.py
```

Arquivo `train.py`:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Modelo base
model.train(data="dataset/dataset.yaml", epochs=50, imgsz=640)
model.export(format="onnx")
```

---

## ✅ Validar o Modelo

```bash
python model/validate.py
```

Arquivo `validate.py`:
```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
metrics = model.val()
print(metrics)
```

---

## 🌍 Executar a API Flask

Para rodar a API, execute:
```bash
python api/app.py
```

Agora, envie uma imagem para previsão:
```bash
curl -X POST -F "file=@test.jpg" http://127.0.0.1:5000/predict
```

Saída esperada:
```json
{
  "detections": [
    {"x1": 150, "y1": 200, "x2": 400, "y2": 450}
  ]
}
```

---

## 📌 Contribuição
Sinta-se à vontade para abrir **issues** e **pull requests**! 🚀

---

## 📜 Licença
Este projeto está sob a licença **MIT**.

