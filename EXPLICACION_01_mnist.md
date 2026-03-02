# Explicación del notebook `01_mnist.ipynb`

## Resumen
Breve guía en español que explica las partes principales del código: carga de librerías, detección de dispositivo (GPU/CPU), carga y preparación del dataset MNIST, definición del modelo, funciones de entrenamiento/validación, bucle de entrenamiento e inferencia.

---

## 1) Importaciones y utilidades
- Se importan módulos de PyTorch: `torch`, `torch.nn`, `DataLoader`, `Adam`.
- `torchvision` y `transforms` se usan para cargar MNIST y convertir imágenes PIL a tensores.
- `matplotlib.pyplot` para visualización.

Código relevante:
```py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
```

## 2) Selección del dispositivo
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` determina si usar GPU (`cuda`) o CPU.
- Mantener código que usa `to(device)` o `.cuda()` hace que el mismo script funcione en ambos entornos.

## 3) Carga del dataset MNIST
- `torchvision.datasets.MNIST("./data/", train=True/False, download=True)` descarga y crea `train_set` y `valid_set`.
- Cada elemento `train_set[i]` devuelve una tupla `(imagen_PIL, etiqueta_int)`.

Código:
```py
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)
```

## 4) Transformaciones y DataLoaders
- Se define `trans = transforms.Compose([transforms.ToTensor()])` para convertir PIL→tensor y normalizar el rango a [0.0, 1.0].
- Se asigna esta transformación a `train_set.transform` y `valid_set.transform`.
- `DataLoader(..., batch_size=32, shuffle=True)` crea iteradores que devuelven batches.

Código:
```py
trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32)
```

## 5) Definición del modelo
- Se construye un `nn.Sequential` con las capas:
  - `nn.Flatten()` convierte `C x H x W` → vector.
  - `nn.Linear(input_size, 512)` + `nn.ReLU()` (capa de entrada)
  - `nn.Linear(512, 512)` + `nn.ReLU()` (capa oculta)
  - `nn.Linear(512, n_classes)` (capa salida, 10 neuronas para dígitos 0-9)
- `input_size` = 1 * 28 * 28 porque MNIST son imágenes escala de grises 28x28.
- Modelo se mueve a `device` con `model.to(device)` y opcionalmente se compila con `torch.compile(model)` (PyTorch 2.x).

Código:
```py
input_size = 1 * 28 * 28
n_classes = 10
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
]
model = nn.Sequential(*layers)
model.to(device)
model = torch.compile(model)
```

## 6) Función de pérdida y optimizador
- `loss_function = nn.CrossEntropyLoss()` para clasificación multiclase (acepta logits sin softmax).
- `optimizer = Adam(model.parameters())` actualiza pesos mediante gradiente.

## 7) Métrica: accuracy por batch
- `get_batch_accuracy(output, y, N)` calcula la fracción correcta en un lote comparando `argmax` de `output` con `y`.
- Nota: en este notebook la función divide `correct / N` (N = tamaño total del dataset) y luego suma esa fracción por batch para obtener un acumulado; es una forma válida aunque inusual (suele dividir por el tamaño del batch para cada batch y luego promediar).

Código:
```py
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N
```

## 8) Función `train()`
Pasos principales por batch:
1. `model.train()` activa comportamiento de entrenamiento.
2. Para cada `x, y` en `train_loader`:
   - Mover `x, y` a `device`.
   - `output = model(x)` (forward).
   - `optimizer.zero_grad()` limpia gradientes previos.
   - `batch_loss = loss_function(output, y)` calcula pérdida.
   - `batch_loss.backward()` propaga gradiente.
   - `optimizer.step()` actualiza parámetros.
   - Acumular `loss` y `accuracy`.
3. Imprimir pérdida y accuracy acumulados.

Código:
```py
def train():
    loss = 0
    accuracy = 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
```

## 9) Función `validate()`
- Similar a `train()` pero en modo evaluación:
  - `model.eval()` y `with torch.no_grad()` para no calcular gradientes.
  - No se llama a `optimizer.step()`.
  - Se acumulan `loss` y `accuracy` sobre `valid_loader`.

Código:
```py
def validate():
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
```

## 10) Bucle de entrenamiento
- Se definen `epochs = 5` y alterna `train()` y `validate()` por cada época.
- Cada epoch es una pasada completa por el dataset de entrenamiento.

Código:
```py
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()
```

## 11) Inferencia rápida
- El modelo se puede invocar como función: `prediction = model(x_0_gpu)`.
- `prediction` es un vector de 10 valores (logits). `prediction.argmax(dim=1)` da la clase predicha.

## 12) Limpieza de memoria
- Para liberar memoria GPU y reiniciar el kernel se usa:
```py
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

---

## Observaciones y recomendaciones
- La forma en que se acumula `accuracy` (dividiendo por `N` en cada batch) funciona porque suman fracciones relativas al total, pero más común es calcular `correct_in_batch / batch_size` y luego promediar o sumar `correct` y dividir al final por `N`.
- Para producción o experimentos reproducibles se suele fijar la semilla (`torch.manual_seed`) y aplicar normalización en las transformaciones.
- Considerar monitorizar `loss` medio por batch en vez de suma total para interpretarlo más fácilmente.

---

Si quieres, puedo:
- Abrir el archivo creado aquí: [EXPLICACION_01_mnist.md](EXPLICACION_01_mnist.md)
- Ajustar el nivel de detalle (más/menos código o diagramas).
- Generar una versión en inglés o una presentación breve.
