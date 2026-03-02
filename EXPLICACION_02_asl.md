# Explicación sencilla del notebook `02_asl.ipynb`

## Resumen
Esta libreta muestra cómo entrenar un modelo de aprendizaje automático para reconocer letras del alfabeto en imágenes de manos (American Sign Language, ASL). Se explica paso a paso: cargar los datos, preparar imágenes, definir el modelo, entrenarlo y evaluarlo.

## 1) Conceptos clave para empezar (sin tecnicismos)
- Imagen: un conjunto de números (píxeles). Por ejemplo, una imagen 28×28 tiene 784 números.
- Etiqueta (label): número que indica qué letra representa la imagen.
- Modelo: programa que aprende a convertir una imagen en una predicción (qué letra es).
- Entrenamiento: mostrar muchas imágenes etiquetadas para que el modelo ajuste sus parámetros.
- Validación: comprobar el modelo con imágenes nuevas para medir su verdadero desempeño.

## 2) Pasos de la libreta

### Importaciones y dispositivo
- Se usan `torch` (PyTorch) y `pandas`.
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` usa GPU si está disponible.

### Cargar los datos
- Los datos vienen en CSV (`sign_mnist_train.csv`, `sign_mnist_valid.csv`). Cada fila = etiqueta + 784 valores de píxel.
- Se leen con `pd.read_csv` y se guardan en `DataFrame`.

### Separar etiquetas e imágenes
- `y_train = train_df.pop('label')` guarda las etiquetas.
- `x_train = train_df.values` obtiene las imágenes como matriz (n_samples × 784).

### Visualizar imágenes
- Para ver una imagen se transforma una fila de 784 valores en 28×28: `row.reshape(28,28)` y se muestra con `matplotlib`.

### Normalizar valores
- Se divide por 255 para que los píxeles estén entre 0 y 1: `x_train = x_train / 255`.

### Dataset personalizado y DataLoader
- Se define `MyDataset(Dataset)` que convierte `x` e `y` a tensores de PyTorch y (en este notebook) los manda al `device` en `__init__`.
- `DataLoader` crea batches (ej. 32) para entrenar en lotes.

### Modelo (red simple)
- Arquitectura en `nn.Sequential`:
  - `nn.Flatten()` (28×28 → 784)
  - `nn.Linear(784, 512)` + `nn.ReLU()`
  - `nn.Linear(512, 512)` + `nn.ReLU()`
  - `nn.Linear(512, 26)` (salida para 26 letras)
- El modelo se mueve a GPU y se compila con `torch.compile(model.to(device))`.

### Pérdida y optimizador
- `loss_function = nn.CrossEntropyLoss()` (adecuado para clasificación multiclase).
- `optimizer = Adam(model.parameters())`.

### Métrica: accuracy por batch
- `get_batch_accuracy(output, y, N)` calcula cuántas predicciones en un batch son correctas y devuelve la fracción.

### Función `train()` (resumen)
- Activa `model.train()`.
- Para cada batch: obtiene `output`, limpia gradientes, calcula `loss`, backward, `optimizer.step()`, acumula pérdida y accuracy.

### Función `validate()` (resumen)
- Activa `model.eval()` y `with torch.no_grad()` (sin gradientes).
- Calcula pérdida y accuracy en datos de validación (no actualiza pesos).

### Bucle de entrenamiento
- Repite `train()` y `validate()` durante varias épocas (p. ej. 20) para mejorar el modelo.

### Sobreajuste (overfitting)
- Si la precisión en entrenamiento sube pero en validación no, el modelo está memorizando datos en vez de generalizar.
- Soluciones: más datos, regularización, data augmentation, redes convolucionales, etc.

### Limpieza de memoria
- Para liberar GPU se sugiere reiniciar el kernel con:
```py
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

## 3) Consejos prácticos
- Normaliza siempre (0–1) para estabilidad.
- Usa `batch_size` (ej. 32) para eficiencia y estabilidad.
- Monitorea pérdida y accuracy en entrenamiento y validación.

---

Si quieres, puedo generar un PDF con esta explicación ahora (lo crearé en la carpeta de trabajo).