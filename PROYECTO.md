# TP Final — Drowsiness Detection: MobileNetV2 vs ResNet50V2

Trabajo final de Visión Computadora II (CEIA-FIUBA).
Compara dos CNNs fine-tuneadas para detectar somnolencia desde crops faciales.

> **Estado actual:** tarea **binaria** (alerta vs somnoliento). Ver sección
> "Hallazgos / decisiones de modelado" abajo — el 3-clases original no generaliza.

## Entorno

Se usa el `.venv` de la materia de **PLN** (`pnl/.venv`, Python 3.11) porque ya tiene
GPU/torch. Se le agregaron `opencv-python`, `albumentations` y `h5py`.

```bash
source ../../pnl/.venv/bin/activate
```

GPU disponible: **RTX 4090 Laptop (16 GB)**, CUDA OK.

### ⚠️ mediapipe NO se usa (WSL2)

MediaPipe **segfaultea en WSL2** al inicializar EGL/OpenGL (Tasks API ≥0.10 falla
incluso con `Delegate.CPU`). Por eso `src/detection/face_mesh.py` fue reescrito con
el **Haar cascade de OpenCV** (CPU, sin OpenGL). Genera landmarks sintéticos (478 pts)
para el bbox y un EAR aproximado (NO usado en entrenamiento). `face_detector.py`
sigue con mediapipe pero **no se usa** en ningún pipeline.

## Dataset: UTA-RLDD

Estructura real en `data/raw/` (por sujeto), ya colocada:

```
data/raw/
  01/  → 0.mov (alerta), 5.mov (baja vig.), 10.MOV (somnoliento)
  ...                                          # 60 sujetos, extensiones .mov/.MOV/.mp4
```

## Pipeline de datos (HDF5 lazy)

```bash
# 1. Extraer crops → crops.h5 (HDF5 incremental, RAM constante)
python -m src.data.extract_crops --data-dir data/raw --output data/processed/crops.h5 --stride 12

# 2. Optimizar para entrenamiento (re-chunk 1img/chunk + copia a ~). IMPRESCINDIBLE.
python -m src.data.optimize_h5 --input data/processed/crops.h5 --to-home
#    → ~/drowsiness_crops.h5  (disco Linux, acceso aleatorio rápido)
```

**Por qué los 2 pasos:** `crops.h5` pesa ~33 GB y no entra en RAM. `extract_crops`
escribe por bloques. `optimize_h5` re-chunkea a 1 imagen/chunk (sin esto, el
DataLoader con shuffle lee ~77 MB por imagen → entrenamiento impracticable).

Resultado: **235.575 crops**, 60 sujetos, clases balanceadas (~35/33/32%).
Datasets en el `.h5`: `X` (N,224,224,3 uint8), `y`, `subjects`, `ear`.

### API de `dataset.py`
- `make_splits(h5_path, keep_classes=None)` → split **por sujeto** (sin fuga), devuelve
  índices. `keep_classes=[0,2]` filtra a binario.
- `make_loaders(h5_path, train_idx, val_idx, test_idx, batch_size, num_workers, label_map=None)`
  → DataLoaders lazy. `label_map={0:0, 2:1}` remapea labels para binario.
- `DrowsinessDataset` lee un crop a la vez del disco (lazy, 1 file/worker).
- `build_transforms(augment)`: train con **augmentation agresiva** (RandomResizedCrop,
  color fuerte, ToGray, CoarseDropout) para romper atajos espurios de iluminación/fondo;
  val/test con CenterCrop central determinista.

## Orden de ejecución (notebooks)

`H5 = '/home/lilidl/drowsiness_crops.h5'` en las notebooks. Lanzar Jupyter con raíz en
el padre: el script `pnl/levantar_jupyter.sh` usa `--ServerApp.root_dir=..`.

1. `01_extract_crops.ipynb` → verificar distribución (lee del `.h5` lazy)
2. `02_training.ipynb` → entrena **MobileNetV2 y ResNet50V2 en binario**, guarda
   checkpoints + history JSON, plotea curvas
3. `03_evaluation.ipynb` → evalúa ambos sobre el **test set** (21.5k crops), matrices
   de confusión, tabla comparativa, dwell time

## Configuración de entrenamiento

| Param | Valor | Nota |
|---|---|---|
| Arch | MobileNetV2 / ResNet50V2 | pesos ImageNet |
| `num_classes` | 2 | binario alerta/somnoliento |
| Fase 1 | 10 ep, lr 1e-3, solo cabeza | warmup, **sin early stopping** |
| Fase 2 | 20 ep, lr 1e-4, +2 bloques | early stopping patience 6 (local a la fase) |
| `dropout` | 0.2 | cabeza |
| weight_decay | 1e-4 | Adam |
| batch_size | 64 | la 4090 va holgada |

**Lógica de `train_model` (importante):** la fase 1 corre completa SIN early stopping
(el backbone congelado se estanca de entrada; cortar ahí saltearía el fine-tuning).
La fase 2 SIEMPRE se ejecuta, con early stopping relativo a su propio mejor val.
El checkpoint guarda el mejor **global** y registra `num_classes`.

## Hallazgos / decisiones de modelado

Recorrido del diagnóstico (clave para el informe):

1. **3-clases + split por sujeto → val ≈ azar (0.31).** El modelo memoriza train
   (0.98) y no transfiere a sujetos nuevos. No es bug ni BatchNorm (verificado):
   las predicciones se reparten al azar.
2. **Causa raíz:** en RLDD cada sujeto graba 3 videos separados (1 por estado actuado)
   → distinta iluminación/fondo/sesión. El modelo aprende **atajos de sesión**, no
   somnolencia. Además las etiquetas son a nivel de **video, no de frame** (en el video
   "somnoliento" hay muchos frames con ojos abiertos idénticos a "alerta") → etiquetas
   ruidosas por frame. La señal de somnolencia que generaliza es **temporal**.
3. **Augmentation agresiva** (color/crop/grayscale) redujo el overfitting pero no
   alcanzó para el 3-clases.
4. **Binario (alerta vs somnoliento) SÍ generaliza:** descartar la clase intermedia
   ambigua lleva el val a **~0.65–0.67** sobre sujetos nuevos. Hay señal facial real.
5. **Tuning de regularización (dropout/wd/label-smoothing, 1 bloque) NO mejoró** —
   incluso empeoró. La config A (2 bloques, dropout 0.2, wd 1e-4) fue la mejor.
   **~0.65 es el techo** del enfoque frame-a-frame binario con split honesto por sujeto.

**Expectativa de resultados:** ambos modelos ~F1 0.65 en test. No llega al 0.80
objetivo original, pero es un resultado legítimo; el valor del TP está en el análisis
del porqué (atajos de sesión, etiquetas por frame, necesidad de modelado temporal).

## Estructura de módulos

```
src/
├── detection/
│   ├── face_detector.py   # mediapipe Tasks API — NO usado (segfault WSL2)
│   └── face_mesh.py        # FaceMesh con Haar cascade de OpenCV
├── features/ear.py         # EAR + BlinkDetector (EAR sintético, no usado en train)
├── data/
│   ├── extract_crops.py    # videos por sujeto → crops.h5 (incremental)
│   ├── optimize_h5.py      # re-chunk 1img/chunk + copia a ~
│   └── dataset.py          # lazy HDF5; make_splits(keep_classes), make_loaders(label_map)
├── models/
│   ├── backbone.py         # DrowsinessModel (num_classes, dropout configurables)
│   └── temporal.py         # TemporalSmoother (deque, modo)
├── training/
│   ├── train.py            # train_model (fase1 sin early-stop + fase2 siempre corre)
│   └── evaluate.py         # evaluate(class_names=...), load_checkpoint (lee num_classes)
└── inference/demo.py       # demo OpenCV side-by-side (ajustar a binario si se usa)
```

## Demo en tiempo real

```bash
python -m src.inference.demo --model-a checkpoints/mobilenetv2_best.pt \
    --model-b checkpoints/resnet50v2_best.pt --source 0
```
Nota: `demo.py` todavía asume 3 clases (CLASS_NAMES, colores). Ajustar a binario si
se va a usar con los checkpoints actuales.
