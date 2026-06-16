# Detección de Somnolencia — UTA-RLDD

Trabajo final de Visión por Computadora II (CEIA-FIUBA). Integrantes:
- [a2308] Liliana Mariel Di Lanzo
- [a2317] Pablo Maximiliano Lulic
- [a2318] Reinaldo Magallanes Saunders

Compara MobileNetV2 vs ResNet50V2, un modelo temporal, y un enfoque de **features de parpadeo**
(EAR/PERCLOS), evaluando todo con el protocolo del paper original (Ghoddoosian et al., 2019):
**5-fold cross-validation subject-independent a nivel de video**.

> **Resultado principal:** el modelo de **parpadeo (libre de apariencia)** alcanza
> **0.692 ± 0.043** video-acc — **supera a las CNN profundas** (~0.56) y se ubica al
> nivel del baseline del paper (~0.65). El 0.80 subject-independent no es alcanzable.

**→ Empezá por `notebooks/05_recorrido_completo.ipynb`**: todo el proceso paso a paso
con evidencia, pensado para la defensa oral.

---

## 1. Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# dlib + predictor de 68 landmarks (para las features de parpadeo):
pip install dlib imutils
mkdir -p models && cd models
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

> **WSL2:** mediapipe segfaultea (OpenGL). Se usa Haar (OpenCV) para detección y
> dlib (C++/CPU) para landmarks. Ambos funcionan en WSL2.

---

## 2. Dataset

UTA-RLDD (Kaggle): https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset
Colocar por sujeto en `data/raw/` (`0.*` alerta, `5.*` baja vig., `10.*` somnoliento).

---

## 3. Pipeline completo

```bash
# (A) Crops faciales → HDF5
python -m src.data.extract_crops --data-dir data/raw --output data/processed/crops.h5 --stride 12
python -m src.data.optimize_h5   --input data/processed/crops.h5 --to-home   # → ~/drowsiness_crops.h5

# (B) CNNs 5-fold subject-independent (comparación MobileNet vs ResNet)
python train_kfold.py --arch mobilenetv2
python train_kfold.py --arch resnet50v2

# (C) Features de parpadeo — el MEJOR modelo
python -m src.features.extract_ear_dlib                            # EAR sobre crops
python -m src.features.extract_ear_dense --workers 6 --stride 2    # EAR denso 15fps desde videos
python blink_features_dense_kfold.py                               # modelo de parpadeo, 5-fold

# (D) Ensemble CNN + parpadeo (no supera al parpadeo solo — resultado negativo documentado)
python ensemble_kfold.py --arch mobilenetv2

# Pipeline completo de una (encadena todo + regenera la notebook 05):
bash run_overnight.sh
```

Los scripts corren por terminal (el kernel de Jupyter se cuelga en WSL2). La notebook
05 se regenera con `notebooks/_build_05.py` (lee los `summary_*.json`) y se ejecuta.

---

## 4. Notebooks

| Notebook | Qué muestra |
|---|---|
| `01_extract_crops` | Distribución del dataset, EAR |
| `02_training` | CNN frame-a-frame, curvas |
| `02b_temporal_model` | Modelo temporal (overfitting) |
| `03_evaluation` | Matrices de confusión, métricas |
| `04_kfold_cross_validation` | 5-fold CV — protocolo del paper |
| **`05_recorrido_completo`** | **★ Todo el proceso paso a paso, con evidencia, para el oral** |

---

## 5. Hallazgos centrales

1. **Atajo de sesión**: cada sujeto grabó sus videos en sesiones distintas → la CNN
   aprende la sesión, no la somnolencia → no generaliza (~0.56 a sujetos nuevos).
2. **El split único engaña**: `val_f1=0.84` vs `test_f1=0.44` en el mismo modelo.
   La métrica honesta es 5-fold subject-independent a nivel de video.
3. **EAR sintético (Haar) = basura**; con dlib la señal de parpadeo es real
   (somnoliento: PERCLOS 3.4×, microsueños 5.0×).
4. **El modelo de parpadeo (libre de apariencia) es el mejor (0.692)** — supera a las
   CNN y replica el insight del paper. El ensemble NO ayuda (la CNN débil arrastra).
5. **Techo ~0.69**, al nivel del paper. 0.80 no es realista subject-independent.

---

## 6. Resultados (5-fold CV subject-independent, video-level, 60 sujetos)

| Modelo | Video acc |
|---|---|
| CNN ResNet50V2 | 0.558 ± 0.042 |
| CNN MobileNetV2 | 0.558 ± 0.082 |
| Ensemble CNN + parpadeo | 0.650 ± 0.082 |
| **★ Parpadeo denso 15fps** | **0.692 ± 0.043** |
| Paper UTA (3-clases) | ~0.65 |

---

## 7. Estructura

```
src/
├── detection/face_mesh.py        # Haar cascade (CPU)
├── features/
│   ├── ear.py                    # EAR (fórmula)
│   ├── extract_ear_dlib.py       # EAR real sobre crops
│   └── extract_ear_dense.py      # ★ EAR denso 15fps desde videos (multiprocessing)
├── data/
│   ├── extract_crops.py / optimize_h5.py
│   ├── dataset.py                # make_splits, make_kfold_splits, video_level_eval
│   └── sequence_dataset.py       # ventanas temporales
├── models/  backbone.py · temporal_model.py
└── training/ train.py · train_temporal.py · evaluate.py

train_kfold.py · blink_features_dense_kfold.py · ensemble_kfold.py · run_overnight.sh
notebooks/_build_05.py            # regenera la notebook 05 desde los JSON de resultados
```
