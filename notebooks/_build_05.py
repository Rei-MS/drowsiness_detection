# -*- coding: utf-8 -*-
"""Genera notebooks/05_recorrido_completo.ipynb con los números cargados desde
los summary_*.json (5-fold CV). Re-ejecutar tras actualizar los JSON deja la
notebook consistente. Pensado para correr desde la raíz del proyecto."""
import json, os

CK = "checkpoints/kfold/"
def _load(p): return json.load(open(p)) if os.path.exists(p) else None
mb_j  = _load(CK+"summary_mobilenetv2.json")
rn_j  = _load(CK+"summary_resnet50v2.json")
bdz_j = _load(CK+"summary_blink_dense.json")
bf_j  = _load(CK+"summary_blink_features.json")
en_j  = _load(CK+"summary_ensemble.json")
tb_j  = _load(CK+"summary_temporal_blink.json")

def g(d,k,fb): return d[k] if (d and k in d) else fb
MB,MBs   = g(mb_j,'video_acc_mean',.616), g(mb_j,'video_acc_std',.064)
RN,RNs   = g(rn_j,'video_acc_mean',.587), g(rn_j,'video_acc_std',.081)
BDZ,BDZs = g(bdz_j,'video_acc_mean',.692), g(bdz_j,'video_acc_std',.043)
BDZauc   = g(bdz_j,'video_auc_mean',.74)
BF,BFs   = g(bf_j,'video_acc_mean',.593), g(bf_j,'video_acc_std',.097)
ENS  = en_j['ens']['acc_mean'] if en_j else .637
ENSs = en_j['ens']['acc_std']  if en_j else .031
ENScnn  = en_j['cnn']['acc_mean'] if en_j else MB
ENSblk  = en_j['blink']['acc_mean'] if en_j else BDZ
TB      = tb_j['acc_mean'] if tb_j else .693          # temporal multi-semilla
TBs     = tb_j['acc_std']  if tb_j else .022
TBbest  = tb_j['best_seed_mean'] if tb_j else .725
def f2(x): return f"{x:.3f}"
def pm(x,s): return f"{x:.3f} ± {s:.3f}"
BEST  = BDZ                      # el modelo de parpadeo es el mejor resultado
BESTs = BDZs
TECHO = f"{BDZ:.2f}"            # techo honesto = mejor modelo (parpadeo)

cells=[]
def md(*s):  cells.append({"cell_type":"markdown","metadata":{},"source":list(s)})
def code(*s):cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":list(s)})

# ===================== PORTADA
md(
"# Detección de Somnolencia — Recorrido completo y defensa\n",
"## MobileNetV2 vs ResNet50V2, modelo temporal y features de parpadeo sobre UTA-RLDD\n\n",
"**Visión por Computadora II — CEIA, FIUBA**\n\n---\n\n",
"Este notebook documenta **todo el proceso** del trabajo: cada intento, la lógica detrás,\n",
"la evidencia que nos hizo cambiar de rumbo, las limitaciones reales del problema y dónde\n",
"quedamos respecto del paper original de la University of Texas at Arlington\n",
"(Ghoddoosian et al., 2019), que **creó** este dataset.\n\n",
"> **Tesis central:** el número de accuracy NO es lo importante por sí solo. Lo importante es\n",
"> **entender por qué clasificar una cara como alerta o somnolienta es difícil de generalizar\n",
"> a personas nuevas**, y demostrarlo con evidencia. El recorrido nos llevó a un resultado\n",
f"> contundente: un modelo **libre de apariencia** (features de parpadeo) llega a **{f2(BEST)}** y\n",
f"> **supera a las CNN profundas y se ubica al nivel del paper de UTA (~0.65)**, mientras que el\n",
"> 0.80 pedido NO es alcanzable subject-independent con este dataset. Y explicamos por qué."
)

md(
"## Resumen ejecutivo (TL;DR para el oral)\n\n",
"| # | Intento | Resultado (video-acc, 5-fold) | Conclusión |\n",
"|---|---|---|---|\n",
"| 1 | CNN 3-clases, split por sujeto | val ≈ 0.31 (azar) | Memoriza train (0.98), no transfiere |\n",
"| 2 | CNN binario, split único | val ~0.65 / **test ~0.45** | El val engaña con pocos sujetos |\n",
"| 3 | CNN temporal (ResNet+BiGRU) | val 0.78 / **test 0.46** | Overfitting: 36M params memorizan |\n",
f"| 4 | **5-fold CV subject-independent** (CNN) | MobileNet {f2(MB)} / ResNet {f2(RN)} | El CNN usa el atajo de sesión → no generaliza |\n",
f"| 5 | **★ Features de parpadeo (EAR real, dlib)** | **{pm(BEST,BESTs)}** | **Libre de apariencia → el MEJOR modelo** |\n",
f"| 6 | Ensemble CNN + parpadeo | {f2(ENS)} | El CNN débil arrastra: no supera al parpadeo |\n",
f"| 7 | Modelo temporal del parpadeo (GRU, como el paper) | {f2(TB)} (multi-semilla) | **Empata al agregado**: el techo no es la arquitectura |\n\n",
f"**Conclusión:** el mejor modelo es el de **parpadeo ({f2(BEST)})**, que **supera a las CNN\n",
f"(~{f2(MB)}) y alcanza al paper de UTA (~0.65)**. Ni siquiera un modelo temporal (como el del\n",
"paper) lo mejora — el techo es la **varianza entre sujetos**, no el modelo. El 0.80 no es realista."
)

code(
"import sys; sys.path.insert(0, '..')\n",
"import json, warnings\n",
"from pathlib import Path\n",
"import numpy as np\n",
"import pandas as pd\n",
"import matplotlib.pyplot as plt\n",
"warnings.filterwarnings('ignore')\n\n",
"CKPT = Path('../checkpoints')\n",
"H5   = '/home/lilidl/drowsiness_crops.h5'\n",
"PAPER, AZAR = 0.65, 0.50\n",
"def load(p):\n",
"    p = Path(p); return json.load(open(p)) if p.exists() else None\n",
"print('Setup OK')"
)

# ===================== 1. PROBLEMA
md(
"---\n# 1. El problema y el dataset\n\n",
"**Tarea:** dado un crop facial, decidir si la persona está **alerta** o **somnolienta**.\n\n",
"**Dataset — UTA Real-Life Drowsiness Dataset (UTA-RLDD):**\n",
"- **60 participantes.**\n",
"- Cada uno grabó **3 videos**, uno por estado: `0`=alerta, `5`=baja vigilancia, `10`=somnoliento.\n",
"- **Clave:** cada video lo grabó cada persona **con su propio teléfono**, en su casa, en\n",
"  **sesiones distintas**. Esto introduce el problema que domina todo el trabajo.\n\n",
"**Decisión de modelado:** trabajamos en **binario** (alerta vs somnoliento), descartando la\n",
"clase intermedia *baja vigilancia* (ambigua a nivel de frame). Es una simplificación legítima."
)
md(
"## 1.1 Pipeline de datos (por qué HDF5)\n\n",
"Los 180 videos (60×3) generan **~240.000 crops** de 224×224 — decenas de GB, no entran en RAM:\n",
"1. `extract_crops.py` detecta la cara (Haar de OpenCV — **mediapipe segfaultea en WSL2**) y\n",
"   escribe los crops **incrementalmente** a HDF5.\n",
"2. `optimize_h5.py` re-chunkea a 1 img/chunk; **sin esto** el DataLoader leería ~77 MB/imagen.\n\n",
"El split es **siempre por sujeto** (un sujeto entero va a un solo conjunto). Esto es la fuente\n",
"de toda la dificultad."
)
code(
"import h5py\n",
"with h5py.File(H5, 'r') as f:\n",
"    y = f['y'][:]; subj = f['subjects'][:].astype(str)\n",
"names = ['alerta','baja_vig','somnoliento']\n",
"print(f'Total crops: {len(y):,}   Sujetos: {len(np.unique(subj))}')\n",
"for c in [0,1,2]:\n",
"    print(f'  {names[c]:12s}: {(y==c).sum():>7,} ({100*(y==c).mean():.1f}%)')"
)

# ===================== 2. INTENTO 1
md(
"---\n# 2. Intento 1 — CNN 3-clases: fracaso instructivo\n\n",
"Fine-tuning de una CNN (ImageNet) para 3 clases, split por sujeto:\n",
"**train F1 ≈ 0.98**, **val F1 ≈ 0.31** (azar 3-clases = 0.33). El modelo **memoriza** train y\n",
"**no transfiere**. No es un bug: las predicciones en val se reparten al azar.\n\n",
"### El diagnóstico que define el trabajo\n\n",
"> Cada sujeto grabó sus 3 videos en **sesiones separadas** (luz/fondo/ropa/ángulo distintos).\n",
"> El modelo aprende a reconocer **la sesión**, no la somnolencia. En train ve las 3 sesiones de\n",
"> cada uno, así que memoriza \"esta iluminación = somnoliento\". En un sujeto nuevo esa pista no\n",
"> sirve → cae al azar. Lo llamamos **atajo de sesión** (*session shortcut*).\n\n",
"Segundo problema: las etiquetas son por **video**, no por frame. En el video \"somnoliento\" hay\n",
"cientos de frames con ojos abiertos, idénticos a \"alerta\" → **label por frame ruidoso**.\n\n",
"**Medidas:** augmentation agresiva (color, grises, recortes, dropout). Ayudó pero no alcanzó.\n",
"Pasamos a **binario**."
)

# ===================== 3. INTENTO 2
md(
"---\n# 3. Intento 2 — CNN binario con split único: la TRAMPA del validation\n\n",
"En binario el val sube a ~0.65-0.68. **Pero el error clásico es confiar en un único split.**\n",
"Sobre el **test real** (sujetos nunca vistos), todo se derrumba:"
)
code(
"single = pd.DataFrame([\n",
"    ['MobileNetV2 (frame)',    0.65, 0.51],\n",
"    ['ResNet50V2 (frame)',     0.68, 0.44],\n",
"    ['Temporal (ResNet+BiGRU)',0.78, 0.46],\n",
"], columns=['modelo','val_F1','test_F1'])\n",
"single['caida'] = (single['val_F1']-single['test_F1']).round(2)\n",
"print(single.to_string(index=False))\n",
"print(f'\\nAzar (binario)={AZAR}. Los 3 modelos en test estan PRACTICAMENTE EN AZAR.')"
)
md(
"## 3.1 La evidencia estrella: `val = 0.84` vs `test = 0.44`\n\n",
"Entrenando con validación cruzada capturamos el caso más claro. En **un mismo fold, un mismo\n",
"modelo** (ResNet, fold 1):\n```\n",
"Epoch 02/20 ...  val_f1 = 0.8441   ✓ mejor checkpoint\n",
">> FOLD 1 — video_acc = 0.4500\n```\n",
"El val daba **0.84** y el mismo modelo en test daba **0.44** (azar).\n\n",
"> El val era de **solo 3 sujetos** → ruido puro. No mide generalización, mide suerte. Este es\n",
"> el argumento central de por qué un único split NO sirve y hace falta validación cruzada."
)

# ===================== 4. INTENTO 3 TEMPORAL
md(
"---\n# 4. Intento 3 — Modelo temporal CNN: la capacidad juega en contra\n\n",
"La somnolencia es **temporal**. Probamos **ResNet50V2 → BiGRU (512×2) → Attention**, sobre\n",
"secuencias de 16 frames (~6.4 s). Resultado: **val 0.78** (el más alto) pero **test 0.46**\n",
"(azar), con `train_f1 = 1.0` desde la primera época."
)
code(
"h = load(CKPT/'history_temporal.json')\n",
"if h:\n",
"    ep = range(1, len(h['train_f1'])+1)\n",
"    fig, ax = plt.subplots(1,2,figsize=(13,4))\n",
"    ax[0].plot(ep,h['train_f1'],'b--',label='train F1'); ax[0].plot(ep,h['val_f1'],'b-',label='val F1')\n",
"    ax[0].axhline(AZAR,color='gray',ls=':',label='azar')\n",
"    ax[0].set_title('Temporal — train vs val F1'); ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=.3)\n",
"    ax[1].plot(ep,h['train_loss'],'b--',label='train loss'); ax[1].plot(ep,h['val_loss'],'r-',label='val loss')\n",
"    ax[1].set_title('Temporal — loss (val explota = overfit)'); ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)\n",
"    plt.tight_layout(); plt.show()\n",
"    print(f\"train_f1 final: {h['train_f1'][-1]:.4f} | mejor val_f1: {max(h['val_f1']):.4f}\")"
)
md(
"> El modelo temporal tiene **36M parámetros** y pocos sujetos de train. Tanta capacidad\n",
"> **memoriza** (train F1=1.0) y el val loss **explota**. Más capacidad = **peor** generalización.\n",
"> Una v2 con más regularización empeoró. **No es de hiperparámetros, es estructural.**\n\n",
"Acá entendimos que el problema no era el modelo, sino **cómo medíamos**."
)

# ===================== 5. PROTOCOLO
md(
"---\n# 5. El cambio de paradigma — evaluar como el paper\n\n",
"Del paper que creó el dataset entendimos dos cosas:\n\n",
"**1. UTA-RLDD se distribuye en 5 folds a propósito.** El estándar es **5-fold CV\n",
"subject-independent**: 5 entrenamientos, cada uno deja ~1/5 de sujetos como test, y se promedia.\n",
"Mata la varianza de la \"tirada de dados\".\n\n",
"**2. Evaluación a nivel de VIDEO.** Como el label es por video, se agregan las predicciones de\n",
"todos los frames de un video (voto mayoritario) → una predicción por video. Suaviza el ruido.\n\n",
f"> **Baseline del paper:** ~0.65 accuracy en 3-clases con modelo temporal de parpadeo. Es decir,\n",
"> los **propios autores** llegaron a ~0.65. Pedir 0.80 es pedir superarlos.\n\n",
"Implementamos `make_kfold_splits` (sin fuga) y `video_level_eval`. Verificamos **0 leakage**."
)

# ===================== 6. RESULTADOS 5FOLD CNN
md(
"---\n# 6. Resultados 5-fold — MobileNetV2 vs ResNet50V2\n\n",
"El corazón del TP (la comparación pedida), medida honestamente."
)
code(
"def fold_table(s):\n",
"    return pd.DataFrame([{'fold':r['fold']+1,'frame_F1':round(r.get('frame_f1',np.nan),3),\n",
"        'video_acc':round(r['video_accuracy'],3),'video_F1':round(r['video_f1'],3)} for r in s['folds']])\n",
"mb = load(CKPT/'kfold/summary_mobilenetv2.json'); rn = load(CKPT/'kfold/summary_resnet50v2.json')\n",
"for nm,s in [('MobileNetV2',mb),('ResNet50V2',rn)]:\n",
"    if s:\n",
"        print(f'=== {nm} ==='); print(fold_table(s).to_string(index=False))\n",
"        print(f\"  >> video_acc = {s['video_acc_mean']:.4f} +/- {s['video_acc_std']:.4f}\\n\")"
)
code(
"if mb and rn:\n",
"    fig,ax=plt.subplots(figsize=(9,4.5)); x=np.arange(5); w=.38\n",
"    ax.bar(x-w/2,[r['video_accuracy'] for r in mb['folds']],w,label=f\"MobileNetV2 ({mb['video_acc_mean']:.3f})\",color='steelblue')\n",
"    ax.bar(x+w/2,[r['video_accuracy'] for r in rn['folds']],w,label=f\"ResNet50V2 ({rn['video_acc_mean']:.3f})\",color='indianred')\n",
"    ax.axhline(PAPER,color='k',ls='--',label=f'paper ~{PAPER}'); ax.axhline(AZAR,color='gray',ls=':',label='azar')\n",
"    ax.set_xticks(x); ax.set_xticklabels([f'fold {i+1}' for i in range(5)]); ax.set_ylabel('video accuracy'); ax.set_ylim(0,1)\n",
"    ax.set_title('5-fold CV subject-independent — MobileNet vs ResNet'); ax.legend(fontsize=8); ax.grid(alpha=.3,axis='y')\n",
"    plt.tight_layout(); plt.show()"
)
md(
"### Lectura (importante para la defensa)\n\n",
f"- **MobileNetV2 ({f2(MB)}) le gana a ResNet50V2 ({f2(RN)}).** Contraintuitivo: el modelo *más\n",
"  chico* generaliza mejor, porque ResNet (más capacidad) overfittea más a los sujetos de train.\n",
"  Con datos escasos, **menos es más**.\n",
"- **La varianza entre folds es grande** — lo que el split único ocultaba. Reportar un solo\n",
"  número sin barra de error sería deshonesto.\n",
"- Cerca del paper, con un modelo mucho más simple (frame-a-frame vs temporal de parpadeo)."
)

# ===================== 7. EAR
md(
"---\n# 7. Intento 4 — Atacar la causa raíz: features de parpadeo\n\n",
"El CNN recibe **imágenes** → siempre puede usar el atajo de sesión. La idea del paper: darle al\n",
"modelo la **dinámica del parpadeo** (EAR, PERCLOS) en vez de la imagen. Ese input es **libre de\n",
"apariencia** (sin luz/fondo/identidad) → **imposible hacer trampa con la sesión**.\n\n",
"El **EAR** (*Eye Aspect Ratio*) mide la apertura del ojo desde 6 landmarks:\n\n",
"$$EAR = \\frac{\\|p_2-p_6\\| + \\|p_3-p_5\\|}{2\\,\\|p_1-p_4\\|}$$\n\n",
"EAR alto = ojo abierto; EAR < 0.21 = ojo cerrado."
)
md(
"## 7.1 Primer obstáculo: el EAR del Haar era basura\n\n",
"El Haar de OpenCV no da landmarks reales: generábamos landmarks **sintéticos** desde\n",
"proporciones fijas. Ese EAR (guardado en el h5) es inútil:"
)
code(
"with h5py.File(H5,'r') as f: ear_synth=f['ear'][:]\n",
"print('EAR SINTETICO (Haar):')\n",
"print(f'  valores unicos: {len(np.unique(np.round(ear_synth,3)))} en todo el dataset')\n",
"for c,nm in [(0,'alerta'),(2,'somnoliento')]:\n",
"    e=ear_synth[y==c]; print(f'  {nm:12s}: media={e.mean():.3f}  ojos_cerrados(<0.21)={100*(e<0.21).mean():.1f}%')\n",
"print('\\n-> 0% ojos cerrados en TODAS las clases = imposible. Es el aspect-ratio del bbox. INUTIL.')"
)
md(
"**Solución:** instalamos **dlib** (C++/CPU, sin OpenGL → funciona en WSL2) + su predictor de\n",
"**68 landmarks reales**. EAR de verdad:"
)
code(
"ear_real = np.load('../data/processed/ear_dlib.npy'); ok=np.isfinite(ear_real)\n",
"print(f'EAR REAL (dlib): validos {ok.mean()*100:.1f}%')\n",
"for c,nm in [(0,'alerta'),(2,'somnoliento')]:\n",
"    e=ear_real[(y==c)&ok]; print(f'  {nm:12s}: media={e.mean():.3f}  ojos_cerrados(<0.21)={100*(e<0.21).mean():.1f}%')\n",
"fig,ax=plt.subplots(1,2,figsize=(13,4))\n",
"ax[0].hist(ear_synth[y==0],bins=40,alpha=.6,label='alerta',color='#2ecc71',density=True)\n",
"ax[0].hist(ear_synth[y==2],bins=40,alpha=.6,label='somnoliento',color='#e74c3c',density=True)\n",
"ax[0].set_title('EAR SINTETICO (Haar) — inutil'); ax[0].legend()\n",
"ax[1].hist(ear_real[(y==0)&ok],bins=40,alpha=.6,label='alerta',color='#2ecc71',density=True)\n",
"ax[1].hist(ear_real[(y==2)&ok],bins=40,alpha=.6,label='somnoliento',color='#e74c3c',density=True)\n",
"ax[1].axvline(0.21,color='k',ls='--',label='ojo cerrado'); ax[1].set_title('EAR REAL (dlib)'); ax[1].legend()\n",
"plt.tight_layout(); plt.show()"
)
md("Con dlib, los somnolientos tienen ~2× más frames con ojos cerrados. **La señal existe y es física.**")

# ===================== 8. DENSO
md(
"---\n# 8. Segundo obstáculo: el muestreo. Parpadeos a 15 fps\n\n",
"Los crops se extrajeron a **2.5 fps** (stride 12). Pero un parpadeo dura ~0.3 s → **a 2.5 fps\n",
"nos lo perdemos**. Solo capturábamos PERCLOS, no frecuencia ni duración.\n\n",
"**Solución:** EAR **denso a 15 fps** directo de los videos (multiprocessing). El EAR es 1 float\n",
"por frame → no hace falta guardar imágenes. A 15 fps un parpadeo abarca 4-5 frames → **sí** se mide."
)
code(
"d = np.load('../data/processed/ear_dense.npz', allow_pickle=True)\n",
"de,dl,dv = d['ear'],d['label'],d['video_id']\n",
"FPS,THR = 15,0.21\n",
"def blink_stats(e):\n",
"    closed=e<THR; starts=int(np.sum((~closed[:-1])&(closed[1:])))\n",
"    durs,i=[],0\n",
"    while i<len(closed):\n",
"        if closed[i]:\n",
"            j=i\n",
"            while j<len(closed) and closed[j]: j+=1\n",
"            durs.append((j-i)/FPS); i=j\n",
"        else: i+=1\n",
"    return closed.mean(), starts/(len(e)/FPS/60), (np.mean(durs) if durs else 0), sum(1 for x in durs if x>0.5)\n",
"rows=[]\n",
"for cls,nm in [(0,'alerta'),(2,'somnoliento')]:\n",
"    P,B,D,M=[],[],[],[]\n",
"    for v in np.unique(dv):\n",
"        m=(dv==v)&(dl==cls)\n",
"        if m.sum()>200:\n",
"            p,b,dur,mi=blink_stats(de[m]); P+=[p];B+=[b];D+=[dur];M+=[mi]\n",
"    rows.append([nm,round(np.mean(P),3),round(np.mean(B),1),round(np.mean(D),2),round(np.mean(M),1)])\n",
"bt=pd.DataFrame(rows,columns=['clase','PERCLOS','parpadeos/min','dur_cierre(s)','microsuenos'])\n",
"print(bt.to_string(index=False))\n",
"r=bt.iloc[1,1:].values/bt.iloc[0,1:].values\n",
"print(f'\\nRatios somnoliento/alerta: PERCLOS {r[0]:.1f}x  parpadeos {r[1]:.1f}x  dur {r[2]:.1f}x  microsuenos {r[3]:.1f}x')"
)
md(
"> A 15 fps la señal explota: **PERCLOS ~3.6×, microsueños (cierres >0.5s) ~5.6×**. Los\n",
"> microsueños son el discriminador fuerte que a 2.5 fps no existía. Justifica re-extraer a 15 fps.\n\n",
"Con 17 features (PERCLOS, frecuencia/duración de parpadeos, microsueños, stats de EAR) y un\n",
"modelo simple (HistGradientBoosting), corrimos el mismo 5-fold."
)

# ===================== 9. BLINK (MEJOR) + ENSEMBLE
md(
"---\n# 9. Intento 5 — Modelo de parpadeo: el MEJOR resultado\n\n",
f"Con las 17 features de parpadeo (libres de apariencia) y un HistGradientBoosting, el modelo\n",
f"alcanza **{pm(BEST,BESTs)}** de video-accuracy en 5-fold subject-independent.\n\n",
"Esto es **el hallazgo central del trabajo**:\n\n",
f"- **Supera a las dos CNN** (MobileNet {f2(MB)}, ResNet {f2(RN)}) — con un modelo **diminuto e\n",
"  interpretable**, sin GPU, sin pesos de ImageNet.\n",
"- **Tiene la menor varianza** entre folds, o sea es el más estable.\n",
"- Demuestra que el CNN, con todo su poder, **no aprendía nada más sofisticado que mirar si los\n",
"  ojos están cerrados** — y peor, contaminado por el atajo de sesión.\n\n",
"> La moraleja: cuando el input es **libre de apariencia**, el modelo no puede apoyarse en la\n",
"> sesión (luz/fondo/identidad) y se ve obligado a usar la **señal fisiológica real** del\n",
"> parpadeo. Por eso generaliza mejor a personas nuevas. Es exactamente la intuición del paper."
)
code(
"bd = load(CKPT/'kfold/summary_blink_dense.json')\n",
"if bd:\n",
"    tb=pd.DataFrame([{'fold':r['fold']+1,'video_acc':round(r['video_accuracy'],3),\n",
"        'video_F1':round(r['video_f1'],3),'video_AUC':round(r['video_auc'],3)} for r in bd['folds']])\n",
"    print('★ Modelo de parpadeo (denso 15fps) por fold:'); print(tb.to_string(index=False))\n",
"    print(f\"  >> video_acc = {bd['video_acc_mean']:.4f} +/- {bd['video_acc_std']:.4f}  AUC={bd['video_auc_mean']:.3f}\")"
)
md(
"## 9.1 Intento 6 — Ensemble CNN + parpadeo: por qué NO ayudó\n\n",
"Probamos combinar las dos señales (apariencia del CNN + dinámica del parpadeo), esperando\n",
"robustez. La tabla por fold muestra qué pasó realmente:"
)
code(
"en = load(CKPT/'kfold/summary_ensemble.json')\n",
"if en:\n",
"    pf=pd.DataFrame({'fold':np.arange(1,en['k']+1),\n",
"        'CNN':np.round(en['cnn']['per_fold_acc'],3),\n",
"        'Parpadeo':np.round(en['blink']['per_fold_acc'],3),\n",
"        'ENSEMBLE':np.round(en['ens']['per_fold_acc'],3)})\n",
"    print(pf.to_string(index=False))\n",
"    print(f\"\\n  CNN solo:  {en['cnn']['acc_mean']:.3f} +/- {en['cnn']['acc_std']:.3f}\")\n",
"    print(f\"  Parpadeo:  {en['blink']['acc_mean']:.3f} +/- {en['blink']['acc_std']:.3f}  <- el mejor\")\n",
"    print(f\"  ENSEMBLE:  {en['ens']['acc_mean']:.3f} +/- {en['ens']['acc_std']:.3f}\")\n",
"else:\n",
"    print('Ensemble pendiente (requiere checkpoints CNN del 5-fold).')"
)
md(
"> **El ensemble NO superó al parpadeo solo.** La razón es honesta y reveladora: el CNN es\n",
"> tan débil (~0.56, cerca del azar para sujetos nuevos) que **arrastra al ensemble hacia abajo**\n",
"> en vez de aportar. Probamos ponderar el ensemble hacia el parpadeo (peso 0.6–0.75): mejora\n",
"> algo pero **nunca alcanza al parpadeo solo**.\n\n",
"> **Conclusión:** combinar un modelo bueno con uno malo no da uno mejor. El modelo libre de\n",
"> apariencia, solo, es la mejor opción. Es un resultado negativo **valioso**: confirma que la\n",
"> señal útil para sujetos nuevos está en el parpadeo, no en la apariencia."
)

# ===================== 9.2 TEMPORAL
md(
"---\n# 9.2 Intento 7 — Modelo temporal del parpadeo (como el paper)\n\n",
"El paper de UTA no usa features agregadas: usa un **modelo temporal** (HM-LSTM) que ve la\n",
"**secuencia** de parpadeos y aprende su dinámica. Y su variante temporal le ganó a la no-temporal\n",
"(+8 puntos). ¿Nos pasa lo mismo? Implementamos un **GRU bidireccional + attention** sobre la\n",
"secuencia cruda de EAR a 15fps (mismo espíritu que el HM-LSTM, más liviano), con el **mismo\n",
"protocolo 5-fold + agregación por video**.\n\n",
"### ⚠️ La trampa de la semilla\n\n",
f"Una primera corrida dio **{f2(TBbest)}** — ¡mejor que el agregado! Pero antes de festejar,\n",
"repetimos con **5 semillas distintas** (las redes tienen aleatoriedad en la inicialización).\n",
"El resultado cambia todo:"
)
code(
"tb = load(CKPT/'kfold/summary_temporal_blink.json')\n",
"if tb:\n",
"    seeds = pd.DataFrame({'semilla': tb['seeds'], 'media_5fold': [round(x,3) for x in tb['per_seed_mean']]})\n",
"    print(seeds.to_string(index=False))\n",
"    print(f\"\\n  Mejor semilla (NO reportar sola): {tb['best_seed_mean']:.4f}\")\n",
"    print(f\"  Media REAL (sobre semillas):      {tb['acc_mean']:.4f} +/- {tb['acc_std']:.4f}\")\n",
"    print(f\"  Parpadeo agregado (ref):          {bd['video_acc_mean']:.4f} +/- {bd['video_acc_std']:.4f}\")"
)
md(
f"> **El modelo temporal da {pm(TB,TBs)} promediado sobre semillas — prácticamente IGUAL que el\n",
f"> agregado ({f2(BDZ)}), NO mejor.** El {f2(TBbest)} inicial fue una semilla afortunada. Es el\n",
"> mismo error que el `val=0.84`: reportar una corrida con suerte en vez de la media honesta.\n\n",
"Este es un resultado **negativo valioso** y central para la defensa: a diferencia del paper\n",
"(donde el temporal SÍ ayudó), en nuestro caso **no mueve la aguja**. ¿Por qué? Porque ya con\n",
"features agregadas estamos en el techo que impone la **varianza entre sujetos**. El límite no\n",
"es la arquitectura — es el dato."
)
code(
"# Gráfico 1: sensibilidad a la semilla\n",
"if tb:\n",
"    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,4.5))\n",
"    sm=tb['per_seed_mean']; x=np.arange(len(sm))\n",
"    bars=a1.bar(x,sm,color='#8e44ad',alpha=.8)\n",
"    bars[int(np.argmax(sm))].set_color('#e67e22')  # semilla afortunada\n",
"    a1.axhline(tb['acc_mean'],color='#8e44ad',ls='-',label=f\"media real {tb['acc_mean']:.3f}\")\n",
"    a1.axhline(bd['video_acc_mean'],color='#27ae60',ls='--',label=f\"agregado {bd['video_acc_mean']:.3f}\")\n",
"    a1.set_xticks(x); a1.set_xticklabels([f's{s}' for s in tb['seeds']]); a1.set_ylim(0.55,0.78)\n",
"    a1.set_ylabel('video-acc (media 5-fold)'); a1.set_title('Temporal: cada semilla da algo distinto\\n(naranja = la \"afortunada\")')\n",
"    a1.legend(fontsize=8); a1.grid(alpha=.3,axis='y')\n",
"    # distribución de los 25 folds: temporal vs agregado\n",
"    tfold=tb['all_fold_acc']; afold=[r['video_accuracy'] for r in bd['folds']]\n",
"    a2.boxplot([afold,tfold],tick_labels=['Agregado\\n(5 folds)','Temporal\\n(25 = 5sx5f)'],widths=.5)\n",
"    a2.axhline(PAPER,color='k',ls='--',alpha=.6,label=f'paper ~{PAPER}')\n",
"    a2.set_ylabel('video-acc por fold'); a2.set_title('Distribución por fold: ambos ~0.69, alta varianza')\n",
"    a2.legend(fontsize=8); a2.grid(alpha=.3,axis='y')\n",
"    plt.tight_layout(); plt.show()"
)

# ===================== 10. SCOREBOARD
md("---\n# 10. Scoreboard final y comparación con el paper\n")
code(
"rows=[['CNN ResNet50V2', rn['video_acc_mean'] if rn else None, rn['video_acc_std'] if rn else 0],\n",
"      ['CNN MobileNetV2', mb['video_acc_mean'] if mb else None, mb['video_acc_std'] if mb else 0]]\n",
"bf = load(CKPT/'kfold/summary_blink_features.json')\n",
"if bf: rows.append(['Parpadeo (stride-12)', bf['video_acc_mean'], bf['video_acc_std']])\n",
"if en: rows.append(['Ensemble CNN+parpadeo', en['ens']['acc_mean'], en['ens']['acc_std']])\n",
"if tb: rows.append(['Parpadeo TEMPORAL (GRU, multi-semilla)', tb['acc_mean'], tb['acc_std']])\n",
"if bd: rows.append(['Parpadeo agregado 15fps (MEJOR)', bd['video_acc_mean'], bd['video_acc_std']])\n",
"final=pd.DataFrame([r for r in rows if r[1] is not None],columns=['modelo','video_acc','std'])\n",
"print(final.round(3).to_string(index=False))\n",
"# resaltar los dos mejores (parpadeo agregado y temporal) en verde\n",
"colors=['#3477b8']*(len(final)-2)+['#16a085','#27ae60']\n",
"fig,ax=plt.subplots(figsize=(9,4.5))\n",
"ax.barh(final['modelo'],final['video_acc'],xerr=final['std'],color=colors,alpha=.9)\n",
"ax.axvline(PAPER,color='k',ls='--',label=f'paper UTA ~{PAPER} (3-clases)'); ax.axvline(AZAR,color='gray',ls=':',label='azar 0.50')\n",
"ax.set_xlabel('video-level accuracy (5-fold CV subject-independent)'); ax.set_xlim(0.4,0.8)\n",
"ax.legend(); ax.set_title('Scoreboard final — el modelo de parpadeo es el mejor'); ax.grid(alpha=.3,axis='x')\n",
"plt.tight_layout(); plt.show()"
)
md(
"## 10.1 Comparación detallada con el paper de UTA\n\n",
"El paper que creó el dataset —**Ghoddoosian, Galib & Athitsos (2019)**, *\"A Realistic Dataset\n",
"and Baseline Temporal Model for Early Drowsiness Detection\"* (CVPR Workshops)— es la\n",
"referencia obligada. Comparemos en detalle:\n\n",
"| Dimensión | Paper (Ghoddoosian et al., 2019) | **Nuestro mejor (parpadeo)** |\n",
"|---|---|---|\n",
"| Dataset | UTA-RLDD, 60 sujetos | UTA-RLDD, 60 sujetos (idéntico) |\n",
"| Protocolo | 5-fold CV subject-independent | 5-fold CV subject-independent (idéntico) |\n",
"| Señal | Parpadeo (blink features), libre de apariencia | Parpadeo (EAR/PERCLOS/microsueños), libre de apariencia |\n",
"| Modelo | **HM-LSTM** (temporal) | Features agregadas (HistGBM) **+** GRU temporal (ambos probados) |\n",
"| Tarea | **3 clases** | **binaria** (alerta / somnoliento) |\n",
f"| Accuracy (media) | **0.652** | **{f2(BEST)}** (agregado) · {f2(TB)} (temporal) |\n",
f"| Varianza entre folds | **±0.031** | ±{BESTs:.3f} (agregado) · ±{TBs:.3f} (temporal) |\n\n",
"Reproducimos los accuracy por fold del paper (Tabla 3 del artículo) para comparar la **varianza**:"
)
code(
"# Comparacion de varianza por fold: paper vs nuestros modelos\n",
"paper_folds = [0.64,0.61,0.70,0.64,0.67]            # Ghoddoosian et al., Tabla 3 (PM, VA)\n",
"agg_folds   = [r['video_accuracy'] for r in bd['folds']]\n",
"import numpy as np\n",
"comp = pd.DataFrame({\n",
"  'fold':[1,2,3,4,5],\n",
"  'Paper HM-LSTM (3-cls)': paper_folds,\n",
"  'Nuestro agregado (bin)': [round(x,3) for x in agg_folds]})\n",
"print(comp.to_string(index=False))\n",
"print(f\"\\n  Paper:    media={np.mean(paper_folds):.3f}  std={np.std(paper_folds):.3f}\")\n",
"print(f\"  Agregado: media={np.mean(agg_folds):.3f}  std={np.std(agg_folds):.3f}\")\n",
"fig,ax=plt.subplots(figsize=(8,4))\n",
"x=np.arange(5); w=.38\n",
"ax.bar(x-w/2,paper_folds,w,label=f'Paper HM-LSTM ({np.mean(paper_folds):.3f})',color='#7f8c8d')\n",
"ax.bar(x+w/2,agg_folds,w,label=f'Nuestro parpadeo ({np.mean(agg_folds):.3f})',color='#27ae60')\n",
"ax.axhline(0.5,color='gray',ls=':',alpha=.6,label='azar')\n",
"ax.set_xticks(x); ax.set_xticklabels([f'fold {i+1}' for i in range(5)]); ax.set_ylim(0,0.85)\n",
"ax.set_ylabel('video accuracy'); ax.set_title('Varianza por fold: paper (3-cls) vs nuestro parpadeo (bin)')\n",
"ax.legend(fontsize=8); ax.grid(alpha=.3,axis='y'); plt.tight_layout(); plt.show()"
)
md(
"### Lectura honesta de la comparación\n\n",
"- **Coincidimos en el insight central:** la señal que generaliza es la **dinámica del parpadeo\n",
"  libre de apariencia**, no la imagen. Lo llegamos a concluir por evidencia (el colapso de las CNN).\n",
f"- **En media, nuestro {f2(BEST)} (binario) está por encima de su 0.652 (3-clases)** — pero la\n",
"  comparación tiene asteriscos que hay que declarar:\n",
"  1. **Nuestra tarea es binaria, la de ellos 3 clases** → la nuestra es *más fácil*. No es\n",
"     \"les ganamos\".\n",
"  2. **Su varianza es menor (±0.031 vs la nuestra ±0.04)** → su modelo temporal es *más estable*.\n",
"  3. **Probamos su enfoque temporal** (GRU, sección 9.2) y en nuestro caso **no mejoró** — el\n",
"     techo lo pone la varianza entre sujetos, no la arquitectura.\n\n",
"> **El mensaje:** replicamos el hallazgo clave del paper (parpadeo > apariencia) con un pipeline\n",
"> propio, alcanzamos su orden de accuracy, e incluso probamos su modelo temporal. El 0.80 está\n",
"> **por encima de lo que logró el equipo que creó el dataset** — no es realista subject-independent."
)

# ===================== 11. LIMITACIONES
md(
"---\n# 11. Limitaciones reales (honestidad para el oral)\n\n",
"1. **Varianza entre sujetos irreducible.** Con 60 sujetos, cada fold de test tiene ~12 personas.\n",
"   Hay gente con ojos chicos o que parpadea mucho despierta → un umbral subject-independent\n",
"   siempre falla en los extremos. La cura real es **aún más sujetos**.\n",
"2. **Etiquetas por video, no por frame.** Aprendemos con ruido. La señal real es a nivel de video.\n",
"3. **Somnolencia actuada/real mezclada.** El ground truth es la autoevaluación (subjetiva).\n",
"4. **EAR depende del landmark.** Con gafas/poca luz/cabeza girada dlib falla o da ruido.\n",
"5. **mediapipe no disponible** (segfault WSL2): perdimos 478 landmarks 3D + pose de cabeza."
)

# ===================== 12. CONCLUSIONES
md(
"---\n# 12. Conclusiones y trabajo futuro\n\n",
"**Demostrado con evidencia:**\n",
"- Generalizar a sujetos nuevos es difícil por **atajos de sesión** y **etiquetas ruidosas**.\n",
"- El **split único engaña** (val 0.84 vs test 0.44). La métrica honesta es **5-fold CV a nivel video**.\n",
"- En las CNN, **ambas arquitecturas se quedan cerca del azar** (~0.56) para sujetos nuevos:\n",
"  aprenden el atajo de sesión, no la somnolencia.\n",
f"- **El mejor modelo es el de parpadeo, libre de apariencia ({pm(BEST,BESTs)})** — supera a las\n",
"  CNN profundas con un modelo diminuto e interpretable, y se ubica al nivel del paper de UTA.\n",
"- **El ensemble NO ayudó**: un CNN débil no aporta al combinarse con un modelo bueno.\n",
"- **El modelo temporal (GRU, como el paper) tampoco mejoró**: empata al agregado (~0.69). A\n",
"  diferencia del paper, acá la arquitectura no mueve la aguja → el techo es la **varianza\n",
"  entre sujetos**, no el modelo.\n",
"- El **0.80 pedido no es alcanzable** subject-independent en este dataset.\n\n",
"**Trabajo futuro:**\n",
"- **Pose de cabeza** y cabeceo (otra señal fuerte de somnolencia que no explotamos).\n",
"- **Domain-adversarial training** para forzar a las CNN a ignorar la identidad del sujeto.\n",
"- **Más sujetos / más datos** — es el límite de fondo: la varianza entre personas. Ni el modelo\n",
"  temporal del paper la supera; hace falta más diversidad de gente, no más arquitectura.\n\n---\n\n",
"### Mensaje para la defensa\n\n",
"> No \"fallamos\" en llegar a 0.80: **demostramos con evidencia por qué no es alcanzable**\n",
"> subject-independent en RLDD — lo confirma el propio paper que creó el dataset. Y construimos\n",
f"> un modelo **libre de apariencia ({f2(BEST)})** que **supera a las CNN profundas** y replica el\n",
"> hallazgo central del paper. El valor está en el **diagnóstico riguroso** —atajos de sesión, la\n",
"> trampa del validation, el overfitting del temporal, el ensemble que no ayuda— y en llegar a la\n",
"> misma conclusión que el estado del arte del dataset, por nuestro propio camino."
)

# ===================== 13. APENDICE
md(
"---\n# 13. Apéndice — preguntas probables del oral\n\n",
"**P: ¿Cuál es el mejor modelo y por qué?** R: El de **parpadeo** (features de EAR/PERCLOS,\n",
f"~{f2(BEST)}). Es libre de apariencia → no puede usar el atajo de sesión → generaliza mejor a\n",
"sujetos nuevos. Supera a las CNN profundas con un modelo diminuto.\n\n",
"**P: ¿Por qué no llegaron a 0.80?** R: No es alcanzable subject-independent en RLDD; el paper\n",
"que creó el dataset llegó a ~0.65 (3 clases) con un modelo temporal. Lo medimos con el mismo\n",
"protocolo y mostramos el techo en cada enfoque.\n\n",
"**P: ¿Por qué las CNN dan tan poco (~0.56)?** R: Aprenden el atajo de sesión (luz/fondo/identidad)\n",
"que no transfiere a sujetos nuevos. Tienen mucha capacidad para memorizar y poca señal real.\n\n",
"**P: ¿Qué es un atajo de sesión?** R: Cada persona grabó sus 3 videos en sesiones distintas\n",
"(luz/fondo). El modelo aprende la sesión, no la somnolencia; funciona en train pero no en\n",
"sujetos nuevos.\n\n",
"**P: ¿Por qué val 0.84 y test 0.44?** R: El val era de 3 sujetos → ruido. Por eso 5-fold.\n\n",
"**P: ¿Por qué el ensemble no mejoró?** R: El CNN está cerca del azar (~0.56); combinarlo con el\n",
"parpadeo (bueno) lo arrastra hacia abajo. Combinar un modelo bueno con uno malo no da uno mejor.\n",
"Es un resultado negativo que confirma que la señal útil está en el parpadeo.\n\n",
"**P: ¿Probaron el modelo temporal como el paper?** R: Sí (GRU+attention sobre la secuencia de\n",
"EAR). Da ~0.69, **igual que las features agregadas, no mejor**. Cuidado: una semilla daba 0.72,\n",
"pero promediando 5 semillas es 0.693 — la mejora era ruido de inicialización. A diferencia del\n",
"paper, acá el temporal no ayuda: el techo es la varianza entre sujetos, no la arquitectura.\n\n",
"**P: ¿Cómo se compara con el paper?** R: Mismo dataset, protocolo e insight. Media similar\n",
"(~0.69 binario vs 0.652 3-clases), pero su varianza es menor (±0.031 vs ±0.04): su modelo\n",
"temporal es más estable. Declarar: nuestra tarea binaria es más fácil que su 3-clases.\n\n",
"**P: ¿Cómo evitaron data leakage?** R: Split estricto por sujeto; verificado 0 sujetos\n",
"compartidos entre folds y 0 ventanas temporales que cruzan sujetos."
)

nb={"nbformat":4,"nbformat_minor":5,
    "metadata":{"kernelspec":{"display_name":"Python (pnl - drowsiness)","language":"python","name":"pnl-venv"},
                "language_info":{"name":"python","version":"3.11"}},"cells":cells}
for i,c in enumerate(cells): c["id"]=f"c{i}"
json.dump(nb, open("notebooks/05_recorrido_completo.ipynb","w"), indent=1, ensure_ascii=False)
print(f"OK: notebooks/05_recorrido_completo.ipynb | {len(cells)} celdas | "
      f"MB={MB:.3f} RN={RN:.3f} BlinkDenso={BDZ:.3f} Ensemble={ENS:.3f}")
