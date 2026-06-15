#!/usr/bin/env bash
# Pipeline final completo (60 sujetos). Deja la notebook 05 finalizada con outputs.
# Uso: nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
set -u
cd /home/lilidl/tp_final_cv2
PY=/home/lilidl/pnl/.venv/bin/python
mkdir -p logs checkpoints/kfold
rm -f logs/OVERNIGHT_DONE.txt logs/OVERNIGHT_FAILED.txt

step () { echo ""; echo "############################################################"; echo "# $(date +%H:%M)  $1"; echo "############################################################"; }

# Limpiar procesos de parpadeo viejos que puedan estar corriendo
pkill -f blink_features_kfold 2>/dev/null; sleep 2

step "1/6  Parpadeo denso (15fps) 5-fold"
$PY blink_features_dense_kfold.py            || echo "WARN blink_dense"

step "2/6  Parpadeo stride-12 5-fold"
$PY blink_features_kfold.py                  || echo "WARN blink_s12"

step "3/6  5-fold MobileNetV2 (~4h)"
$PY train_kfold.py --arch mobilenetv2        || { echo "FAIL mobilenet" > logs/OVERNIGHT_FAILED.txt; }

step "4/6  5-fold ResNet50V2 (~4h)"
$PY train_kfold.py --arch resnet50v2         || { echo "FAIL resnet" >> logs/OVERNIGHT_FAILED.txt; }

step "5/6  Ensemble CNN + parpadeo"
$PY ensemble_kfold.py --arch mobilenetv2     || echo "WARN ensemble"

step "6/6  Regenerar y ejecutar notebook 05"
$PY notebooks/_build_05.py
cd notebooks
$PY -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 05_recorrido_completo.ipynb
cd ..

# Resumen final
{
  echo "PIPELINE COMPLETO — $(date)"
  echo ""
  for f in mobilenetv2 resnet50v2 blink_dense blink_features; do
    $PY -c "import json,os;p='checkpoints/kfold/summary_$f.json';d=json.load(open(p)) if os.path.exists(p) else None;print(f'  $f: '+(f\"{d['video_acc_mean']:.4f} +/- {d['video_acc_std']:.4f}\" if d else 'NO'))" 2>/dev/null
  done
  $PY -c "import json,os;p='checkpoints/kfold/summary_ensemble.json';d=json.load(open(p)) if os.path.exists(p) else None;print(f'  ENSEMBLE: '+(f\"{d['ens']['acc_mean']:.4f} +/- {d['ens']['acc_std']:.4f}\" if d else 'NO'))" 2>/dev/null
} > logs/OVERNIGHT_DONE.txt
cat logs/OVERNIGHT_DONE.txt
echo ""
echo "=== TODO LISTO. Notebook: notebooks/05_recorrido_completo.ipynb ==="
