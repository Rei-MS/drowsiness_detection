"""
Re-chunk de crops.h5 para acceso aleatorio eficiente.

El extract_crops escribe en chunks grandes (512 imágenes) — óptimo para escritura
secuencial pero pésimo para el acceso aleatorio del DataLoader con shuffle=True
(lee 77MB por imagen). Este script reescribe el archivo con chunks de 1 imagen.

Opcionalmente copia el resultado al home de Linux (~) en lugar de /mnt/c, mucho
más rápido para las miles de lecturas chicas del entrenamiento.

Uso:
    python -m src.data.optimize_h5 --input data/processed/crops.h5
    python -m src.data.optimize_h5 --input data/processed/crops.h5 --to-home
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

BATCH = 512  # lectura secuencial por bloques (rápida en el archivo origen)


def optimize(input_path: Path, output_path: Path) -> None:
    with h5py.File(input_path, "r") as fin:
        n, h, w, c = fin["X"].shape
        print(f"Origen: {n} crops {(h, w, c)}")

        with h5py.File(output_path, "w") as fout:
            ds_X = fout.create_dataset(
                "X", shape=(n, h, w, c), dtype="uint8",
                chunks=(1, h, w, c),          # 1 imagen por chunk → acceso aleatorio óptimo
                compression="gzip", compression_opts=4,
            )
            # copiar arrays chicos de una
            fout.create_dataset("y",        data=fin["y"][:])
            fout.create_dataset("ear",      data=fin["ear"][:])
            fout.create_dataset("subjects", data=fin["subjects"][:],
                                dtype=h5py.string_dtype())

            # copiar X secuencialmente por bloques
            for i in range(0, n, BATCH):
                j = min(i + BATCH, n)
                ds_X[i:j] = fin["X"][i:j]
                print(f"\r  {j}/{n}", end="", flush=True)
            print()

    size_mb = output_path.stat().st_size / 1e6
    print(f"Guardado: {output_path}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/crops.h5")
    parser.add_argument("--output", default=None,
                        help="default: <input> con sufijo _opt, o ~/crops.h5 si --to-home")
    parser.add_argument("--to-home", action="store_true",
                        help="escribir en ~/drowsiness_crops.h5 (disco Linux, más rápido)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    elif args.to_home:
        output_path = Path.home() / "drowsiness_crops.h5"
    else:
        output_path = input_path.with_name(input_path.stem + "_opt.h5")

    optimize(input_path, output_path)
    print(f"\nApuntá la variable H5 de las notebooks a:\n  {output_path}")
