"""
Descarga los datos del Google Drive del grupo y los deja en la carpeta ./data/.
Funciona en local, Google Colab y cualquier instancia cloud.

Uso:
    python data_loader.py              # descarga todo
    python data_loader.py --list       # solo lista archivos sin descargar
"""

import argparse
import os
import sys

FOLDER_ID = "1Kjo6YMekC3ZSYjp2wgdoIvzetYeyGh6Y"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("Instalando gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
        return gdown


def download_from_drive(list_only: bool = False):
    gdown = _ensure_gdown()

    os.makedirs(DATA_DIR, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

    if list_only:
        files = gdown.download_folder(url, output=DATA_DIR, quiet=True, use_cookies=False, skip_download=True)
        print("Archivos en el Drive:")
        for f in (files or []):
            print(f"  {os.path.basename(f)}")
        return files

    print(f"Descargando datos en: {DATA_DIR}")
    files = gdown.download_folder(url, output=DATA_DIR, quiet=False, use_cookies=False)
    print(f"\nDescarga completa. {len(files or [])} archivo(s) en {DATA_DIR}")
    return files


# ── Helper para usar desde notebooks ──────────────────────────────────────────

def load_dataframes():
    """Descarga los datos (si no existen) y devuelve un dict {nombre: DataFrame}."""
    import pandas as pd

    files = [f for f in os.listdir(DATA_DIR) if not f.startswith(".")] if os.path.isdir(DATA_DIR) else []
    if not files:
        download_from_drive()

    dataframes = {}
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = fname.rsplit(".", 1)[-1].lower()
        try:
            if ext == "csv":
                dataframes[fname] = pd.read_csv(fpath)
            elif ext in ("xls", "xlsx"):
                dataframes[fname] = pd.read_excel(fpath)
            elif ext == "parquet":
                dataframes[fname] = pd.read_parquet(fpath)
            elif ext == "json":
                dataframes[fname] = pd.read_json(fpath)
        except Exception as e:
            print(f"No se pudo leer {fname}: {e}")

    return dataframes


# ── Google Colab: montar Drive directamente (alternativa) ─────────────────────

def mount_colab_drive(drive_path: str = "/content/drive"):
    """Solo funciona en Google Colab. Monta el Drive en /content/drive."""
    try:
        from google.colab import drive
        drive.mount(drive_path)
        print(f"Drive montado en {drive_path}")
        print(f"Tus datos están en: {drive_path}/MyDrive/  (busca la carpeta del grupo)")
    except ImportError:
        print("Esta función solo funciona en Google Colab. Usa download_from_drive() en su lugar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos del Drive del grupo")
    parser.add_argument("--list", action="store_true", help="Solo lista los archivos sin descargar")
    args = parser.parse_args()

    download_from_drive(list_only=args.list)
