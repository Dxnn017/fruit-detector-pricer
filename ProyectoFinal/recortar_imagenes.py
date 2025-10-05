import os
import cv2
import numpy as np
from glob import glob

# Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset_Segmentado')
MASKS_DIR = os.path.join(BASE_DIR, 'Morfologia', 'masks_clean')
RECORTES_DIR = os.path.join(BASE_DIR, 'recortes')

# Crear carpetas de salida
def crear_estructura_recortes():
    for split in ['train', 'test']:
        split_path = os.path.join(RECORTES_DIR, split)
        os.makedirs(split_path, exist_ok=True)
        clases = os.listdir(os.path.join(DATASET_DIR, split))
        for clase in clases:
            os.makedirs(os.path.join(split_path, clase), exist_ok=True)

# Encontrar bounding box de la máscara
def bounding_box(mask):
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

# Procesar imágenes
def procesar_split(split):
    img_dir = os.path.join(DATASET_DIR, split)
    mask_dir = os.path.join(MASKS_DIR, split)
    out_dir = os.path.join(RECORTES_DIR, split)
    clases = os.listdir(img_dir)
    for clase in clases:
        img_clase_dir = os.path.join(img_dir, clase)
        mask_clase_dir = os.path.join(mask_dir, clase)
        out_clase_dir = os.path.join(out_dir, clase)
        imgs = glob(os.path.join(img_clase_dir, '*.jpg'))
        for img_path in imgs:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_clase_dir, img_name)
            if not os.path.exists(mask_path):
                print(f"No existe máscara para {img_path}")
                continue
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            x, y, w, h = bounding_box(mask_bin)
            img_crop = img[y:y+h, x:x+w]
            mask_crop = mask_bin[y:y+h, x:x+w]
            # Aplicar máscara: fondo negro
            img_crop[mask_crop == 0] = 0
            out_path = os.path.join(out_clase_dir, img_name)
            cv2.imwrite(out_path, img_crop)
            print(f"Guardado: {out_path}")

if __name__ == "__main__":
    crear_estructura_recortes()
    for split in ['train', 'test']:
        procesar_split(split)
    print("Recorte y aplicación de máscara completados.")
