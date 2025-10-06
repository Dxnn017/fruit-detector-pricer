import os
import cv2
import numpy as np
from glob import glob

# Rutas base

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILTRADO_DIR = os.path.join(BASE_DIR, 'Dataset_Filtrado')  # Imágenes originales a color
DATASET_SEGMENTADO_DIR = os.path.join(BASE_DIR, 'Dataset_Segmentado')  # Para referencia de clases/estructura
MASKS_DIR = os.path.join(BASE_DIR, 'Morfologia', 'masks_clean')
RECORTES_DIR = os.path.join(BASE_DIR, 'recortes')

# Crear carpetas de salida
def crear_estructura_recortes():
    for split in ['train', 'test']:
        split_path = os.path.join(RECORTES_DIR, split)
        os.makedirs(split_path, exist_ok=True)
        clases = os.listdir(os.path.join(DATASET_SEGMENTADO_DIR, split))
        for clase in clases:
            os.makedirs(os.path.join(split_path, clase), exist_ok=True)

# Encontrar bounding box de la máscara
def bounding_box(mask):
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

# Procesar imágenes
def procesar_split(split):
    # Usar la estructura de Dataset_Segmentado para clases y nombres, pero cargar imagen original de Dataset_Filtrado
    seg_dir = os.path.join(DATASET_SEGMENTADO_DIR, split)
    mask_dir = os.path.join(MASKS_DIR, split)
    out_dir = os.path.join(RECORTES_DIR, split)
    filtrado_dir = os.path.join(DATASET_FILTRADO_DIR, split)
    clases = os.listdir(seg_dir)
    for clase in clases:
        seg_clase_dir = os.path.join(seg_dir, clase)
        mask_clase_dir = os.path.join(mask_dir, clase)
        out_clase_dir = os.path.join(out_dir, clase)
        filtrado_clase_dir = os.path.join(filtrado_dir, clase)
        imgs = glob(os.path.join(seg_clase_dir, '*.jpg'))
        for img_path in imgs:
            img_name = os.path.basename(img_path)
            name_base, _ = os.path.splitext(img_name)
            mask_path = os.path.join(mask_clase_dir, img_name)
            img_filtrado_path = os.path.join(filtrado_clase_dir, img_name)
            if not os.path.exists(mask_path):
                print(f"No existe máscara para {img_path}")
                continue
            if not os.path.exists(img_filtrado_path):
                print(f"No existe imagen original filtrada para {img_filtrado_path}")
                continue
            img = cv2.imread(img_filtrado_path)  # Imagen original a color
            mask = cv2.imread(mask_path, 0)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            x, y, w, h = bounding_box(mask_bin)
            img_crop = img[y:y+h, x:x+w]
            mask_crop = mask_bin[y:y+h, x:x+w]
            # Aplicar máscara: fondo negro
            img_crop_masked = img_crop.copy()
            img_crop_masked[mask_crop == 0] = 0
            # Guardar máscara recortada
            mask_out_path = os.path.join(out_clase_dir, f"{name_base}_mask.png")
            cv2.imwrite(mask_out_path, mask_crop)
            # Guardar imagen recortada con fondo negro
            recorte_out_path = os.path.join(out_clase_dir, f"{name_base}_recorte.png")
            cv2.imwrite(recorte_out_path, img_crop_masked)
            print(f"Guardado: {mask_out_path} y {recorte_out_path}")

if __name__ == "__main__":
    crear_estructura_recortes()
    for split in ['train', 'test']:
        procesar_split(split)
    print("Recorte y aplicación de máscara completados.")
