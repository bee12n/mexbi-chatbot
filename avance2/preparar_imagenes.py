# -*- coding: utf-8 -*-
"""
Prepara las fotos de bandas de Mexbelt para entrenar la CNN.

- Acepta una carpeta por tipo (pvc/, modular/, malla/, hule/) o archivos cuyo nombre
  empieza con el tipo (pvc_001.jpg, modular-foto3.png ...).
- Corrige la orientación EXIF de las fotos del celular, convierte a RGB/JPG
  (incluye HEIC de iPhone si está pillow-heif), reduce a 512 px por lado
  y elimina duplicados exactos.

Uso:  python preparar_imagenes.py <carpeta_o_zip_origen> datos/imagenes
"""
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

CLASES = {
    "pvc": "pvc_pu_lisa", "pu": "pvc_pu_lisa", "lisa": "pvc_pu_lisa", "poliuretano": "pvc_pu_lisa",
    "modular": "modular_plastica", "plastica": "modular_plastica",
    "malla": "malla_metalica", "metalica": "malla_metalica", "metal": "malla_metalica",
    "hule": "hule_caucho", "caucho": "hule_caucho", "lona": "hule_caucho", "chevron": "hule_caucho",
}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


def clase_de(ruta: str):
    partes = [p.lower() for p in ruta.replace("\\", "/").split("/")]
    for parte in reversed(partes):          # primero el nombre del archivo, luego carpetas
        for clave, clase in CLASES.items():
            if parte.startswith(clave) or f"/{clave}" in "/" + parte:
                return clase
    return None


def preparar(origen: str, destino: str, lado_max: int = 512):
    tmp = None
    if zipfile.is_zipfile(origen):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(origen).extractall(tmp)
        origen = tmp
    vistos, conteo, omitidas = set(), Counter(), []
    for raiz, _, archivos in os.walk(origen):
        for nombre in sorted(archivos):
            if os.path.splitext(nombre)[1].lower() not in EXT or nombre.startswith("._"):
                continue
            ruta = os.path.join(raiz, nombre)
            clase = clase_de(os.path.relpath(ruta, origen))
            if clase is None:
                omitidas.append(ruta); continue
            try:
                im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
            except Exception as e:
                omitidas.append(f"{ruta} ({e})"); continue
            im.thumbnail((lado_max, lado_max))
            h = hashlib.md5(im.tobytes()).hexdigest()
            if h in vistos:
                continue
            vistos.add(h)
            os.makedirs(os.path.join(destino, clase), exist_ok=True)
            conteo[clase] += 1
            im.save(os.path.join(destino, clase, f"{clase}_{conteo[clase]:03d}.jpg"), quality=92)
    if tmp:
        shutil.rmtree(tmp)
    print("Imágenes por clase:", dict(conteo))
    if omitidas:
        print(f"Omitidas ({len(omitidas)}):", *omitidas[:10], sep="\n  ")
    return conteo


if __name__ == "__main__":
    preparar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "datos/imagenes")
