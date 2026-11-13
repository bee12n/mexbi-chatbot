# -*- coding: utf-8 -*-
"""
Clasificador visual de Mexbi: tipo de superficie de banda a partir de una foto.

Datos: 19 fotos reales de muestras de banda de Mexbelt (catálogo Chiorino y hule NT).
Como son pocas fotos, de cada una saco recortes (parches) de la zona de la banda
y evalúo con validación "dejando una foto fuera" (leave-one-photo-out): el modelo
nunca ve parches de la foto con la que se evalúa.

CNN: ResNet50 preentrenada en ImageNet (transferencia de aprendizaje). La base
queda congelada como extractor de características y encima se entrena una
cabeza densa nueva con softmax de 3 clases.

Salida: modelos/clasificador_bandas.keras, modelos/clases_bandas.json, resultados/*
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)

sys.path.insert(0, "datos")
from regiones_banda import FOTOS, CLASES_SUPERFICIE

SEMILLA = 42
TAM = 224
PARCHES_POR_FOTO = 24
LADO = 170                     # lado del recorte en píxeles de la foto original
PESOS = os.environ.get("PESOS_RESNET", "imagenet")
rng = np.random.default_rng(SEMILLA)
tf.keras.utils.set_random_seed(SEMILLA)
os.makedirs("modelos", exist_ok=True); os.makedirs("resultados", exist_ok=True)
os.makedirs("datos/parches", exist_ok=True)

CLASES = sorted(CLASES_SUPERFICIE)
NOMBRES = {"tela": "Tela", "grabado_fino": "Grabado fino", "perfil_agarre": "Perfil de agarre"}


# ---------------- 1. Recortes de la zona de la banda ----------------
def encoger(poligono, factor=0.92):
    p = np.array(poligono, float); c = p.mean(0)
    return c + (p - c) * factor


def sacar_parches(archivo, poligono, n):
    img = Image.open(f"datos/fotos_originales/{archivo}").convert("RGB")
    camino = Path(encoger(poligono))
    xs, ys = zip(*poligono)
    lado, parches, intentos = LADO, [], 0
    while len(parches) < n:
        intentos += 1
        if intentos % 400 == 0:
            lado = int(lado * 0.9)          # banda angosta: recorte más chico
        cx, cy = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        h = lado / 2
        esquinas = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx, cy)]
        if camino.contains_points(esquinas).all():
            parches.append(img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((TAM, TAM)))
    return parches


X, y, grupo = [], [], []
for i, (archivo, (clase, modelo, poligono)) in enumerate(FOTOS.items()):
    for j, p in enumerate(sacar_parches(archivo, poligono, PARCHES_POR_FOTO)):
        X.append(np.asarray(p)); y.append(CLASES.index(clase)); grupo.append(i)
        if j < 3:
            p.save(f"datos/parches/{archivo[:-4]}_{clase}_{j}.jpg")
X, y, grupo = np.array(X, "float32"), np.array(y), np.array(grupo)
fotos = list(FOTOS)
y_foto = np.array([CLASES.index(FOTOS[f][0]) for f in fotos])
print(f"{len(fotos)} fotos -> {len(X)} parches; fotos por clase:",
      {c: int((y_foto == k).sum()) for k, c in enumerate(CLASES)})

# ---------------- 2. ResNet50 como extractor (congelada) ----------------
base = tf.keras.applications.ResNet50(include_top=False, weights=PESOS, pooling="avg", input_shape=(TAM, TAM, 3))
base.trainable = False
pre = tf.keras.applications.resnet50.preprocess_input


def caracteristicas(imgs):
    return base.predict(pre(imgs.copy()), batch_size=32, verbose=0)


# aumento de datos: giros y espejos (la textura puede venir en cualquier orientación)
aumentos = [lambda a: a, lambda a: a[:, :, ::-1], lambda a: a[:, ::-1, :], lambda a: np.rot90(a, 1, (1, 2))]
F_aum = np.stack([caracteristicas(np.ascontiguousarray(f(X))) for f in aumentos])   # (4, N, 2048)
F = F_aum[0]
print("Características:", F.shape)


def cabeza():
    m = tf.keras.Sequential([
        tf.keras.Input((2048,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASES), activation="softmax"),
    ])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def entrenar(indices, epocas=40):
    Xtr = np.concatenate([F_aum[k][indices] for k in range(len(aumentos))])
    ytr = np.tile(y[indices], len(aumentos))
    m = cabeza()
    hist = m.fit(Xtr, ytr, epochs=epocas, batch_size=32, verbose=0, shuffle=True)
    return m, hist


# ---------------- 3. Validación leave-one-photo-out ----------------
prob_parche = np.zeros((len(X), len(CLASES)))
for g in range(len(fotos)):
    tr, te = grupo != g, grupo == g
    m, _ = entrenar(np.where(tr)[0])
    prob_parche[te] = m.predict(F[te], verbose=0)
    print(f"  fold {g + 1:2d}/{len(fotos)} {fotos[g]} real={CLASES[y_foto[g]]:14s} "
          f"pred={CLASES[prob_parche[te].mean(0).argmax()]}")

pred_parche = prob_parche.argmax(1)
prob_foto = np.array([prob_parche[grupo == g].mean(0) for g in range(len(fotos))])
pred_foto = prob_foto.argmax(1)


def metricas(real, pred):
    return {"accuracy": accuracy_score(real, pred),
            "precision_macro": precision_score(real, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(real, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(real, pred, average="macro", zero_division=0)}


nombres = [NOMBRES[c] for c in CLASES]
rep_parche = classification_report(y, pred_parche, target_names=nombres, digits=3, zero_division=0)
rep_foto = classification_report(y_foto, pred_foto, target_names=nombres, digits=3, zero_division=0)
print("\n=== Nivel parche ===\n", rep_parche, "\n=== Nivel foto ===\n", rep_foto)
open("resultados/reporte_imagenes.txt", "w", encoding="utf-8").write(
    "VALIDACIÓN LEAVE-ONE-PHOTO-OUT\n\nNivel parche\n" + rep_parche + "\nNivel foto\n" + rep_foto)

detalle = [{"foto": f, "modelo": FOTOS[f][1], "real": CLASES[y_foto[g]], "prediccion": CLASES[pred_foto[g]],
            "confianza": round(float(prob_foto[g].max()), 3),
            "parches_correctos": f"{int((pred_parche[grupo == g] == y_foto[g]).sum())}/{PARCHES_POR_FOTO}"}
           for g, f in enumerate(fotos)]
json.dump({"n_fotos": len(fotos), "n_parches": int(len(X)), "parches_por_foto": PARCHES_POR_FOTO,
           "clases": CLASES, "parche": metricas(y, pred_parche), "foto": metricas(y_foto, pred_foto),
           "detalle": detalle}, open("resultados/metricas_imagenes.json", "w"), indent=2, ensure_ascii=False)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (r, p, t) in zip(axs, [(y, pred_parche, f"Por parche (n={len(X)})"), (y_foto, pred_foto, f"Por foto (n={len(fotos)})")]):
    ConfusionMatrixDisplay(confusion_matrix(r, p, labels=range(len(CLASES))), display_labels=nombres).plot(
        ax=ax, cmap="Greys", colorbar=False)
    ax.set_title(t); ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
plt.suptitle("Matriz de confusión: superficie de banda (leave-one-photo-out)")
plt.tight_layout(); plt.savefig("resultados/matriz_confusion_imagenes.png", dpi=200)

# muestra de parches
fig, axs = plt.subplots(3, 6, figsize=(12, 6.3))
for fila, c in enumerate(CLASES):
    idx = rng.choice(np.where(y == fila)[0], 6, replace=False)
    for ax, i in zip(axs[fila], idx):
        ax.imshow(X[i].astype("uint8")); ax.set_xticks([]); ax.set_yticks([])
    axs[fila][0].set_ylabel(NOMBRES[c], fontsize=11)
plt.suptitle("Ejemplos de parches por clase"); plt.tight_layout()
plt.savefig("resultados/ejemplos_parches.png", dpi=150)

# ---------------- 4. Modelo final con todas las fotos ----------------
m_final, hist = entrenar(np.arange(len(X)))
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(hist.history["loss"], color="#222", label="Pérdida"); ax.plot(hist.history["accuracy"], "--", color="#888", label="Accuracy")
ax.set_xlabel("Época"); ax.set_title("Entrenamiento de la cabeza (modelo final)"); ax.legend()
plt.tight_layout(); plt.savefig("resultados/curvas_entrenamiento.png", dpi=200)

entrada = tf.keras.Input((TAM, TAM, 3))
salida = m_final(base(pre(entrada)))
modelo = tf.keras.Model(entrada, salida, name="mexbi_resnet50_superficie")
modelo.save("modelos/clasificador_bandas.keras")
json.dump(CLASES, open("modelos/clases_bandas.json", "w"))
print("Parámetros totales:", modelo.count_params(),
      "| entrenables:", int(sum(np.prod(v.shape) for v in m_final.trainable_variables)))
