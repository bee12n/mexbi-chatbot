# -*- coding: utf-8 -*-
"""
Clasificador de intenciones de Mexbi (Avance 2).

1. Carga datos/intenciones_mexbi.csv
2. Preprocesa con NLTK (preprocesamiento.py)
3. Representa el texto con Bag of Words y con TF-IDF
4. Entrena Naive Bayes multinomial y SVM lineal
5. Evalúa con partición 80/20 estratificada, validación cruzada de 5 pliegues
   y un conjunto de prueba externo escrito aparte (frases que el modelo nunca vio)
6. Guarda el mejor modelo en modelos/clasificador_intenciones.joblib
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocesamiento import preprocesar

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
SEMILLA = 42

df = pd.read_csv("datos/intenciones_mexbi.csv")
externo = pd.read_csv("datos/frases_prueba_externa.csv")
df["proc"] = df["texto"].apply(preprocesar)
df["proc_sin_stem"] = df["texto"].apply(lambda t: preprocesar(t, usar_stemming=False))
externo["proc"] = externo["texto"].apply(preprocesar)
externo["proc_sin_stem"] = externo["texto"].apply(lambda t: preprocesar(t, usar_stemming=False))

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    df["proc"], df["intencion"], df.index, test_size=0.2, stratify=df["intencion"],
    random_state=SEMILLA)


def vectorizador(tipo):
    # unigramas + bigramas porque "cuanto cuesta" o "hasta luego" dicen más juntas
    if tipo == "BoW":
        return CountVectorizer(ngram_range=(1, 2), min_df=1)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)


def clasificador(tipo):
    return MultinomialNB(alpha=0.3) if tipo == "Naive Bayes" else LinearSVC(C=1.0, random_state=SEMILLA)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas = []
for rep in ["BoW", "TF-IDF"]:
    for clf in ["Naive Bayes", "SVM lineal"]:
        for stem in [True, False]:
            col = "proc" if stem else "proc_sin_stem"
            pipe = Pipeline([("vec", vectorizador(rep)), ("clf", clasificador(clf))])
            cv_f1 = cross_val_score(pipe, df[col], df["intencion"], cv=cv, scoring="f1_macro")
            pipe.fit(df.loc[idx_tr, col], y_tr)
            pred = pipe.predict(df.loc[idx_te, col])
            pred_ext = pipe.predict(externo[col])
            filas.append({
                "Representación": rep, "Clasificador": clf, "Stemming": "Sí" if stem else "No",
                "CV F1 macro (5 pliegues)": f"{cv_f1.mean():.3f} ± {cv_f1.std():.3f}",
                "Accuracy prueba": accuracy_score(y_te, pred),
                "F1 macro prueba": f1_score(y_te, pred, average="macro"),
                "Accuracy externa": accuracy_score(externo["intencion"], pred_ext),
                "F1 macro externa": f1_score(externo["intencion"], pred_ext, average="macro"),
            })

comparacion = pd.DataFrame(filas)
comparacion.to_csv("resultados/comparacion_modelos_texto.csv", index=False)
pd.set_option("display.width", 200)
print(comparacion.round(3).to_string(index=False))

# ---- modelo final: TF-IDF + SVM lineal con stemming (ver comparación) ----
final = Pipeline([("vec", vectorizador("TF-IDF")),
                  ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEMILLA), cv=5))])
final.fit(X_tr, y_tr)
pred = final.predict(X_te)
# Para usarlo en el bot lo reentreno con el 100 % de las frases; el conjunto externo
# sigue sin tocarse en el entrenamiento, así que es la prueba más realista.
final.fit(df["proc"], df["intencion"])
pred_ext = final.predict(externo["proc"])

etiquetas = sorted(df["intencion"].unique())
reporte = classification_report(y_te, pred, digits=3, zero_division=0)
reporte_ext = classification_report(externo["intencion"], pred_ext, digits=3, zero_division=0)
print("\n=== Modelo final: TF-IDF + SVM (prueba 20 %) ===\n", reporte)
print("=== Modelo final (100 % de datos): conjunto externo ===\n", reporte_ext)
with open("resultados/reporte_texto.txt", "w", encoding="utf-8") as fh:
    fh.write("PRUEBA 20 %\n" + reporte + "\nCONJUNTO EXTERNO\n" + reporte_ext)

errores = externo.assign(prediccion=pred_ext)[lambda d: d["prediccion"] != d["intencion"]]
errores[["texto", "intencion", "prediccion"]].to_csv("resultados/errores_externo.csv", index=False)
print("Errores en el conjunto externo:\n", errores[["texto", "intencion", "prediccion"]].to_string(index=False))

metricas = {
    "n_frases": int(len(df)), "n_intenciones": int(df["intencion"].nunique()),
    "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)), "n_externo": int(len(externo)),
    "vocabulario": int(len(final.named_steps["vec"].vocabulary_)),
    "prueba": {"accuracy": accuracy_score(y_te, pred),
               "precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
               "recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
               "f1_macro": f1_score(y_te, pred, average="macro")},
    "externo": {"accuracy": accuracy_score(externo["intencion"], pred_ext),
                "precision_macro": precision_score(externo["intencion"], pred_ext, average="macro", zero_division=0),
                "recall_macro": recall_score(externo["intencion"], pred_ext, average="macro", zero_division=0),
                "f1_macro": f1_score(externo["intencion"], pred_ext, average="macro")},
}
json.dump(metricas, open("resultados/metricas_texto.json", "w"), indent=2)

# Matriz de confusión del conjunto externo (es la más honesta)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(confusion_matrix(externo["intencion"], pred_ext, labels=etiquetas),
                       display_labels=etiquetas).plot(ax=ax, cmap="Greys", colorbar=False, xticks_rotation=60)
ax.set_title("Matriz de confusión: intenciones (conjunto externo, 70 frases)")
ax.set_xlabel("Intención predicha"); ax.set_ylabel("Intención real")
plt.tight_layout(); plt.savefig("resultados/matriz_confusion_texto.png", dpi=200)

# Distribución del dataset
fig, ax = plt.subplots(figsize=(8, 4.5))
conteo = df["intencion"].value_counts().sort_values()
ax.barh(conteo.index, conteo.values, color="#4a4a4a")
for i, v in enumerate(conteo.values):
    ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Número de frases"); ax.set_title(f"Frases por intención (total {len(df)})")
plt.tight_layout(); plt.savefig("resultados/distribucion_intenciones.png", dpi=200)

joblib.dump(final, "modelos/clasificador_intenciones.joblib")
print("\nModelo guardado en modelos/clasificador_intenciones.joblib")
