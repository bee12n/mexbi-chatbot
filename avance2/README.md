# Mexbi v0.2 – Avance 2 (multimodal)

| Archivo | Qué hace |
|---|---|
| `generar_dataset.py` | Construye `datos/intenciones_mexbi.csv` (413 frases, 14 intenciones) |
| `datos/frases_prueba_externa.csv` | 70 frases de prueba que nunca se usan para entrenar |
| `preprocesamiento.py` | Normalización, tokenización, stopwords y stemming (NLTK) |
| `clasificador_texto.py` | Compara BoW/TF-IDF + Naive Bayes/SVM y guarda el modelo final |
| `datos/fotos_originales/`, `datos/regiones_banda.py` | 19 fotos de muestras de Mexbelt y la región de la banda en cada una |
| `clasificador_imagenes.py` | ResNet50 + cabeza densa, validación leave-one-photo-out |
| `mexbi_v2.py` | Asistente multimodal (texto + imagen) |
| `pruebas_v2.py` | Conversaciones de prueba |
| `Mexbi_v2_multimodal.ipynb` | Notebook para Google Colab |

```bash
pip install nltk scikit-learn tensorflow joblib
python generar_dataset.py && python clasificador_texto.py && python clasificador_imagenes.py
python pruebas_v2.py
python mexbi_v2.py     # chat; "/foto ruta.jpg" para mandar imagen
```
El modelo `modelos/clasificador_bandas.keras` (~95 MB) se genera con `clasificador_imagenes.py`.
