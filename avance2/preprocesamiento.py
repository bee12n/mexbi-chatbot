# -*- coding: utf-8 -*-
"""
Preprocesamiento de texto de Mexbi (viene del Avance 1, ahora como módulo aparte).

mensaje -> normalización -> tokenización -> stopwords -> stemming (Snowball)

Lo dejo en un módulo propio para usar exactamente el mismo proceso al entrenar
el clasificador y cuando el bot recibe un mensaje en vivo.
"""
import re
import unicodedata

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

for recurso, ruta in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(ruta)
    except LookupError:
        nltk.download(recurso, quiet=True)

STEMMER = SnowballStemmer("spanish")


def quitar_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


# "no" y "sin" se quedan porque cambian el sentido; "que", "cual", "cuanto", "donde"
# y "como" también, porque distinguen preguntas (¿cuánto? = precio, ¿dónde? = ubicación)
CONSERVAR = {"no", "sin", "que", "cual", "cuanto", "cuanta", "donde", "como", "cuando"}
STOPWORDS = {quitar_acentos(w) for w in stopwords.words("spanish")} - CONSERVAR

# Unifico abreviaturas y formas de chat que salen mucho con los clientes
SINONIMOS = {"q": "que", "xq": "porque", "pq": "porque", "mts": "metros", "mt": "metros",
             "pulg": "pulgadas", "in": "pulgadas", "porfa": "favor", "plis": "favor",
             "pls": "favor", "info": "informacion", "tel": "telefono", "cel": "telefono",
             "wa": "whatsapp", "whats": "whatsapp", "ocupo": "necesito", "ocupamos": "necesitamos"}


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, sin signos; los números con unidad se vuelven etiquetas."""
    texto = quitar_acentos(texto.lower())
    texto = re.sub(r"\bso\s*-?\s*\d{3,}\b", " folio_pedido ", texto)        # SO1045 -> folio
    texto = re.sub(r"\d+(?:[.,]\d+)?\s*(mm|cm|\"|pulgadas|pulg|in)\b", " medida_ancho ", texto)
    texto = re.sub(r"\d+(?:[.,]\d+)?\s*(m|mts|metros)\b", " medida_largo ", texto)
    texto = re.sub(r"\d+", " numero ", texto)
    texto = re.sub(r"[^a-zñ_ ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto: str) -> list:
    tokens = word_tokenize(texto, language="spanish")
    return [SINONIMOS.get(t, t) for t in tokens]


def quitar_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


def stemming(tokens: list) -> list:
    return [t if "_" in t else STEMMER.stem(t) for t in tokens]


def preprocesar(texto: str, usar_stemming: bool = True, detalle: bool = False):
    limpio = normalizar(texto)
    tokens = tokenizar(limpio)
    sin_stop = quitar_stopwords(tokens)
    final = stemming(sin_stop) if usar_stemming else sin_stop
    if detalle:
        return {"original": texto, "normalizado": limpio, "tokens": tokens,
                "sin_stopwords": sin_stop, "raices": final}
    return " ".join(final)


if __name__ == "__main__":
    for ej in ["¿Cuánto cuesta una banda modular de 600 mm por 12 metros?",
               "Hola, ¿cómo va mi pedido SO1045? ya me urge",
               "q onda, ocupo una banda pa alimentos porfa"]:
        d = preprocesar(ej, detalle=True)
        for k, v in d.items():
            print(f"{k:14s}: {v}")
        print()
