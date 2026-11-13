# -*- coding: utf-8 -*-
"""
Mexbi v0.2: primera versión multimodal del asistente de Mexbelt.

Cambios contra el Avance 1 (v0.1, solo reglas):
  - La intención ya no se decide contando palabras clave: la predice un
    clasificador TF-IDF + SVM entrenado (clasificador_texto.py), con un
    umbral de confianza para no contestar a ciegas.
  - Las reglas se quedan solo para lo que hacen bien: extraer datos concretos
    (folio SO, ancho, largo, material) y manejar el contexto de la conversación.
  - Nuevo componente visual: el cliente manda una foto y una CNN con ResNet50
    (clasificador_imagenes.py) identifica el tipo de banda. Mexbi traduce ese
    resultado a lenguaje natural y lo usa para seguir con la cotización.

Uso en consola:  python mexbi_v2.py
  Escribe normal, o "/foto ruta/a/imagen.jpg" para mandar una imagen.
"""
import json
import os
import random
import re

import joblib
import numpy as np

from preprocesamiento import normalizar, preprocesar, quitar_acentos

DIR = os.path.dirname(os.path.abspath(__file__))
UMBRAL_TEXTO = 0.30      # debajo de esto, Mexbi pide aclarar en vez de adivinar
UMBRAL_IMAGEN = 0.60     # debajo de esto, pide otra foto

BASE_CONOCIMIENTO = {
    "ciudad": "Monterrey, Nuevo León",
    "horario": "lunes a viernes de 8:00 a 18:00 h y sábados de 8:00 a 13:00 h",
    "correo": "ventas@mexbelt.com",
    "tipos_banda": {
        "PVC / PU lisa": "uso general, paquetería, empaque y, en PU, contacto directo con alimentos (FDA)",
        "Modular plástica": "lavado intensivo, curvas, congelado y procesos húmedos",
        "Hule / caucho": "trabajo pesado: minería, agregados, cemento y acero",
        "Malla metálica": "hornos, enfriamiento y altas temperaturas",
    },
    "industrias": {
        "alimentos": "banda de PU o modular plástica (grado alimenticio)",
        "mineria": "banda de hule con lonas (EP) y cubierta resistente a la abrasión",
        "paqueteria": "banda de PVC con cubierta de baja fricción, o rugosa si es inclinado",
        "automotriz": "banda de PVC o modular según el peso de las piezas",
        "panaderia": "banda de PU, o malla metálica si pasa por horno",
        "farmaceutica": "banda de PU sanitaria",
    },
    "pedidos": {"SO1045": "en producción, con entrega estimada el viernes",
                "SO1052": "listo para envío; sale mañana con el chofer",
                "SO1060": "entregado e instalado"},
}

# Lo que Mexbi le dice al cliente según la superficie que vio la CNN
INFO_VISUAL = {
    "tela": ("una banda con superficie de tela (acabado textil)",
             "Es de baja fricción: se usa para transporte horizontal, cajas, paquetería y acumulación de producto. "
             "En hule verde es la típica de papel y madera; en TPU azul o blanco, de alimentos.",
             "superficie de tela"),
    "grabado_fino": ("una banda con grabado fino (pirámide negativa o grano de arroz)",
                     "Da un agarre ligero y se limpia fácil; la usamos mucho en panadería, chocolate, carnes "
                     "e inclinaciones suaves.",
                     "grabado fino"),
    "perfil_agarre": ("una banda con perfil de agarre (conos, diente de sierra o ranuras)",
                      "Es para inclinados, para que el producto no ruede o se resbale, y para rebanadoras "
                      "de carne y queso.",
                      "perfil de agarre"),
}

RESPUESTAS_FIJAS = {
    "saludo": ["¡Hola! Soy Mexbi, el asistente de Mexbelt. Te ayudo con tipos de banda, cotizaciones, "
               "instalación, reparaciones o el estatus de tu pedido. También puedes mandarme una foto "
               "de tu banda y te digo qué tipo es."],
    "despedida": ["¡Gracias por escribir a Mexbelt! Aquí andamos cuando nos necesites."],
    "agradecimiento": ["¡Con gusto! ¿Te ayudo con algo más?", "Para eso estoy. ¿Algo más?"],
    "instalacion": ["Sí, instalamos en tu planta. Primero hacemos visita técnica para medir y revisar el "
                    "transportador, luego planeamos la banda y el herramental, y la instalación normalmente "
                    "es en fin de semana para no parar tu línea. ¿Quieres que un asesor agende la visita?"],
    "reparacion": ["Hacemos empalmes y reparaciones: vulcanizado en caliente o en frío y empalme mecánico con "
                   "grapas. Si me mandas una foto de la banda te digo de qué tipo es; si la línea está parada, "
                   "pide un asesor y lo atendemos como urgente."],
    "tiempo_entrega": ["Las bandas de inventario se confeccionan en 1 a 3 días hábiles; las de importación o "
                       "fabricación especial tardan de 2 a 6 semanas. Si es urgente te paso con un asesor."],
    "fichas_tecnicas": ["Te mandamos la ficha técnica de cualquier banda (espesor, lonas, carga de trabajo, "
                        "temperatura y certificados como FDA). ¿De qué modelo o tipo de banda la necesitas?"],
}

ACLARAR = ("No estoy segura de haberte entendido. ¿Me lo dices de otra forma? Puedo ayudarte con tipos de "
           "banda, recomendación, cotizaciones, instalación, reparaciones, pedidos, tiempos de entrega, "
           "horario, fichas técnicas o identificar tu banda con una foto.")


class ClasificadorVisual:
    """Envuelve la CNN para que el bot solo reciba (clase, confianza, top-2)."""

    def __init__(self, ruta_modelo=None, ruta_clases=None):
        import tensorflow as tf
        self.tf = tf
        self.modelo = tf.keras.models.load_model(ruta_modelo or os.path.join(DIR, "modelos/clasificador_bandas.keras"))
        self.clases = json.load(open(ruta_clases or os.path.join(DIR, "modelos/clases_bandas.json")))

    def predecir(self, ruta_imagen):
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(ruta_imagen)).convert("RGB")
        # 5 recortes (centro y esquinas) y promedio, igual que con los parches de entrenamiento
        w, h = img.size; lado = int(min(w, h) * 0.7)
        cajas = [((w - lado) // 2, (h - lado) // 2), (0, 0), (w - lado, 0), (0, h - lado), (w - lado, h - lado)]
        lote = np.stack([np.asarray(img.crop((x, y, x + lado, y + lado)).resize((224, 224)), dtype="float32")
                         for x, y in cajas])
        prob = self.modelo.predict(lote, verbose=0).mean(0)
        orden = prob.argsort()[::-1]
        return [(self.clases[i], float(prob[i])) for i in orden[:2]]


class Mexbi:
    def __init__(self, semilla=7, usar_vision=True):
        random.seed(semilla)
        self.clf_texto = joblib.load(os.path.join(DIR, "modelos/clasificador_intenciones.joblib"))
        self.vision = None
        if usar_vision and os.path.exists(os.path.join(DIR, "modelos/clasificador_bandas.keras")):
            self.vision = ClasificadorVisual()
        self.contexto = None
        self.cotizacion = {}
        self.historial = []

    # ------------------------------------------------------------------ texto
    def clasificar_texto(self, mensaje):
        prob = self.clf_texto.predict_proba([preprocesar(mensaje)])[0]
        i = prob.argmax()
        return self.clf_texto.classes_[i], float(prob[i])

    def _extraer_cotizacion(self, mensaje):
        t = quitar_acentos(mensaje.lower())
        ancho = re.search(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|pulgadas|pulg|in|\")", t)
        largo = re.search(r"(\d+(?:[.,]\d+)?)\s*(m|mts|metros)\b", t) or \
            re.search(r"largo\s*(?:de)?\s*(\d+(?:[.,]\d+)?)\s*(mm)", t)     # bug del Avance 1: largo en mm
        material = next((m for m in ["pvc", "pu", "poliuretano", "modular", "hule", "caucho", "malla"]
                         if re.search(rf"\b{m}\b", t)), None)
        if material:
            self.cotizacion["material"] = material.upper()
        if ancho:
            self.cotizacion["ancho"] = f"{ancho.group(1)} {ancho.group(2)}"
        if largo and largo.group(0) != (ancho.group(0) if ancho else None):
            self.cotizacion["largo"] = f"{largo.group(1)} {largo.group(2)}"
        return bool(ancho or largo or material)

    def _estado_cotizacion(self):
        faltan = [d for d in ["material", "ancho", "largo"] if d not in self.cotizacion]
        if faltan:
            self.contexto = "cotizacion"
            return "Anotado. Para la cotización me falta: " + ", ".join(faltan) + "."
        self.contexto = None
        d = self.cotizacion
        sup = f" con {d['superficie']}" if "superficie" in d else ""
        return (f"Perfecto, registré tu solicitud: banda {d['material']}{sup}, ancho {d['ancho']}, largo {d['largo']}. "
                "Un asesor te manda la cotización formal en menos de 24 horas hábiles.")

    def _pedido(self, mensaje):
        m = re.search(r"\bso\s*-?\s*(\d{3,})\b", mensaje.lower()) or \
            (re.search(r"\b(\d{4,})\b", mensaje) if self.contexto == "estatus_pedido" else None)
        if not m:
            self.contexto = "estatus_pedido"
            return "Claro, ¿me compartes tu número de pedido? Empieza con SO, por ejemplo SO1045."
        self.contexto = None
        folio = f"SO{m.group(1)}"
        estado = BASE_CONOCIMIENTO["pedidos"].get(folio)
        return f"Tu pedido {folio} está {estado}." if estado else \
            f"No encuentro el pedido {folio}. Revisa el número o pide un asesor para que lo busque."

    def _recomendacion(self, mensaje):
        t = normalizar(mensaje)
        for ind, sug in BASE_CONOCIMIENTO["industrias"].items():
            if ind[:-1] in t:
                self.contexto = "cotizacion"
                return f"Para ese proceso normalmente recomendamos {sug}. Si me pasas ancho y largo te armo la cotización."
        self.contexto = "recomendacion"
        return "Con gusto. ¿En qué industria o proceso la vas a usar? (alimentos, minería, paquetería, panadería…)"

    def _responder_intencion(self, intencion, mensaje):
        b = BASE_CONOCIMIENTO
        if intencion in RESPUESTAS_FIJAS:
            return random.choice(RESPUESTAS_FIJAS[intencion])
        if intencion == "tipos_banda":
            return ("Manejamos estas familias de banda:\n" +
                    "\n".join(f"• {k}: {v}" for k, v in b["tipos_banda"].items()) +
                    "\n¿Para qué proceso la necesitas? Si ya tienes una, mándame foto y te digo cuál es.")
        if intencion == "recomendacion":
            return self._recomendacion(mensaje)
        if intencion == "cotizacion":
            if self._extraer_cotizacion(mensaje):
                return self._estado_cotizacion()
            self.contexto = "cotizacion"
            return ("¡Claro! Para cotizarte necesito material, ancho, largo, tipo de empalme y cantidad. "
                    "Si no sabes el material, mándame una foto de la banda.")
        if intencion == "estatus_pedido":
            return self._pedido(mensaje)
        if intencion == "horario_ubicacion":
            return f"Estamos en {b['ciudad']}. Atendemos {b['horario']}. También puedes escribir a {b['correo']}."
        if intencion == "asesor":
            self.contexto = None
            return f"Te comunico con un asesor de ventas. Déjame tu nombre, empresa y teléfono, o escribe a {b['correo']}."
        if intencion == "enviar_foto":
            self.contexto = "esperando_foto"
            return ("¡Sí! Mándame la foto de la banda. Que se vea de cerca la superficie (sin etiqueta ni fondo), con "
                    "buena luz; con eso te digo qué tipo de superficie es.")
        return ACLARAR

    # ----------------------------------------------------------------- imagen
    def recibir_imagen(self, ruta_imagen):
        if self.vision is None:
            return "imagen", 0.0, "Por ahora no puedo revisar fotos; te paso con un asesor para que la vea."
        (clase, conf), (clase2, conf2) = self.vision.predecir(ruta_imagen)
        if conf < UMBRAL_IMAGEN:
            nombre1, nombre2 = INFO_VISUAL[clase][0], INFO_VISUAL[clase2][0]
            self.contexto = "esperando_foto"
            texto = (f"No alcanzo a distinguir bien la banda. Puede ser {nombre1} ({conf:.0%}) o {nombre2} "
                     f"({conf2:.0%}). ¿Me mandas otra foto más de cerca y con mejor luz?")
        else:
            nombre, explicacion, superficie = INFO_VISUAL[clase]
            self.cotizacion["superficie"] = superficie
            self.contexto = "cotizacion"
            texto = (f"Por la foto parece {nombre}, con {conf:.0%} de confianza. {explicacion} "
                     "Para cotizarte una igual pásame el material (PVC, PU o hule), el ancho y el largo.")
        return f"imagen:{clase}", conf, texto

    # ------------------------------------------------------------------ ciclo
    def responder(self, mensaje=None, imagen=None):
        if imagen:
            intencion, conf, respuesta = self.recibir_imagen(imagen)
        else:
            intencion, conf = self.clasificar_texto(mensaje)
            respuesta = None
            # 1) si Mexbi esperaba un dato, primero intenta usarlo (reglas)
            if self.contexto == "cotizacion" and intencion not in ("despedida", "asesor", "estatus_pedido") \
                    and self._extraer_cotizacion(mensaje):
                respuesta, intencion = self._estado_cotizacion(), "cotizacion (seguimiento)"
            elif self.contexto == "estatus_pedido" and re.search(r"\d{3,}", mensaje):
                respuesta, intencion = self._pedido(mensaje), "estatus_pedido (seguimiento)"
            elif self.contexto == "recomendacion" and any(i[:-1] in normalizar(mensaje) for i in BASE_CONOCIMIENTO["industrias"]):
                respuesta, intencion = self._recomendacion(mensaje), "recomendacion (seguimiento)"
            # 2) si no, decide el clasificador, con umbral de confianza
            if respuesta is None:
                respuesta = self._responder_intencion(intencion, mensaje) if conf >= UMBRAL_TEXTO else ACLARAR
                if conf < UMBRAL_TEXTO:
                    intencion = f"baja confianza ({intencion})"
        self.historial.append({"usuario": mensaje or f"[imagen] {os.path.basename(imagen)}",
                               "intencion": intencion, "confianza": round(conf, 3), "bot": respuesta})
        return intencion, conf, respuesta


def chatear():
    bot = Mexbi(semilla=None)
    print("Mexbi: ¡Hola! Soy Mexbi de Mexbelt. Escribe tu mensaje o '/foto ruta.jpg'. 'salir' para terminar.")
    while True:
        entrada = input("Tú: ").strip()
        if entrada.lower().startswith("/foto"):
            intencion, conf, r = bot.responder(imagen=entrada[5:].strip())
        else:
            intencion, conf, r = bot.responder(entrada)
        print(f"  [{intencion} · {conf:.0%}]\nMexbi: {r}")
        if intencion == "despedida":
            break


if __name__ == "__main__":
    chatear()
