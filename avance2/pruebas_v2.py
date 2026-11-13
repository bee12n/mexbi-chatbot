# -*- coding: utf-8 -*-
"""Conversaciones de prueba de Mexbi v0.2 (texto + imagen)."""
import sys
from mexbi_v2 import Mexbi

IMG = sys.argv[1] if len(sys.argv) > 1 else None   # carpeta con fotos de prueba

CONVERSACIONES = [
    ("1. Saludo y catálogo", ["Hola, buenas tardes", "¿qué tipos de banda manejan?", "gracias"]),
    ("2. Recomendación con contexto", ["no sé qué banda necesito", "es para una línea de alimentos", "va, 600 mm por 12 metros de PU"]),
    ("3. Cotización en un mensaje", ["cotízame una banda de pvc de 24 pulgadas por 15 metros"]),
    ("4. Pedido sin folio", ["quiero saber cómo va lo que pedí", "es el 1052"]),
    ("5. Frases que en v0.1 fallaban", ["¿Venden motores eléctricos?", "ayer le había pedido al vendedor una cotización", "ocupo que vengan a montar la banda"]),
    ("6. Reparación y escalamiento", ["se nos reventó la banda en la noche", "pásame con un asesor"]),
]

MULTIMODAL = [
    ("7. Foto -> identificación -> cotización", ["no sé qué banda tengo, te paso foto",
        ("img", "datos/fotos_demo/cliente_foto_15.jpg"), "es PU de 600 mm por 8 metros"]),
    ("8. Foto de banda con perfil", ["¿qué banda es esta?", ("img", "datos/fotos_demo/cliente_foto_17.jpg")]),
    ("9. Foto de grabado fino", [("img", "datos/fotos_demo/cliente_foto_18.jpg"), "gracias"]),
    ("10. Foto completa con etiqueta y mesa", [("img", "datos/fotos_demo/cliente_foto_completa_04.jpg")]),
]

def correr(bot, titulo, turnos):
    print("=" * 72); print("PRUEBA", titulo); print("=" * 72)
    for t in turnos:
        if isinstance(t, tuple):
            intencion, conf, r = bot.responder(imagen=t[1]); print(f"Usuario : [envía foto {t[1].split('/')[-1]}]")
        else:
            intencion, conf, r = bot.responder(t); print(f"Usuario : {t}")
        print(f"[intención: {intencion} | confianza: {conf:.0%}]")
        print(f"Mexbi   : {r}\n")

if __name__ == "__main__":
    for titulo, turnos in CONVERSACIONES:
        correr(Mexbi(usar_vision=False), titulo, turnos)
    for titulo, turnos in MULTIMODAL:
        correr(Mexbi(), titulo, turnos)
