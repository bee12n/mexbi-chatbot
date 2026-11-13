# -*- coding: utf-8 -*-
"""
Construcción del conjunto de datos de intenciones de Mexbi (Avance 2).

Las frases salen de cómo escriben los clientes de Mexbelt por WhatsApp, correo y
teléfono: con y sin acentos, con faltas, abreviaturas y mezclando temas. Hay frases
escritas a mano y otras que se arman con plantillas + valores del negocio
(materiales, industrias, medidas, folios) para tener más variedad.

Salida:
  datos/intenciones_mexbi.csv   -> texto, intencion, origen
"""
import csv
import random
import itertools

random.seed(42)

MATERIALES = ["pvc", "PU", "poliuretano", "modular", "hule", "caucho", "malla metálica",
              "banda de malla", "banda modular plástica", "banda lisa", "banda verde de pvc",
              "banda blanca sanitaria", "banda con chevron", "banda de lonas"]
INDUSTRIAS = ["alimentos", "minería", "paquetería", "automotriz", "panadería", "farmacéutica",
              "una pedrera", "una cementera", "un centro de distribución", "una planta de tortillas",
              "una empacadora de carne", "una línea de botellas", "reciclaje", "una fundidora"]
FOLIOS = ["SO1045", "SO1052", "SO1060", "so 1071", "SO-1088", "1094", "SO1102", "la orden 1110"]
MEDIDAS = ["600 mm", "24 pulgadas", "18\"", "1200 mm", "800mm", "36 in", "450 mm", "1 metro de ancho"]
LARGOS = ["12 metros", "8 m", "30 mts", "15 metros", "un rollo de 50 m", "6.5 m", "20 metros"]

FRASES = {
    "saludo": [
        "hola", "Hola buenas tardes", "buenos días", "buenas noches", "qué tal", "hola mexbi",
        "buen día", "holaa", "hola que tal", "saludos", "hey hola", "buenas", "hola, buenas",
        "hola buen dia", "q tal", "hola, ¿hay alguien?", "hola, ¿me pueden atender?",
        "buenas tardes, ¿con quién tengo el gusto?", "hola mexbelt", "hello", "ola",
        "buenos dias mexbelt", "hola, ¿están atendiendo?", "buenas, una consulta",
    ],
    "despedida": [
        "adiós", "bye", "hasta luego", "nos vemos", "chao", "hasta pronto", "me retiro",
        "es todo, adiós", "eso sería todo", "hasta mañana", "nos hablamos luego", "bye bye",
        "ya me voy", "luego les escribo", "me despido", "salir", "terminar", "cerrar chat",
        "ya no necesito nada, adiós", "listo, hasta luego", "eso es todo por hoy",
    ],
    "agradecimiento": [
        "gracias", "muchas gracias", "mil gracias", "muy amable", "te agradezco", "excelente, gracias",
        "perfecto gracias", "gracias por la info", "gracias por tu ayuda", "super, gracias",
        "ok gracias", "va, gracias", "muy amables", "se agradece", "gracias!!", "thanks",
        "gracias por responder tan rápido", "buenísimo, gracias", "de acuerdo, gracias",
        "muchísimas gracias", "gracias, me sirvió mucho", "perfecto, muy claro",
    ],
    "tipos_banda": [
        "¿qué tipos de banda manejan?", "qué bandas venden", "¿tienen catálogo?",
        "¿manejan banda modular?", "¿venden banda de pvc?", "¿qué materiales tienen?",
        "¿tienen banda de malla metálica?", "¿manejan banda de hule para trabajo pesado?",
        "quiero ver su catálogo de productos", "¿qué opciones de banda tienen?",
        "¿qué productos manejan?", "¿tienen banda de poliuretano?", "¿qué diferencia hay entre pvc y pu?",
        "¿manejan bandas sanitarias?", "¿tienen bandas con taquetes o perfiles?",
        "¿venden bandas de caucho?", "me puedes decir qué familias de bandas tienen",
        "¿qué es una banda modular?", "¿manejan banda tipo chevron?", "¿solo venden bandas?",
        "¿tienen bandas de colores?", "¿qué marcas de banda manejan?",
    ],
    "recomendacion": [
        "¿qué banda me recomiendas?", "¿cuál me conviene?", "no sé qué banda necesito",
        "¿qué banda sirve para alimentos?", "necesito una banda para minería, ¿cuál uso?",
        "¿qué banda es mejor para un horno?", "¿qué me recomiendas para cargar piedra?",
        "¿cuál banda aguanta aceite?", "busco una banda para producto congelado",
        "¿qué banda uso en un transportador inclinado?", "ayúdame a escoger una banda",
        "¿qué banda resiste más la abrasión?", "¿cuál es la mejor banda para cajas?",
        "tengo una línea de pan, ¿qué banda me sugieres?", "¿qué banda aguanta altas temperaturas?",
        "¿qué banda pongo para una curva?", "¿cuál banda es grado alimenticio?",
        "¿me puedes asesorar con el tipo de banda?", "¿qué banda le pongo a mi transportador?",
        "¿cuál es la indicada para lavado constante?",
    ],
    "cotizacion": [
        "quiero cotizar una banda", "¿cuánto cuesta?", "precio de banda modular",
        "¿me pasas una cotización?", "¿cuánto sale el metro de pvc?", "necesito un presupuesto",
        "¿cuánto vale una banda de hule?", "quiero comprar una banda", "cotízame por favor",
        "¿qué precio tiene la banda de pu?", "¿me cotizas 2 piezas?", "costo de una banda",
        "¿manejan precios de mayoreo?", "¿cuánto me cobrarían por una banda de 10 metros?",
        "mándame precio", "cuanto cuesta la banda blanca", "necesito precio urgente",
        "¿en cuánto me dejas el rollo?", "¿tienen lista de precios?", "quiero hacer un pedido nuevo",
        "pásame costo por metro lineal", "¿cuánto costaría cambiar toda la banda?",
    ],
    "instalacion": [
        "¿ustedes instalan?", "¿hacen instalación en planta?", "¿pueden venir a instalar la banda?",
        "necesito que me cambien la banda", "¿tienen técnicos para montaje?",
        "¿cuánto cobran por instalar?", "¿hacen visitas técnicas?", "¿pueden medir mi transportador?",
        "quiero que vengan a ver mi equipo", "¿instalan en fin de semana?", "¿hacen el cambio de banda en sitio?",
        "¿me pueden mandar un técnico?", "¿van a otras ciudades a instalar?",
        "¿la instalación está incluida?", "necesito montaje de banda nueva el sábado",
        "¿cuánto tardan en instalar una banda?", "¿hacen levantamiento de medidas?",
        "¿pueden agendar una visita a la planta?", "¿el técnico trae su herramienta?",
        "¿instalan transportadores completos?",
    ],
    "reparacion": [
        "se rompió la banda", "¿hacen empalmes?", "necesito vulcanizar una banda",
        "la banda está rota", "¿reparan bandas?", "se abrió el empalme", "la banda se desgastó",
        "¿hacen empalme en frío?", "tengo la línea parada porque se rompió la banda",
        "¿ponen grapas?", "se rajó la banda a la mitad", "la banda tiene un corte",
        "¿pueden reparar una banda de hule?", "el empalme ya no aguanta", "¿hacen vulcanizado en caliente?",
        "se le zafaron las grapas", "la orilla de la banda está deshilachada",
        "se despegó la banda", "¿tienen servicio de reparación urgente?", "¿cuánto cuesta un empalme?",
        "mi banda se está desgarrando", "se nos quemó un tramo de banda",
    ],
    "estatus_pedido": [
        "¿cómo va mi pedido?", "estatus de mi orden", "¿ya salió mi pedido?", "quiero rastrear mi pedido",
        "¿dónde está mi banda?", "¿mi pedido ya está listo?", "seguimiento a mi orden",
        "¿cuándo llega lo que pedí?", "no me ha llegado mi pedido", "¿ya enviaron mi banda?",
        "¿en qué va mi orden de compra?", "¿me das el status de mi pedido?",
        "¿ya está en producción mi banda?", "quiero saber de mi pedido", "¿ya facturaron mi pedido?",
        "mi pedido venía para hoy y no ha llegado", "¿tienen novedades de mi orden?",
        "¿me confirman si ya surtieron mi pedido?",
    ],
    "tiempo_entrega": [
        "¿cuánto tardan en entregar?", "¿en cuántos días me llega?", "¿qué tiempo de entrega manejan?",
        "¿tienen entrega inmediata?", "¿hacen envíos a otros estados?", "¿la tienen en existencia?",
        "¿cuánto tarda una banda especial?", "¿entregan el mismo día?", "¿envían a Saltillo?",
        "¿cuánto se tarda la importación?", "¿tienen en stock?", "¿hacen envíos foráneos?",
        "necesito la banda para mañana", "¿la pueden tener para el viernes?", "¿cuál es el tiempo de fabricación?",
        "¿el envío tiene costo?", "¿entregan en planta?", "¿cuánto tiempo tardan en confeccionarla?",
        "¿me la mandan por paquetería?", "¿la tienen disponible ya?",
    ],
    "horario_ubicacion": [
        "¿en qué horario atienden?", "¿dónde están ubicados?", "¿a qué hora abren?",
        "¿abren los sábados?", "¿cuál es su dirección?", "¿a qué hora cierran?",
        "¿tienen sucursal en Guadalajara?", "¿dónde los encuentro?", "¿cuál es su horario?",
        "¿atienden en domingo?", "¿puedo pasar a recoger?", "pásame la ubicación",
        "¿en qué parte de Monterrey están?", "¿tienen mostrador?", "¿abren hoy?",
        "¿hasta qué hora puedo pasar?", "¿cómo llego a sus oficinas?", "¿tienen bodega en Apodaca?",
        "¿atienden en días festivos?",
    ],
    "fichas_tecnicas": [
        "¿me mandas la ficha técnica?", "necesito las especificaciones de la banda",
        "¿tienen certificado FDA?", "¿qué temperatura aguanta?", "¿cuál es la carga de trabajo?",
        "¿me pasas la hoja técnica?", "¿cuántas lonas tiene?", "¿qué espesor tiene la banda?",
        "necesito el datasheet", "¿tiene certificación para alimentos?", "¿cuál es la tensión máxima?",
        "¿qué dureza tiene la cubierta?", "¿me compartes la ficha de la banda modular?",
        "¿cuál es el diámetro mínimo de polea?", "¿la banda es antiestática?",
        "¿es resistente a grasas y aceites?", "¿me pasas los datos técnicos?",
        "¿tienen hoja de seguridad del material?", "¿cuál es el paso de la banda modular?",
    ],
    "asesor": [
        "quiero hablar con un asesor", "¿me comunicas con una persona?", "necesito hablar con un humano",
        "pásame con ventas", "¿me pueden llamar?", "quiero que me contacte un vendedor",
        "¿cuál es su teléfono?", "¿tienen whatsapp?", "dame el contacto de un ingeniero",
        "prefiero hablar por teléfono", "¿me agendas una llamada?", "con un ejecutivo por favor",
        "¿puede atenderme alguien?", "no quiero hablar con un bot", "¿me pasas el correo de ventas?",
        "quiero hablar con Oscar", "¿con quién veo lo de mi cuenta?", "comunícame con el área comercial",
        "márquenme por favor", "¿me pueden marcar a este número?",
    ],
    "enviar_foto": [
        "te mando una foto de mi banda", "¿te puedo enviar una foto?", "¿qué banda es esta?",
        "no sé qué banda tengo, te paso foto", "mira esta foto", "te adjunto imagen de la banda",
        "¿me ayudas a identificar esta banda?", "¿puedes ver qué tipo de banda es en la foto?",
        "aquí está la foto de la banda que quiero", "¿con una foto me puedes decir qué banda es?",
        "te comparto una imagen", "quiero una igual a la de la foto", "¿reconoces esta banda?",
        "¿de qué material es esta banda?", "mando foto para que me digan qué es",
        "¿necesitas foto de la banda?", "te envío imagen del transportador",
        "esta es la banda que traigo actualmente", "¿puedo mandarte fotos para cotizar?",
    ],
}

# Plantillas para dar variedad con valores reales del negocio
PLANTILLAS = {
    "cotizacion": [
        "¿cuánto cuesta una {mat} de {med}?", "cotización de {mat} de {med} por {lar}",
        "precio de {mat} de {lar}", "necesito cotizar {mat} {med} x {lar}",
        "¿cuánto me sale una {mat} de {lar}?", "quiero precio de {mat} de {med}",
    ],
    "recomendacion": [
        "¿qué banda me recomiendas para {ind}?", "necesito una banda para {ind}",
        "¿cuál banda sirve para {ind}?", "trabajo en {ind}, ¿qué banda uso?",
        "¿qué me conviene para {ind}?",
    ],
    "tipos_banda": [
        "¿manejan {mat}?", "¿tienen {mat}?", "¿venden {mat}?", "¿qué tipos de {mat} tienen?",
    ],
    "estatus_pedido": [
        "¿cómo va mi pedido {fol}?", "estatus del {fol}", "¿ya salió el {fol}?",
        "rastrear pedido {fol}", "¿qué pasó con {fol}?", "¿me das seguimiento al {fol}?",
    ],
    "reparacion": [
        "se rompió mi {mat}", "¿pueden empalmar una {mat} de {med}?", "¿reparan {mat}?",
        "necesito vulcanizar una {mat} de {med}",
    ],
    "fichas_tecnicas": [
        "ficha técnica de la {mat}", "¿qué temperatura aguanta la {mat}?",
        "especificaciones de la {mat} de {med}",
    ],
    "instalacion": [
        "¿instalan {mat} en {ind}?", "necesito instalar una {mat} de {lar}",
    ],
    "tiempo_entrega": [
        "¿cuánto tardan en entregar una {mat} de {med}?", "¿tienen en existencia {mat} de {med}?",
    ],
    "enviar_foto": [
        "te mando foto de la {mat} que tengo", "¿con foto me dices si es {mat}?",
    ],
}


def ruido(texto: str) -> str:
    """Simula cómo escribe la gente en chat: sin acentos, sin signos, minúsculas."""
    r = random.random()
    if r < 0.25:
        texto = texto.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))
    if random.random() < 0.25:
        texto = texto.replace("¿", "").replace("?", "")
    if random.random() < 0.15:
        texto = texto.lower()
    return texto


def generar():
    filas = []
    for intencion, frases in FRASES.items():
        for f in frases:
            filas.append((f, intencion, "manual"))
    for intencion, plantillas in PLANTILLAS.items():
        combinaciones = []
        for p in plantillas:
            for _ in range(4):
                combinaciones.append(p.format(mat=random.choice(MATERIALES), ind=random.choice(INDUSTRIAS),
                                              med=random.choice(MEDIDAS), lar=random.choice(LARGOS),
                                              fol=random.choice(FOLIOS)))
        for f in dict.fromkeys(combinaciones):        # quita duplicados
            filas.append((ruido(f), intencion, "plantilla"))
    random.shuffle(filas)
    return filas


if __name__ == "__main__":
    filas = generar()
    with open("datos/intenciones_mexbi.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["texto", "intencion", "origen"])
        w.writerows(filas)
    from collections import Counter
    c = Counter(i for _, i, _ in filas)
    print(f"{len(filas)} frases, {len(c)} intenciones")
    for k, v in sorted(c.items()):
        print(f"  {k:18s} {v}")
