# -*- coding: utf-8 -*-
"""
Región de la banda en cada foto (polígono en píxeles de la foto original 1080x2340)
y la clase de superficie que tiene según la etiqueta de la muestra.

Marqué a mano solo la parte de la banda para que la CNN aprenda la textura
y no lea la tarjeta impresa ni la mesa.
"""
CLASES_SUPERFICIE = {
    "tela": "Superficie de tela (fabric light/medium/heavy embossed)",
    "grabado_fino": "Grabado fino tipo pirámide negativa o grano de arroz",
    "perfil_agarre": "Perfil de agarre: conos, diente de sierra o ranuras",
}

FOTOS = {
    # foto: (clase, modelo de la muestra, polígono)
    "foto_01.jpg": ("tela", "NT4 HS (NA1141) - hule verde", [(150, 895), (1000, 935), (1000, 1125), (150, 1095)]),
    "foto_02.jpg": ("tela", "NT3 HS (NA1140) - hule verde", [(270, 1335), (600, 1310), (1050, 1345), (1050, 1470), (600, 1440), (270, 1490)]),
    "foto_03.jpg": ("tela", "1M5 U0-U2 HP FL BLUE A (NA1013)", [(150, 1070), (1080, 890), (1080, 1230), (150, 1420)]),
    "foto_04.jpg": ("tela", "1M5 U3-U3 HP FL/FM W (NA1191)", [(150, 1220), (1050, 975), (1060, 1290), (200, 1540)]),
    "foto_05.jpg": ("grabado_fino", "2M5 U0-U2 HP PN BLUE A (NA811)", [(150, 1200), (1000, 830), (1060, 1150), (200, 1530)]),
    "foto_06.jpg": ("tela", "2M5 U0-U2 HP FH W A (NA1182)", [(170, 1170), (1050, 910), (1060, 1200), (200, 1470)]),
    "foto_07.jpg": ("grabado_fino", "2M5 U0-U2 HP PN W A (NA842)", [(170, 1190), (1050, 870), (1060, 1240), (250, 1560)]),
    "foto_08.jpg": ("perfil_agarre", "2M5 U0-U7 HP LG BLUE S A (NA1193)", [(200, 1290), (1050, 860), (1060, 1170), (280, 1600)]),
    "foto_09.jpg": ("perfil_agarre", "2M5 U0-U8 HP CC BLUE (NA1130)", [(200, 1290), (1000, 740), (1060, 990), (330, 1580)]),
    "foto_10.jpg": ("perfil_agarre", "2M5 U0-U15 HP ST W A (NA1087)", [(150, 1150), (950, 850), (1060, 1100), (200, 1470)]),
    "foto_11.jpg": ("tela", "2M5 U0-U7 HP FH Blue S A (NA1340)", [(180, 1200), (1050, 730), (1060, 1090), (250, 1530)]),
    "foto_12.jpg": ("grabado_fino", "2M5 U0-U7 HP RG Blue (NA1518)", [(160, 1050), (1000, 820), (1060, 1120), (200, 1380)]),
    "foto_13.jpg": ("perfil_agarre", "2M5 U0-U8 HP STL Blue (NA1324)", [(180, 1210), (1060, 800), (1060, 1090), (260, 1530)]),
    "foto_14.jpg": ("grabado_fino", "2M5 U2-U2 HP PN BLUE A (NA1302)", [(200, 1280), (1060, 810), (1060, 1110), (270, 1590)]),
    "foto_15.jpg": ("tela", "2M5 U3-U3 HP FM/FH BLUE A (NA1329)", [(120, 940), (1050, 930), (1050, 1250), (120, 1260)]),
    "foto_16.jpg": ("perfil_agarre", "2M5 U3-U8 HP FM CC BLUE (NA1320)", [(120, 990), (1040, 985), (1040, 1270), (120, 1290)]),
    "foto_17.jpg": ("perfil_agarre", "2M5 U3-U15 HP FM ST BLUE A (NA1337)", [(80, 920), (1050, 910), (1050, 1220), (80, 1230)]),
    "foto_18.jpg": ("grabado_fino", "COMPACT 25 PN BLUE (NA1294)", [(120, 1000), (1060, 990), (1060, 1300), (120, 1320)]),
    "foto_19.jpg": ("grabado_fino", "2T12 U3-U3 HP RG Blue (NA1393)", [(440, 950), (610, 950), (610, 1440), (440, 1440)]),
}
