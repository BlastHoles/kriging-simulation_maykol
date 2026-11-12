"""Los 10 experimentos didácticos (solo lógica, sin Streamlit).

Cada escenario define: pregunta de aprendizaje, parámetros que se modifican, una predicción
(opción múltiple) ANTES de ejecutar, la ejecución (mapas), la interpretación y preguntas para
reflexionar. Los valores propios de cada escenario son SUPUESTOS DIDÁCTICOS, no del artículo.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .data import Grilla, p_a_K, zona_de_x
from .interpolation import ConfigInterp, ErrorKriging, detalle_celda, interpolar_campo, kriging
from .variogram import Variograma


@dataclass
class Escenario:
    id: int
    titulo: str
    pregunta: str
    modifica: str
    prediccion: str
    opciones: list[str]
    correcta: int
    interpretacion: str
    reflexion: list[str]
    extra: str = ""                                   # vista adicional que dibuja la app
    controles: dict = field(default_factory=dict)     # nombre -> (etiqueta, min, max, defecto, paso)
    usa_datos_propios: bool = False
    requiere_modflow: bool = False


ESCENARIOS: list[Escenario] = [
    Escenario(
        1, "Efecto del rango/escala `a`",
        "¿Cómo cambia el mapa de K (y la incertidumbre) cuando solo se modifica `a`?",
        "Solo `a` (100, 350 y 700 m). Kriging ordinario; modelo, sill, nugget y puntos fijos.",
        "Si aumento `a` manteniendo todo lo demás, el mapa de log10(K) y la varianza media de kriging…",
        ["se vuelven más suaves/conectados entre puntos y la varianza media baja",
         "se vuelven más rugosos y la varianza media sube",
         "no cambian: `a` solo afecta al gráfico del variograma"], 0,
        "Con `a` pequeño la correlación se pierde rápido: cada punto solo influye en su entorno inmediato y "
        "lejos de los puntos la estimación vuelve hacia un valor medio (con varianza alta). Con `a` grande los "
        "puntos «se ven» entre sí, el campo se interpola de forma más continua y la incertidumbre disminuye. "
        "Ojo: un `a` demasiado grande para la densidad de datos no se puede verificar y es un supuesto fuerte.",
        ["¿Qué rango justificaría tu variograma experimental con tan pocos pares de puntos?",
         "Si `a` > tamaño del dominio, ¿qué estás asumiendo sobre la continuidad del acuífero?"],
        extra="",
    ),
    Escenario(
        2, "Sill y nugget",
        "¿Qué controla el sill y qué controla el nugget?",
        "Sill 1 → 4 (nugget 0) y nugget 0 → 1 (sill 1). Kriging ordinario, `a` y puntos fijos.",
        "Si multiplico el sill por 4 (nugget = 0, todo lo demás igual), en kriging ordinario…",
        ["cambia la estimación de log10(K), pero no la varianza",
         "la estimación NO cambia y la varianza de kriging se multiplica por 4",
         "cambian ambas",
         "no cambia ninguna"], 1,
        "En kriging ordinario los pesos dependen solo de las covarianzas RELATIVAS: multiplicar todo por 4 no "
        "cambia w, así que la estimación es idéntica; la varianza sí escala ×4 porque está en unidades de p². "
        "El nugget, en cambio, cambia la proporción entre C(0) y C(h): reparte los pesos de forma más pareja "
        "(más «promedio»), sube la varianza y, por la convención usada (GSLIB), deja el valor exacto en las "
        "celdas de los puntos pero con un salto hacia sus vecinas.",
        ["¿Por qué el sill no importa para los pesos pero sí para la varianza?",
         "¿El nugget representa error de medición o variabilidad a escala menor que la separación entre puntos?"],
    ),
    Escenario(
        3, "Exponencial vs esférico vs gaussiano",
        "¿Cómo cambia la interpolación al cambiar la forma de la covarianza (mismo `a`)?",
        "Solo el modelo (exponencial, esférico, gaussiano), con a, sill y nugget iguales.",
        "Cerca del origen (h pequeño), ¿qué modelo es el más «suave» (correlación más cercana a 1)?",
        ["exponencial", "esférico", "gaussiano"], 2,
        "Cerca del origen el gaussiano es parabólico (el más suave, ρ(0.25a)=0.94), el exponencial cae en línea "
        "recta (ρ=0.78) y el esférico, cuyo `a` es el rango verdadero, cae aún más rápido al inicio (ρ=0.63). "
        "Un modelo muy suave como el gaussiano puede producir pesos negativos grandes, sobrepasos fuera del "
        "rango de los datos y matrices mal condicionadas. Las tres curvas NO son intercambiables con el mismo `a`: "
        "tienen distinto rango práctico.",
        ["¿Qué modelo elegirías para un acuífero con contactos geológicos abruptos?",
         "¿Por qué el mismo valor de `a` significa cosas distintas según el modelo?"],
        extra="curvas",
    ),
    Escenario(
        4, "Agregar un punto piloto",
        "¿Qué cambia cuando se agrega información en la zona más incierta?",
        "Se agrega un punto en la celda de mayor varianza; tú eliges su valor.",
        "Al agregar un punto en la celda de mayor varianza, el mapa de varianza…",
        ["baja sobre todo cerca del punto nuevo y el mapa cambia principalmente en su entorno",
         "baja por igual en todo el dominio",
         "no cambia: la varianza no depende de dónde están los puntos"], 0,
        "La varianza de kriging depende solo de la GEOMETRÍA de los puntos y del modelo de covarianza (no de los "
        "valores): un punto nuevo reduce la incertidumbre en su vecindad (hasta 0 en su propia celda). La "
        "estimación sí depende del valor del punto nuevo; los pesos de las celdas vecinas se redistribuyen.",
        ["¿Dónde pondrías el siguiente punto para reducir más la varianza media?",
         "¿Reducir la varianza de kriging garantiza que K real esté mejor estimada?"],
        extra="pesos", controles={"p_nuevo": ("Valor del punto nuevo (log10 K)", -1.0, 3.0, 0.3, 0.05)},
    ),
    Escenario(
        5, "Duplicar parval1 de un punto",
        "¿Cómo se propaga un cambio de K en un punto a log10(K) y a K en el mapa?",
        "K de un punto × factor (equivale a sumar log10(factor) a p). Kriging ordinario.",
        "Si duplico K en un punto piloto, log10(K) en esa celda…",
        ["se duplica (por ejemplo de 1.0 a 2.0)", "aumenta en log10(2) ≈ 0.301", "aumenta en 1"], 1,
        "p = log10(K): multiplicar K por 2 SUMA log10(2) = 0.301 a p. Como el kriging es lineal en p, el cambio en "
        "otra celda es w·0.301 (w = peso del punto en esa celda), y en K es un factor 10^(w·0.301): igual a 2 "
        "solo en el punto, menor lejos y MENOR que 1 donde el peso es negativo. Si en cambio se aplicaran los "
        "pesos directamente a K (sin log), la interpolación sería aritmética y no geométrica.",
        ["¿Por qué interpolar en log10(K) y no en K?",
         "¿Qué pasa con K en una celda donde el peso de ese punto es negativo?"],
        extra="razonK", controles={"indice": ("Número de punto (1 = primero de la tabla)", 1, 50, 1, 1),
                                   "factor": ("Factor multiplicador de K", 0.1, 10.0, 2.0, 0.1)},
    ),
    Escenario(
        6, "Anisotropía y orientación",
        "¿Cómo cambia la interpolación si la continuidad es mayor en una dirección?",
        "Razón de anisotropía y azimut del eje mayor (desde el norte, sentido horario).",
        "Con razón 3 y eje mayor a azimut 0° (norte–sur), la correlación llega más lejos en dirección…",
        ["norte–sur (a lo largo del eje mayor)", "este–oeste", "igual en todas las direcciones"], 0,
        "La anisotropía geométrica «estira» la distancia en la dirección menor: h_ef = √(u² + (R·v)²). A lo largo "
        "del eje mayor el rango es `a`; perpendicular, a/R. El mapa de covarianza respecto al objetivo se vuelve "
        "una elipse y los puntos alineados con el eje mayor reciben más peso.",
        ["¿Qué evidencia geológica justificaría una orientación de canales o fracturas?",
         "¿Qué pasa si el azimut elegido está 90° equivocado?"],
        extra="covmaps", controles={"razon": ("Razón de anisotropía (≥ 1)", 1.0, 10.0, 3.0, 0.5),
                                    "azimut": ("Azimut del eje mayor (°)", 0.0, 180.0, 45.0, 5.0)},
    ),
    Escenario(
        7, "Con y sin pesos negativos",
        "¿Qué se gana y qué se pierde al eliminar pesos negativos y renormalizar?",
        "Kriging ordinario con y sin eliminación de pesos negativos (como pyEMU).",
        "Al eliminar los pesos negativos y renormalizar, la varianza de kriging calculada con esos pesos…",
        ["aumenta o queda igual (los pesos ya no son los de mínima varianza)",
         "disminuye en todas las celdas", "no cambia"], 0,
        "Los pesos del kriging ordinario minimizan la varianza entre todos los pesos que suman 1. Cualquier otro "
        "conjunto (por ejemplo, el que resulta de recortar negativos) tiene varianza ≥ la óptima. A cambio, "
        "la estimación pasa a ser un promedio ponderado con pesos ≥ 0, por lo que nunca sale del rango de los "
        "datos (evita sobrepasos). Es un compromiso entre optimalidad estadística y plausibilidad física.",
        ["¿Prefieres un estimador óptimo que se pase del rango de datos o uno robusto que no se pase?",
         "¿Qué efecto tiene el efecto pantalla (screening) en estos pesos negativos?"],
        extra="pesos",
    ),
    Escenario(
        8, "Kriging ordinario vs IDW",
        "¿Qué hace el kriging que no hace el IDW?",
        "IDW (potencia 2) vs kriging ordinario; con agrupamiento opcional de puntos junto a uno existente.",
        "En kriging, los puntos agrupados (casi redundantes) reciben en conjunto…",
        ["el mismo peso total que en IDW", "menos peso total que en IDW (la información redundante se reparte)",
         "más peso total que en IDW"], 1,
        "El IDW solo mira distancias: un grupo de puntos cercanos suma mucho peso aunque aporte información "
        "redundante. El kriging usa también la covarianza ENTRE puntos (matriz C), por lo que reparte el peso "
        "dentro del grupo y reduce la sobrerrepresentación. Además entrega una varianza, cosa que el IDW no hace.",
        ["¿En qué situaciones el IDW y el kriging darían casi lo mismo?",
         "¿Por qué el punto más cercano no siempre tiene el mayor peso en kriging?"],
        extra="pesos", controles={"n_extra": ("Puntos extra agrupados junto a un punto central", 0, 3, 2, 1)},
    ),
    Escenario(
        9, "Mover una observación e identificabilidad",
        "¿A qué parámetros (puntos piloto) es sensible una observación de carga, y cambia eso al moverla?",
        "Posición (fila, columna) de dos observaciones hipotéticas A y B. Requiere MODFLOW 6.",
        "Una sola observación de carga hidráulica…",
        ["alcanza para estimar los 9 valores de los puntos piloto por separado",
         "aporta una sola ecuación: no identifica los 9 parámetros, y al moverla cambia a cuáles es sensible",
         "tiene la misma sensibilidad a todos los puntos piloto sin importar dónde esté"], 1,
        "El Jacobiano de una observación tiene una sola fila (rango 1): con 9 parámetros queda un espacio nulo de "
        "dimensión 8, es decir, infinitas combinaciones de valores reproducen la misma carga. Dónde se ubique "
        "la observación cambia el vector de sensibilidades ∂h/∂p. Esto es una sensibilidad LOCAL por diferencias "
        "finitas (no una prueba formal de identificabilidad) y usa los valores didácticos del panel de flujo.",
        ["¿Cuántas observaciones independientes necesitarías para 9 parámetros?",
         "¿Qué tipo de datos (flujos, cargas en varios puntos, trazadores) ayudarían a identificar mejor?"],
        extra="sensibilidad", requiere_modflow=True,
        controles={"fa": ("Obs. A: fila", 0, 14, 7, 1), "ca": ("Obs. A: columna", 0, 19, 5, 1),
                   "fb": ("Obs. B: fila", 0, 14, 7, 1), "cb": ("Obs. B: columna", 0, 19, 10, 1)},
    ),
    Escenario(
        10, "Discontinuidad geológica: zonas juntas vs separadas",
        "¿Qué pasa con un contacto geológico abrupto si interpolo ambas zonas juntas?",
        "Datos DIDÁCTICOS con dos zonas (oeste: K baja; este: K alta). Juntas vs separadas en x = 500 m.",
        "Al interpolar ambas zonas juntas con un solo variograma, en la frontera x = 500 m el mapa…",
        ["muestra una transición suave que borra el salto real",
         "reproduce el salto abrupto exacto", "queda sin estimar (huecos)"], 0,
        "Un único modelo estacionario supone continuidad espacial: mezcla puntos de ambos lados y el salto se "
        "suaviza. Al separar las zonas, cada lado usa solo sus puntos y aparece un salto en la frontera, pero "
        "eso requiere conocer la frontera (interpretación geológica) y tener suficientes puntos en cada zona. "
        "Los puntos de este escenario son inventados para ilustrar; no son datos del proyecto.",
        ["¿Cómo decidirías dónde trazar la frontera de las zonas?",
         "¿Qué riesgo hay si la frontera dibujada no coincide con la geología real?"],
        extra="perfil", usa_datos_propios=True,
    ),
]


def datos_discontinuidad(g: Grilla) -> pd.DataFrame:
    """Conjunto didáctico: oeste (x<500) K baja, este (x≥500) K alta. Valores inventados."""
    celdas = [(2, 2, 0.4), (7, 6, 0.5), (11, 3, 0.6), (13, 8, 0.5),
              (1, 12, 1.6), (4, 17, 1.7), (8, 13, 1.5), (10, 18, 1.6), (13, 14, 1.7)]
    filas = []
    for k, (f, c, p) in enumerate(celdas, 1):
        x, y = g.centro(f, c)
        filas.append({"nombre": f"D{k}", "x": x, "y": y, "p": p})
    return pd.DataFrame(filas)


# --- Ejecución -------------------------------------------------------------------------------------
def _caso(g, et, puntos, cfg, deriva=None):
    return {"etiqueta": et, "puntos": puntos, "cfg": cfg, "resultado": interpolar_campo(g, puntos, cfg, deriva)}


def _metricas(casos) -> pd.DataFrame:
    filas = []
    for c in casos:
        r = c["resultado"]
        filas.append({"caso": c["etiqueta"], "p̂ mín": float(np.nanmin(r.p)), "p̂ máx": float(np.nanmax(r.p)),
                      "σ² media": float(np.nanmean(r.varianza)) if r.varianza is not None else np.nan,
                      "σ² máx": float(np.nanmax(r.varianza)) if r.varianza is not None else np.nan,
                      "celdas con pesos negativos": r.n_celdas_pesos_neg})
    return pd.DataFrame(filas)


def _ok(base: ConfigInterp, **kw_vario) -> ConfigInterp:
    v = replace(base.vario, **kw_vario) if kw_vario else base.vario
    return ConfigInterp(metodo="ordinario", vario=v)


def ejecutar(esc: Escenario, g: Grilla, puntos: pd.DataFrame, base: ConfigInterp, params: dict,
             deriva=None, flujo=None) -> dict:
    """Devuelve {'casos', 'metricas', 'objetivo', 'extra': {...}, 'avisos': [...]}"""
    P = puntos.reset_index(drop=True)
    out: dict = {"extra": {}, "objetivo": None, "avisos": []}
    n = esc.id
    if n == 1:
        casos = [_caso(g, f"a = {a} m", P, _ok(base, a=float(a))) for a in (100, 350, 700)]
    elif n == 2:
        v0 = {"sill": 1.0, "nugget": 0.0}
        casos = [_caso(g, "sill 1, nugget 0", P, _ok(base, **v0)),
                 _caso(g, "sill 4, nugget 0", P, _ok(base, sill=4.0, nugget=0.0)),
                 _caso(g, "sill 1, nugget 1", P, _ok(base, sill=1.0, nugget=1.0))]
    elif n == 3:
        casos = [_caso(g, m.capitalize(), P, _ok(base, modelo=m)) for m in ("exponencial", "esferico", "gaussiano")]
    elif n == 4:
        b = _caso(g, "Sin punto nuevo", P, _ok(base))
        var = b["resultado"].varianza.copy()
        for _, r in P.iterrows():
            ij = g.celda_de_xy(r.x, r.y)
            if ij:
                var[ij] = -1
        fi, co = np.unravel_index(np.nanargmax(var), var.shape)
        x, y = g.centro(int(fi), int(co))
        P2 = pd.concat([P, pd.DataFrame([{"nombre": "PP_nuevo", "x": x, "y": y, "p": float(params["p_nuevo"])}])],
                       ignore_index=True)
        casos = [b, _caso(g, "Con punto nuevo", P2, _ok(base))]
        out["objetivo"] = (int(fi), int(min(max(co + 2, 0), g.ncols - 1)) if co + 2 < g.ncols else int(co - 2))
        out["avisos"].append(f"El punto nuevo se colocó en la celda de mayor varianza (fila {fi}, columna {co}).")
    elif n == 5:
        i = int(min(max(params["indice"], 1), len(P))) - 1
        P2 = P.copy()
        P2.loc[i, "p"] = P2.loc[i, "p"] + np.log10(params["factor"])
        casos = [_caso(g, "Base", P, _ok(base)),
                 _caso(g, f"K de {P.loc[i, 'nombre']} × {params['factor']:g}", P2, _ok(base))]
        out["extra"]["razonK"] = casos[1]["resultado"].K / casos[0]["resultado"].K
        out["extra"]["delta_p"] = casos[1]["resultado"].p - casos[0]["resultado"].p
        out["extra"]["punto"] = P.loc[i, "nombre"]
        out["extra"]["delta_p_esperado"] = float(np.log10(params["factor"]))
        out["objetivo"] = g.celda_de_xy(P.loc[i, "x"], P.loc[i, "y"])
    elif n == 6:
        a = base.vario.a
        casos = [_caso(g, "Isotrópico", P, _ok(base, razon_anisotropia=1.0, azimut=0.0)),
                 _caso(g, f"R={params['razon']:g}, azimut 0°", P, _ok(base, razon_anisotropia=params["razon"], azimut=0.0)),
                 _caso(g, f"R={params['razon']:g}, azimut {params['azimut']:g}°", P,
                       _ok(base, razon_anisotropia=params["razon"], azimut=params["azimut"]))]
        out["objetivo"] = (g.nfilas // 2, g.ncols // 2)
    elif n == 7:
        casos = [_caso(g, "Con pesos negativos (óptimo)", P, _ok(base)),
                 _caso(g, "Sin pesos negativos (renormalizado)", P,
                       ConfigInterp(metodo="ordinario", vario=base.vario, quitar_negativos=True))]
        XX, YY = g.mallas()
        r = kriging(P[["x", "y"]].to_numpy(float), P["p"].to_numpy(float),
                    np.column_stack([XX.ravel(), YY.ravel()]), base.vario, "ordinario")
        k = int(np.argmin(r.pesos.min(axis=1)))
        out["objetivo"] = (k // g.ncols, k % g.ncols)
        out["extra"]["peso_min"] = float(r.pesos.min())
        out["extra"]["var_aumenta"] = bool((casos[1]["resultado"].varianza - casos[0]["resultado"].varianza
                                            >= -1e-9).all())
    elif n == 8:
        P2 = P.copy()
        k0 = int(np.argmin(np.hypot(P["x"] - P["x"].mean(), P["y"] - P["y"].mean())))
        base_pt = P.loc[k0]
        offs = [(g.dx, 0.0), (0.0, g.dy), (-g.dx, 0.0)]
        extra = [{"nombre": f"{base_pt['nombre']}-{chr(98 + j)}", "x": base_pt["x"] + dx, "y": base_pt["y"] + dy,
                  "p": float(base_pt["p"])} for j, (dx, dy) in enumerate(offs[:int(params["n_extra"])])]
        if extra:
            P2 = pd.concat([P, pd.DataFrame(extra)], ignore_index=True)
        casos = [_caso(g, "IDW (potencia 2)", P2, ConfigInterp(metodo="idw", potencia_idw=2.0)),
                 _caso(g, "Kriging ordinario", P2, _ok(base))]
        f0, c0 = g.celda_de_xy(base_pt["x"], base_pt["y"])
        out["objetivo"] = (f0, min(c0 + 3, g.ncols - 1))
        out["extra"]["grupo"] = [base_pt["nombre"]] + [e["nombre"] for e in extra]
        out["avisos"].append(f"Los puntos extra se agruparon junto a {base_pt['nombre']} con su mismo valor "
                             "(supuesto didáctico, representan información redundante).")
        P = P2
    elif n == 9:
        if flujo is None:
            raise ErrorKriging("Este escenario necesita MODFLOW 6 (ejecutable y FloPy). Instálalos o desactívalo.")
        from .flow import sensibilidades
        cfg = _ok(base)
        casos = [_caso(g, "Base", P, cfg)]

        def construir(pv):
            Q = P.copy()
            Q["p"] = pv
            return interpolar_campo(g, Q, cfg).K

        obs = {f"Obs A (f{int(params['fa'])},c{int(params['ca'])})": (int(params["fa"]), int(params["ca"])),
               f"Obs B (f{int(params['fb'])},c{int(params['cb'])})": (int(params["fb"]), int(params["cb"]))}
        S = sensibilidades(construir, P["nombre"].tolist(), P["p"].to_numpy(float), obs, g,
                           flujo["par"], flujo["exe"])
        out["extra"]["sens"] = pd.DataFrame({"punto": P["nombre"].tolist(), **{k: v for k, v in S.items()}})
    elif n == 10:
        D = datos_discontinuidad(g)
        P = D
        v = base.vario
        casos = [_caso(g, "Zonas juntas", D, ConfigInterp(metodo="ordinario", vario=v)),
                 _caso(g, "Zonas separadas (x = 500 m)", D,
                       ConfigInterp(metodo="ordinario", vario=v, separar_zonas=True, x_corte=500.0))]
        out["extra"]["fila"] = g.nfilas // 2
    else:
        raise ValueError(f"Escenario desconocido: {n}")
    out["casos"] = casos
    out["puntos_usados"] = P
    out["metricas"] = _metricas(casos)
    if out["objetivo"] is not None and esc.extra == "pesos":
        det = []
        for c in casos:
            det.append(detalle_celda(g, c["puntos"], c["cfg"], *out["objetivo"]))
        out["extra"]["detalles"] = det
    return out
