"""Datos: grilla del acuífero, puntos piloto, conversiones K <-> log10(K),
validaciones y lectura/escritura de CSV.

Convenciones (iguales a MODFLOW 6):
  * x crece hacia el este (columnas), y crece hacia el norte.
  * La fila 0 de las matrices es la fila NORTE (arriba en los mapas).
  * Los puntos y celdas se expresan en metros.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- Valores del proyecto (enunciado) ---------------------------------------
P_PROYECTO = [0.8, 1.0, 1.4, 0.6, 1.1, 1.3, 0.9, 1.2, 1.5]  # p = log10(K), K en m/día
# SUPUESTO DIDÁCTICO: el enunciado no entrega las coordenadas de los 9 puntos
# (solo dice que están en centros de celda). Se usa una malla regular 3x3
# y se asignan los valores en orden (fila por fila, de norte a sur).
# Sustitúyela cargando tus coordenadas reales en la pestaña «Puntos piloto».
FILAS_PROYECTO = [2, 7, 12]
COLS_PROYECTO = [3, 10, 16]

ZONA_UNICA = "única"
ZONAS_DIVIDIDAS = ("Oeste", "Este")


@dataclass(frozen=True)
class Grilla:
    """Grilla estructurada de una capa (como DIS en MODFLOW 6)."""

    nfilas: int = 15
    ncols: int = 20
    dx: float = 50.0  # ancho de columna (m)
    dy: float = 50.0  # alto de fila (m)

    @property
    def Lx(self) -> float:
        return self.ncols * self.dx

    @property
    def Ly(self) -> float:
        return self.nfilas * self.dy

    @property
    def n(self) -> int:
        return self.nfilas * self.ncols

    @property
    def x_centros(self) -> np.ndarray:
        return (np.arange(self.ncols) + 0.5) * self.dx

    @property
    def y_centros_filas(self) -> np.ndarray:
        """y del centro de cada fila; la fila 0 es la más al norte."""
        return self.Ly - (np.arange(self.nfilas) + 0.5) * self.dy

    def mallas(self) -> tuple[np.ndarray, np.ndarray]:
        """XX, YY con forma (nfilas, ncols)."""
        return np.meshgrid(self.x_centros, self.y_centros_filas)

    def centro(self, fila: int, col: int) -> tuple[float, float]:
        return float((col + 0.5) * self.dx), float(self.Ly - (fila + 0.5) * self.dy)

    def celda_de_xy(self, x: float, y: float) -> tuple[int, int] | None:
        """Fila/columna de la celda que contiene (x, y); None si está fuera."""
        if not (0 <= x <= self.Lx and 0 <= y <= self.Ly):
            return None
        col = min(int(x // self.dx), self.ncols - 1)
        fila = min(int((self.Ly - y) // self.dy), self.nfilas - 1)
        return fila, col


# --- Conversiones -----------------------------------------------------------
def p_a_K(p):
    """K [m/día] = 10**p, con p = log10(K [m/día])."""
    return np.power(10.0, np.asarray(p, dtype=float))


def K_a_p(K):
    """p = log10(K [m/día]). K debe ser estrictamente positiva."""
    K = np.asarray(K, dtype=float)
    if np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("K debe ser un número positivo (m/día): log10 no está definido para K ≤ 0.")
    return np.log10(K)


# --- Puntos piloto ----------------------------------------------------------
COLUMNAS = ["nombre", "x", "y", "p"]


def puntos_proyecto(grilla: Grilla | None = None) -> pd.DataFrame:
    """Escenario inicial: 9 puntos en centros de celda (disposición 3x3 supuesta)."""
    g = grilla or Grilla()
    filas = []
    k = 0
    for f in FILAS_PROYECTO:
        for c in COLS_PROYECTO:
            x, y = g.centro(f, c)
            filas.append({"nombre": f"PP{k + 1}", "x": x, "y": y, "p": P_PROYECTO[k]})
            k += 1
    return pd.DataFrame(filas, columns=COLUMNAS)


def zona_de_x(x, x_corte: float):
    """Zona didáctica según una línea vertical (falla ficticia) en x = x_corte."""
    return np.where(np.asarray(x, dtype=float) < x_corte, ZONAS_DIVIDIDAS[0], ZONAS_DIVIDIDAS[1])


def zonas_celdas(grilla: Grilla, x_corte: float) -> np.ndarray:
    XX, _ = grilla.mallas()
    return zona_de_x(XX, x_corte)


def tabla_completa(puntos: pd.DataFrame, separar_zonas: bool, x_corte: float) -> pd.DataFrame:
    """Tabla con p, K y zona para mostrar al estudiante."""
    t = puntos.copy()
    t["K (m/día)"] = p_a_K(t["p"])
    t["zona"] = zona_de_x(t["x"], x_corte) if separar_zonas else ZONA_UNICA
    return t.rename(columns={"p": "p = log10(K)"})


def validar_puntos(puntos: pd.DataFrame, grilla: Grilla, separar_zonas=False,
                   x_corte: float = 500.0) -> list[tuple[str, str]]:
    """Devuelve [(nivel, mensaje)], con nivel en {'error','advertencia','info'}."""
    msgs: list[tuple[str, str]] = []
    if puntos is None or len(puntos) == 0:
        return [("error", "No hay puntos piloto. Agrega al menos uno.")]
    for col in COLUMNAS:
        if col not in puntos.columns:
            return [("error", f"Falta la columna «{col}».")]
    if puntos[["x", "y", "p"]].isna().any().any():
        msgs.append(("error", "Hay celdas vacías en x, y o valor. Complétalas o elimina esas filas."))
        return msgs
    if not np.isfinite(puntos[["x", "y", "p"]].to_numpy(float)).all():
        msgs.append(("error", "Hay valores no finitos (inf/NaN) en los puntos."))
        return msgs
    if puntos["nombre"].isna().any() or (puntos["nombre"].astype(str).str.strip() == "").any():
        msgs.append(("advertencia", "Hay puntos sin nombre."))
    if puntos["nombre"].duplicated().any():
        msgs.append(("advertencia", "Hay nombres repetidos; usa nombres únicos para no confundir los pesos."))
    dup = puntos.duplicated(subset=["x", "y"], keep=False)
    if dup.any():
        nombres = ", ".join(map(str, puntos.loc[dup, "nombre"]))
        msgs.append(("error", f"Coordenadas duplicadas ({nombres}): la matriz de kriging sería singular. "
                              "Mueve o elimina uno de los puntos."))
    fuera = ~((puntos["x"].between(0, grilla.Lx)) & (puntos["y"].between(0, grilla.Ly)))
    if fuera.any():
        nombres = ", ".join(map(str, puntos.loc[fuera, "nombre"]))
        msgs.append(("advertencia", f"Fuera del dominio [0,{grilla.Lx:.0f}]×[0,{grilla.Ly:.0f}] m: {nombres}. "
                                    "El kriging funciona, pero el punto queda fuera del mapa."))
    if separar_zonas:
        z = zona_de_x(puntos["x"], x_corte)
        for nombre in ZONAS_DIVIDIDAS:
            n = int((z == nombre).sum())
            if n == 0:
                msgs.append(("advertencia", f"La zona «{nombre}» no contiene puntos: sus celdas no se pueden estimar."))
            elif n < 3:
                msgs.append(("info", f"La zona «{nombre}» tiene solo {n} punto(s): estimación muy poco informada."))
    if len(puntos) < 3:
        msgs.append(("info", "Con menos de 3 puntos el kriging universal con tendencia plana no es posible."))
    return msgs


# --- CSV ---------------------------------------------------------------------
_ALIAS = {
    "nombre": ["nombre", "name", "id", "punto"],
    "x": ["x", "este", "easting"],
    "y": ["y", "norte", "northing"],
    "p": ["p", "log10k", "logk", "log10(k)", "log10_k", "p_log10k"],
    "K": ["k", "parval1", "k_m_dia", "k (m/día)", "k (m/dia)"],
}


def _buscar(cols: dict[str, str], clave: str):
    for a in _ALIAS[clave]:
        if a in cols:
            return cols[a]
    return None


def leer_csv(origen) -> pd.DataFrame:
    """Lee un CSV (ruta o archivo subido). Acepta columna p (=log10 K) o K (m/día)."""
    try:
        if isinstance(origen, (str, Path)):
            ruta = Path(origen)
            if not ruta.is_file():
                raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
            df = pd.read_csv(ruta, sep=None, engine="python")
        else:
            df = pd.read_csv(origen, sep=None, engine="python")
    except FileNotFoundError:
        raise
    except Exception as e:  # formato ilegible
        raise ValueError(f"No se pudo leer el CSV: {e}") from e
    cols = {str(c).strip().lower(): c for c in df.columns}
    cx, cy = _buscar(cols, "x"), _buscar(cols, "y")
    if cx is None or cy is None:
        raise ValueError("El CSV debe tener columnas x e y (en metros). "
                         f"Columnas encontradas: {list(df.columns)}")
    cp, cK, cn = _buscar(cols, "p"), _buscar(cols, "K"), _buscar(cols, "nombre")
    if cp is None and cK is None:
        raise ValueError("El CSV debe tener una columna «p» (log10 K) o «K» (m/día).")
    out = pd.DataFrame({"x": pd.to_numeric(df[cx], errors="coerce"),
                        "y": pd.to_numeric(df[cy], errors="coerce")})
    if cp is not None:
        out["p"] = pd.to_numeric(df[cp], errors="coerce")
    else:
        out["p"] = K_a_p(pd.to_numeric(df[cK], errors="coerce").to_numpy())
    out.insert(0, "nombre", df[cn].astype(str) if cn else [f"PP{i + 1}" for i in range(len(df))])
    out = out.dropna(subset=["x", "y", "p"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("El CSV no contiene filas válidas (x, y y valor numéricos).")
    return out[COLUMNAS]


def a_csv(puntos: pd.DataFrame) -> str:
    t = puntos.copy()
    t["K_m_dia"] = p_a_K(t["p"])
    return t.rename(columns={"p": "p_log10K"}).to_csv(index=False)


def leer_deriva(origen, grilla: Grilla) -> np.ndarray:
    """Variable auxiliar (deriva externa) en la grilla: CSV de nfilas×ncols o 1 columna de n valores."""
    arr = pd.read_csv(origen, header=None).to_numpy(float)
    if arr.size == grilla.n:
        arr = arr.reshape(grilla.nfilas, grilla.ncols)
    if arr.shape != (grilla.nfilas, grilla.ncols):
        raise ValueError(f"La deriva debe tener {grilla.nfilas}×{grilla.ncols} valores "
                         f"(fila 0 = norte); se recibió forma {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("La deriva contiene valores no finitos.")
    return arr


def deriva_didactica(grilla: Grilla) -> np.ndarray:
    """Variable auxiliar SINTÉTICA (supuesto didáctico, no es un dato real):
    gradiente suave este-oeste con una ondulación, que podría representar p. ej. una
    cota del terreno. Úsala solo para entender la mecánica del kriging con deriva externa."""
    XX, YY = grilla.mallas()
    return 100.0 * XX / grilla.Lx + 10.0 * np.sin(2 * np.pi * YY / grilla.Ly)
