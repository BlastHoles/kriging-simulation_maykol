"""Interpolación espacial: IDW y kriging (simple, ordinario, universal, deriva externa).

Todas las fórmulas se resuelven con el sistema aumentado

        [ C   F ] [ w ]   [ c  ]
        [ Fᵀ  0 ] [ μ ] = [ f₀ ]

donde C es la covarianza entre puntos de datos, c entre datos y el objetivo, F contiene
las funciones de la tendencia evaluadas en los datos y f₀ en el objetivo.
  * Ordinario:   F = 1 (una columna de unos)          → Σw = 1
  * Universal:   F = [1, x, y, ...]                    → tendencia polinómica
  * Deriva ext.: F = [1, s(x)] con s una variable auxiliar conocida en TODA la grilla
  * Simple:      sin F ni μ; media conocida m:  p̂ = m + Σ wᵢ (pᵢ − m)

Varianza de kriging:  σ² = C(0) − wᵀc − μᵀf₀.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Grilla, p_a_K, zona_de_x, zonas_celdas, ZONA_UNICA, ZONAS_DIVIDIDAS
from .variogram import Variograma

METODOS = {
    "idw": "IDW (inverso de la distancia)",
    "simple": "Kriging simple",
    "ordinario": "Kriging ordinario",
    "universal": "Kriging universal",
    "ked": "Kriging con deriva externa",
}
TENDENCIAS = {"constante": "Constante (≡ ordinario)", "plano": "Plano (1, x, y)",
              "cuadratica": "Cuadrática (1, x, y, x², xy, y²)"}

COND_AVISO = 1e8
COND_ERROR = 1e11
PESO_AVISO = 10.0      # |w| máximo a partir del cual se avisa de sobrepasos extremos
PESO_ERROR = 1e3       # |w| máximo a partir del cual se considera inestable


class ErrorKriging(Exception):
    """Error de kriging con mensaje comprensible para el estudiante."""


@dataclass(frozen=True)
class ConfigInterp:
    metodo: str = "ordinario"
    vario: Variograma = field(default_factory=Variograma)
    potencia_idw: float = 2.0
    media_simple: float | None = None      # None → media muestral (se avisa)
    tendencia: str = "plano"               # solo universal
    quitar_negativos: bool = False         # solo ordinario
    separar_zonas: bool = False
    x_corte: float = 500.0

    def etiqueta(self) -> str:
        if self.metodo == "idw":
            return f"IDW (potencia {self.potencia_idw:g})"
        v = self.vario
        extra = ""
        if self.metodo == "universal":
            extra = f", tendencia {self.tendencia}"
        if self.metodo == "ordinario" and self.quitar_negativos:
            extra = ", sin pesos negativos"
        return f"{METODOS[self.metodo]} ({v.modelo}, a={v.a:g} m{extra})"


@dataclass
class ResultadoKriging:
    estimacion: np.ndarray            # (nt,)
    varianza: np.ndarray | None       # (nt,) o None (IDW)
    pesos: np.ndarray                 # (nt, n)
    multiplicadores: np.ndarray       # (nt, m)  (m puede ser 0)
    C: np.ndarray | None              # (n, n)
    c: np.ndarray | None              # (n, nt)
    cond: float = np.nan
    avisos: list[str] = field(default_factory=list)
    n_celdas_pesos_neg: int = 0
    n_varianza_recortada: int = 0
    media_usada: float | None = None


# --- IDW ----------------------------------------------------------------------
def idw(xy, z, xy_obj, potencia: float = 2.0) -> ResultadoKriging:
    xy, z, xy_obj = np.asarray(xy, float), np.asarray(z, float), np.asarray(xy_obj, float)
    if potencia <= 0:
        raise ErrorKriging("La potencia del IDW debe ser > 0.")
    d = np.hypot(xy[:, 0][:, None] - xy_obj[:, 0][None, :], xy[:, 1][:, None] - xy_obj[:, 1][None, :])
    coincide = d < 1e-9
    w = 1.0 / np.maximum(d, 1e-9) ** potencia
    w[:, coincide.any(axis=0)] = coincide[:, coincide.any(axis=0)].astype(float)
    w = w / w.sum(axis=0, keepdims=True)
    return ResultadoKriging(estimacion=w.T @ z, varianza=None, pesos=w.T,
                            multiplicadores=np.zeros((xy_obj.shape[0], 0)), C=None, c=None)


# --- Tendencias -------------------------------------------------------------------
def base_tendencia(xy, metodo: str, tendencia: str = "plano", deriva=None, escala: float = 1000.0):
    """Matriz F (n, m) con las funciones de la tendencia. Coordenadas reescaladas para
    mejorar el condicionamiento (no cambia la estimación)."""
    xy = np.asarray(xy, float)
    n = xy.shape[0]
    if metodo == "ordinario":
        return np.ones((n, 1))
    if metodo == "simple":
        return np.zeros((n, 0))
    x, y = xy[:, 0] / escala, xy[:, 1] / escala
    if metodo == "ked":
        if deriva is None:
            raise ErrorKriging("El kriging con deriva externa necesita una variable auxiliar "
                               "conocida en los puntos y en toda la grilla. Cárgala o activa la didáctica.")
        return np.column_stack([np.ones(n), np.asarray(deriva, float)])
    if tendencia == "constante":
        return np.ones((n, 1))
    if tendencia == "plano":
        return np.column_stack([np.ones(n), x, y])
    if tendencia == "cuadratica":
        return np.column_stack([np.ones(n), x, y, x * x, x * y, y * y])
    raise ErrorKriging(f"Tendencia desconocida: {tendencia}")


# --- Kriging --------------------------------------------------------------------------
def kriging(xy_datos, z, xy_obj, vario: Variograma, metodo: str = "ordinario",
            media: float | None = None, tendencia: str = "plano",
            deriva_datos=None, deriva_obj=None, quitar_negativos: bool = False) -> ResultadoKriging:
    xy = np.asarray(xy_datos, float)
    z = np.asarray(z, float)
    xo = np.asarray(xy_obj, float)
    n, nt = xy.shape[0], xo.shape[0]
    avisos: list[str] = []
    if n == 0:
        raise ErrorKriging("No hay puntos de datos para interpolar.")
    if not (np.isfinite(xy).all() and np.isfinite(z).all()):
        raise ErrorKriging("Hay valores no finitos en los datos.")
    if len(np.unique(xy, axis=0)) < n:
        raise ErrorKriging("Hay puntos con coordenadas duplicadas: la matriz de covarianza sería singular.")

    C = vario.covarianza(xy[:, 0][:, None] - xy[:, 0][None, :], xy[:, 1][:, None] - xy[:, 1][None, :])
    c = vario.covarianza(xy[:, 0][:, None] - xo[:, 0][None, :], xy[:, 1][:, None] - xo[:, 1][None, :])

    F = base_tendencia(xy, metodo, tendencia, deriva_datos)
    m = F.shape[1]
    if metodo == "simple":
        f0 = np.zeros((0, nt))
    elif metodo == "ked":
        if deriva_obj is None:
            raise ErrorKriging("Falta la deriva externa en los objetivos.")
        f0 = np.vstack([np.ones(nt), np.asarray(deriva_obj, float)])
    else:
        f0 = base_tendencia(xo, metodo, tendencia).T
    if m > 0:
        if n < m or np.linalg.matrix_rank(F) < m:
            raise ErrorKriging(
                f"Con {n} punto(s) no se puede ajustar una tendencia de {m} coeficientes: se necesitan "
                f"al menos {m} puntos y que no estén alineados (o, en deriva externa, que la variable "
                "auxiliar varíe entre los puntos). Usa una tendencia más simple o agrega puntos.")

    A = np.zeros((n + m, n + m))
    A[:n, :n] = C
    A[:n, n:] = F
    A[n:, :n] = F.T
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = np.inf
    if not np.isfinite(cond) or cond > COND_ERROR:
        raise ErrorKriging(
            f"La matriz de kriging es singular o casi singular (número de condición ≈ {cond:.1e}). "
            "Causas típicas: puntos casi superpuestos, modelo gaussiano con rango muy grande respecto a "
            "la separación entre puntos, o tendencia demasiado flexible. Prueba un nugget pequeño "
            "(p. ej. 1e-4·sill), otro modelo o un rango menor.")
    if cond > COND_AVISO:
        avisos.append(f"Matriz mal condicionada (cond ≈ {cond:.1e}): los pesos pueden ser sensibles al redondeo.")

    rhs = np.vstack([c, f0])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as e:
        raise ErrorKriging(f"No se pudo resolver el sistema de kriging: {e}") from e
    w, mu = sol[:n], sol[n:]
    wmax = float(np.abs(w).max())
    if not np.isfinite(wmax) or wmax > PESO_ERROR:
        raise ErrorKriging(
            f"Los pesos de kriging son numéricamente inestables (|w| máx ≈ {wmax:.1e}): la matriz está casi "
            "singular (puntos casi superpuestos o modelo demasiado suave, como el gaussiano con rango grande). "
            "Prueba un nugget pequeño (p. ej. 1e-4·sill), el modelo exponencial/esférico, o separa los puntos.")
    if wmax > PESO_AVISO:
        avisos.append(f"Pesos de magnitud extrema (|w| máx ≈ {wmax:.1f}): la estimación puede sobrepasar mucho el "
                      "rango de los datos.")

    media_usada = None
    if metodo == "simple":
        media_usada = float(media) if media is not None else float(z.mean())
        if media is None:
            avisos.append("Kriging simple: se usó la media muestral como si fuera conocida (supuesto didáctico).")
        est = media_usada + w.T @ (z - media_usada)
    else:
        est = w.T @ z
    var = vario.c0 - np.einsum("ij,ij->j", w, c) - (np.einsum("kj,kj->j", mu, f0) if m else 0.0)

    n_neg = int((w < -1e-6).any(axis=0).sum())
    if quitar_negativos and metodo == "ordinario":
        w2 = np.clip(w, 0.0, None)
        w2 = w2 / w2.sum(axis=0, keepdims=True)
        w = w2
        est = w.T @ z
        # MSE de pesos arbitrarios con Σw = 1:  σ² = C(0) − 2 wᵀc + wᵀ C w
        var = vario.c0 - 2 * np.einsum("ij,ij->j", w, c) + np.einsum("ij,ij->j", w, C @ w)
        mu = np.full_like(mu, np.nan)
    elif quitar_negativos:
        avisos.append("La eliminación de pesos negativos solo se aplica al kriging ordinario; se ignoró.")

    tol = 1e-9 * vario.c0
    n_rec = int((var < 0).sum())
    if n_rec:
        if (var < -tol).any():
            avisos.append(f"Varianza claramente negativa (mín {var.min():.2e}): posible inestabilidad numérica.")
        var = np.clip(var, 0.0, None)

    return ResultadoKriging(estimacion=est, varianza=var, pesos=w.T, multiplicadores=mu.T, C=C, c=c,
                            cond=cond, avisos=avisos, n_celdas_pesos_neg=n_neg,
                            n_varianza_recortada=n_rec, media_usada=media_usada)


# --- Dispatcher y campos ----------------------------------------------------------------
def _resolver(cfg: ConfigInterp, xy, z, xy_obj, deriva_datos=None, deriva_obj=None) -> ResultadoKriging:
    if cfg.metodo == "idw":
        return idw(xy, z, xy_obj, cfg.potencia_idw)
    return kriging(xy, z, xy_obj, cfg.vario, cfg.metodo, cfg.media_simple, cfg.tendencia,
                   deriva_datos, deriva_obj, cfg.quitar_negativos)


def _deriva_en_puntos(grilla: Grilla, deriva_grilla, puntos: pd.DataFrame):
    """Deriva muestreada en la celda que contiene a cada punto (aproximación documentada)."""
    if deriva_grilla is None:
        return None
    vals = []
    for x, y in zip(puntos["x"], puntos["y"]):
        ij = grilla.celda_de_xy(x, y)
        if ij is None:
            raise ErrorKriging("Un punto cae fuera del dominio: no se puede muestrear la deriva externa.")
        vals.append(deriva_grilla[ij])
    return np.array(vals)


@dataclass
class ResultadoCampo:
    p: np.ndarray                    # (nfilas, ncols) log10(K)
    K: np.ndarray                    # m/día
    varianza: np.ndarray | None
    avisos: list[str]
    cfg: ConfigInterp
    n_celdas_pesos_neg: int = 0
    n_varianza_recortada: int = 0


def interpolar_campo(grilla: Grilla, puntos: pd.DataFrame, cfg: ConfigInterp,
                     deriva_grilla=None) -> ResultadoCampo:
    """Estima p = log10(K) en todas las celdas (y K = 10**p)."""
    XX, YY = grilla.mallas()
    p_campo = np.full(XX.shape, np.nan)
    var_campo = None if cfg.metodo == "idw" else np.full(XX.shape, np.nan)
    avisos: list[str] = []
    n_neg = n_rec = 0

    if cfg.separar_zonas:
        zona_cel = zonas_celdas(grilla, cfg.x_corte)
        zona_pts = zona_de_x(puntos["x"].to_numpy(), cfg.x_corte)
        grupos = [(zn, zona_pts == zn, zona_cel == zn) for zn in ZONAS_DIVIDIDAS]
    else:
        grupos = [(ZONA_UNICA, np.ones(len(puntos), bool), np.ones(XX.shape, bool))]

    for nombre_zona, mp, mc in grupos:
        if not mc.any():
            continue
        if not mp.any():
            avisos.append(f"La zona «{nombre_zona}» no tiene puntos: sus celdas quedan sin estimar (NaN).")
            continue
        sub = puntos[mp]
        xy = sub[["x", "y"]].to_numpy(float)
        obj = np.column_stack([XX[mc], YY[mc]])
        dd = _deriva_en_puntos(grilla, deriva_grilla, sub) if cfg.metodo == "ked" else None
        do = deriva_grilla[mc] if (cfg.metodo == "ked" and deriva_grilla is not None) else None
        try:
            r = _resolver(cfg, xy, sub["p"].to_numpy(float), obj, dd, do)
        except ErrorKriging as e:
            if cfg.separar_zonas:
                avisos.append(f"Zona «{nombre_zona}»: {e}")
                continue
            raise
        p_campo[mc] = r.estimacion
        if var_campo is not None:
            var_campo[mc] = r.varianza
        avisos += [a for a in r.avisos if a not in avisos]
        n_neg += r.n_celdas_pesos_neg
        n_rec += r.n_varianza_recortada

    if not np.isfinite(p_campo[~np.isnan(p_campo)]).all():
        raise ErrorKriging("La interpolación produjo valores no finitos.")
    return ResultadoCampo(p=p_campo, K=p_a_K(p_campo), varianza=var_campo, avisos=avisos, cfg=cfg,
                          n_celdas_pesos_neg=n_neg, n_varianza_recortada=n_rec)


# --- Pesos de una celda -------------------------------------------------------------------
@dataclass
class DetalleCelda:
    tabla: pd.DataFrame
    C: np.ndarray | None
    nombres: list[str]
    multiplicadores: np.ndarray
    suma_pesos: float
    p_hat: float
    K_hat: float
    varianza: float | None
    tabla_idw: pd.DataFrame
    avisos: list[str]
    c0: float


def detalle_celda(grilla: Grilla, puntos: pd.DataFrame, cfg: ConfigInterp, fila: int, col: int,
                  deriva_grilla=None, potencia_idw: float | None = None) -> DetalleCelda:
    x0, y0 = grilla.centro(fila, col)
    pts = puntos
    if cfg.separar_zonas:
        zona_obj = zona_de_x(np.array([x0]), cfg.x_corte)[0]
        pts = puntos[zona_de_x(puntos["x"].to_numpy(), cfg.x_corte) == zona_obj]
        if len(pts) == 0:
            raise ErrorKriging(f"La zona «{zona_obj}» de la celda objetivo no tiene puntos.")
    xy = pts[["x", "y"]].to_numpy(float)
    z = pts["p"].to_numpy(float)
    obj = np.array([[x0, y0]])
    dd = _deriva_en_puntos(grilla, deriva_grilla, pts) if cfg.metodo == "ked" else None
    do = np.array([deriva_grilla[fila, col]]) if (cfg.metodo == "ked" and deriva_grilla is not None) else None
    r = _resolver(cfg, xy, z, obj, dd, do)
    w = r.pesos[0]
    d = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
    tabla = pd.DataFrame({"punto": pts["nombre"].astype(str).to_numpy(), "x (m)": xy[:, 0], "y (m)": xy[:, 1],
                          "p = log10(K)": z, "distancia al objetivo (m)": d})
    if r.c is not None:
        tabla["C(punto, objetivo)"] = r.c[:, 0]
    tabla["peso w"] = w
    tabla["contribución w·p"] = w * z
    r_idw = idw(xy, z, obj, potencia_idw if potencia_idw is not None else cfg.potencia_idw)
    tabla_idw = pd.DataFrame({"punto": tabla["punto"], "peso IDW": r_idw.pesos[0]})
    p_hat = float(r.estimacion[0])
    if cfg.metodo == "simple":
        p_hat = float(r.media_usada + (w * (z - r.media_usada)).sum())
    return DetalleCelda(tabla=tabla, C=r.C, nombres=tabla["punto"].tolist(), multiplicadores=r.multiplicadores[0],
                        suma_pesos=float(w.sum()), p_hat=p_hat, K_hat=float(p_a_K(p_hat)),
                        varianza=None if r.varianza is None else float(r.varianza[0]),
                        tabla_idw=tabla_idw, avisos=r.avisos, c0=cfg.vario.c0)


# --- Variograma experimental ------------------------------------------------------------------
def variograma_experimental(xy, z, n_lags: int = 8, lag_max: float | None = None) -> pd.DataFrame:
    """Estimador clásico de Matheron, omnidireccional: γ(h) = 1/(2N) Σ (zᵢ − zⱼ)²."""
    xy, z = np.asarray(xy, float), np.asarray(z, float)
    n = len(z)
    if n < 3:
        return pd.DataFrame(columns=["h (m)", "gamma", "pares"])
    i, j = np.triu_indices(n, 1)
    h = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
    g = 0.5 * (z[i] - z[j]) ** 2
    lag_max = lag_max or h.max() / 2
    bordes = np.linspace(0, lag_max, n_lags + 1)
    filas = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (h > a) & (h <= b)
        if m.any():
            filas.append({"h (m)": float(h[m].mean()), "gamma": float(g[m].mean()), "pares": int(m.sum())})
    return pd.DataFrame(filas, columns=["h (m)", "gamma", "pares"])


# --- Ejemplo numérico mínimo (introducción) ----------------------------------------------------------
def ejemplo_minimo(vario: Variograma | None = None):
    """3 puntos alineados y un objetivo: permite mostrar cada paso del kriging ordinario."""
    v = vario or Variograma(modelo="exponencial", sill=1.0, nugget=0.0, a=350.0)
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    z = np.array([1.0, 1.4, 0.8])
    obj = np.array([[150.0, 0.0]])
    r = kriging(xy, z, obj, v, "ordinario")
    return v, xy, z, obj, r
