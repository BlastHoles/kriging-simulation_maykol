"""Comparación con los factores de kriging ordinario de pyEMU (opcional).

Hechos verificados sobre pyEMU 1.7.0 (inspeccionando su código fuente):
  * ExpVario/GauVario/SphVario usan  C(h) = contribution · {exp(-h/a), exp(-(h/a)²), esférico(h/a)}.
    Es decir, a = e-folding (exp), escala (gauss), rango (esf): igual que este paquete.
  * `bearing` es el azimut (grados desde el norte, sentido horario) del eje mayor y
    `anisotropy` multiplica la distancia perpendicular: razón a_mayor/a_menor.
  * OrdinaryKrige.calc_factors tiene remove_negative_factors=True por defecto: elimina los
    factores negativos y renormaliza (Deutsch, 1996). OJO: err_var se calcula ANTES de eliminarlos.
  * pyEMU NO transforma: los factores se aplican a lo que le des (K física o log10 K).
  * HALLAZGO (comprobado numéricamente aquí): pyEMU resuelve el mismo sistema aumentado
    [C 1; 1ᵀ 0][w; μ] = [c; 1], pero calcula  err_var = sill + μ − Σ wᵢcᵢ,  mientras que la
    expresión estándar es  σ² = C(0) − Σ wᵢcᵢ − μ.  Ambas difieren en 2μ. La expresión estándar
    coincide con el error cuadrático medio exacto  C(0) − 2wᵀc + wᵀCw  (verificado), por eso este
    paquete usa esa. Los PESOS sí coinciden con los de pyEMU a precisión de máquina.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from .data import p_a_K
from .interpolation import kriging
from .variogram import Variograma


def pyemu_disponible() -> tuple[bool, str]:
    try:
        import pyemu  # noqa: F401
        return True, getattr(pyemu, "__version__", "?")
    except Exception as e:  # ImportError u otros problemas de dependencias
        return False, str(e)


def factores_pyemu(puntos: pd.DataFrame, v: Variograma, xo: float, yo: float, quitar_negativos: bool):
    """Calcula los factores de pyEMU para un objetivo. Devuelve (DataFrame, err_var, info)."""
    import pyemu
    from pyemu.utils import geostats as g

    clase = {"exponencial": g.ExpVario, "gaussiano": g.GauVario, "esferico": g.SphVario}[v.modelo]
    vario = clase(contribution=v.sill, a=v.a, anisotropy=v.razon_anisotropia, bearing=v.azimut)
    gs = g.GeoStruct(nugget=v.nugget, variograms=[vario])
    # pyEMU pasa los nombres a minúsculas internamente: usamos identificadores propios y luego
    # los traducimos a los nombres originales del estudiante.
    ids = [f"p{i:04d}" for i in range(len(puntos))]
    mapa = dict(zip(ids, puntos["nombre"].astype(str)))
    pd_ = pd.DataFrame({"name": ids, "x": puntos["x"].to_numpy(float), "y": puntos["y"].to_numpy(float)})
    ok = g.OrdinaryKrige(gs, pd_)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_original = ok.calc_factors([float(xo)], [float(yo)], minpts_interp=1, maxpts_interp=len(pd_),
                                      search_radius=1e10, remove_negative_factors=quitar_negativos)
    fila = ok.interp_data.iloc[0]
    nombres = [mapa[n] for n in fila["inames"]]
    factores = np.asarray(fila["ifacts"], float)
    err_var = float(np.asarray(df_original["err_var"].iloc[0]))
    out = pd.DataFrame({"punto": nombres, "peso pyEMU": factores})
    return out, err_var, {"version": pyemu.__version__}


def comparar(puntos: pd.DataFrame, v: Variograma, xo: float, yo: float, quitar_negativos: bool):
    """Tabla de pesos propios vs pyEMU + diagnóstico en texto."""
    res = kriging(puntos[["x", "y"]].to_numpy(float), puntos["p"].to_numpy(float), np.array([[xo, yo]]), v,
                  "ordinario", quitar_negativos=quitar_negativos)
    propio = pd.DataFrame({"punto": puntos["nombre"].astype(str).to_numpy(), "peso propio": res.pesos[0]})
    # pesos propios sin quitar negativos, solo para saber si había alguno
    res_crudo = kriging(puntos[["x", "y"]].to_numpy(float), puntos["p"].to_numpy(float), np.array([[xo, yo]]), v,
                        "ordinario", quitar_negativos=False)
    py, err_var, info = factores_pyemu(puntos, v, xo, yo, quitar_negativos)
    tabla = propio.merge(py, on="punto", how="left").fillna({"peso pyEMU": 0.0})
    tabla["diferencia"] = tabla["peso propio"] - tabla["peso pyEMU"]
    z = puntos["p"].to_numpy(float)
    p_prop = float(res.estimacion[0])
    p_py = float((tabla["peso pyEMU"].to_numpy() * z).sum())
    K_prop_log = float(p_a_K(p_prop))
    K_py_log = float(p_a_K(p_py))
    K_py_lineal = float((tabla["peso pyEMU"].to_numpy() * p_a_K(z)).sum())
    neg_crudos = res_crudo.pesos[0] < -1e-9
    diag = {
        "max_dif": float(np.abs(tabla["diferencia"]).max()),
        "hubo_negativos": bool(neg_crudos.any()),
        "puntos_negativos": puntos["nombre"].astype(str).to_numpy()[neg_crudos].tolist(),
        "p_propio": p_prop, "p_pyemu": p_py,
        "K_propio": K_prop_log, "K_pyemu_log": K_py_log, "K_pyemu_lineal": K_py_lineal,
        "var_propia": float(res.varianza[0]), "err_var_pyemu": err_var,
        "var_propia_cruda": float(res_crudo.varianza[0]),
        "mu_propio": float(res_crudo.multiplicadores[0, 0]),
        # reproducimos la fórmula de pyEMU con nuestros pesos crudos: sill + μ − Σ w c
        "err_var_pyemu_reproducida": float(v.c0 + res_crudo.multiplicadores[0, 0]
                                           - (res_crudo.pesos[0] * res_crudo.c[:, 0]).sum()),
        "version": info["version"],
        "suma_propia": float(res.pesos[0].sum()), "suma_pyemu": float(tabla["peso pyEMU"].sum()),
    }
    return tabla, diag
