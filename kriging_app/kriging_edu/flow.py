"""Panel hidrogeológico opcional: usa el campo de K interpolado como entrada a MODFLOW 6 (FloPy).

MODELO CONCEPTUAL (valores DIDÁCTICOS, no del proyecto original):
  * Una capa CONFINADA (icelltype=0) de espesor b, régimen estacionario.
  * Cargas constantes (CHD) en la columna oeste y la columna este; fronteras norte/sur impermeables.
  * Recarga uniforme R (m/día) sobre toda la capa y un pozo de bombeo (Q < 0 extrae, m³/día).
  * Transmisividad T = K·b (m²/día). Carga h en metros. Abatimiento = h(sin bombeo) − h(con bombeo).
Todos los archivos generados van a un directorio de trabajo propio (workspace/); nunca se escribe
sobre los archivos originales del proyecto.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Grilla

RAIZ = Path(__file__).resolve().parent.parent
WORKSPACE = RAIZ / "workspace"


@dataclass(frozen=True)
class ParametrosFlujo:
    h_oeste: float = 30.0       # m (didáctico)
    h_este: float = 20.0        # m (didáctico)
    recarga: float = 0.001      # m/día (didáctico, ≈ 365 mm/año)
    caudal_pozo: float = -1000.0  # m³/día; negativo = extracción (didáctico)
    fila_pozo: int = 7
    col_pozo: int = 10
    espesor: float = 50.0       # m (didáctico)

    def validar(self, g: Grilla):
        if not (0 <= self.fila_pozo < g.nfilas and 0 <= self.col_pozo < g.ncols):
            raise ValueError(f"El pozo debe estar dentro de la grilla (fila 0–{g.nfilas - 1}, col 0–{g.ncols - 1}).")
        if self.espesor <= 0:
            raise ValueError("El espesor debe ser > 0.")
        if self.caudal_pozo > 0:
            raise ValueError("En MODFLOW 6 el bombeo es NEGATIVO (m³/día). Un valor positivo sería inyección.")
        if self.recarga < 0:
            raise ValueError("La recarga no puede ser negativa en este modelo didáctico.")


def localizar_mf6(ruta_usuario: str | None = None) -> str | None:
    """Busca el ejecutable mf6: ruta indicada → variable MF6_EXE → PATH → carpeta bin local."""
    candidatos = []
    if ruta_usuario:
        candidatos.append(ruta_usuario)
    if os.environ.get("MF6_EXE"):
        candidatos.append(os.environ["MF6_EXE"])
    for c in candidatos:
        p = Path(c).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    w = shutil.which("mf6")
    if w:
        return w
    for extra in (RAIZ / "bin" / "mf6", RAIZ / "bin" / "mf6.exe"):
        if extra.is_file():
            return str(extra)
    return None


def flopy_disponible() -> tuple[bool, str]:
    try:
        import flopy
        return True, flopy.__version__
    except Exception as e:
        return False, str(e)


def _dir_trabajo() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run_", dir=WORKSPACE))


def correr_modelo(K: np.ndarray, g: Grilla, par: ParametrosFlujo, exe: str, con_bombeo: bool) -> np.ndarray:
    """Ejecuta un modelo MF6 estacionario y devuelve h con forma (nfilas, ncols), en metros."""
    import flopy

    par.validar(g)
    K = np.asarray(K, float)
    if K.shape != (g.nfilas, g.ncols) or not np.isfinite(K).all() or (K <= 0).any():
        raise ValueError("El campo de K debe tener la forma de la grilla y ser finito y positivo (m/día). "
                         "Si interpolaste por zonas, asegúrate de que todas las celdas tengan estimación.")
    ws = _dir_trabajo()
    try:
        sim = flopy.mf6.MFSimulation(sim_name="krig", sim_ws=str(ws), exe_name=exe, verbosity_level=0)
        flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)], time_units="DAYS")
        flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-8, inner_dvclose=1e-9,
                             outer_maximum=200, inner_maximum=400, linear_acceleration="BICGSTAB")
        gwf = flopy.mf6.ModflowGwf(sim, modelname="krig", save_flows=True)
        flopy.mf6.ModflowGwfdis(gwf, length_units="METERS", nlay=1, nrow=g.nfilas, ncol=g.ncols,
                                delr=g.dx, delc=g.dy, top=par.espesor, botm=0.0)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=K[np.newaxis, :, :])
        flopy.mf6.ModflowGwfic(gwf, strt=0.5 * (par.h_oeste + par.h_este))
        chd = [[(0, i, 0), par.h_oeste] for i in range(g.nfilas)] + \
              [[(0, i, g.ncols - 1), par.h_este] for i in range(g.nfilas)]
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: chd})
        rch = [[(0, i, j), par.recarga] for i in range(g.nfilas) for j in range(g.ncols)]
        flopy.mf6.ModflowGwfrch(gwf, stress_period_data={0: rch})
        if con_bombeo and par.caudal_pozo != 0:
            flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: [[(0, par.fila_pozo, par.col_pozo), par.caudal_pozo]]})
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord="krig.hds", saverecord=[("HEAD", "LAST")])
        sim.write_simulation(silent=True)
        ok, _ = sim.run_simulation(silent=True)
        if not ok:
            raise RuntimeError("MODFLOW 6 terminó con error o no convergió. Revisa K, recarga y bombeo "
                               "(¿bombeo tan grande que seca la celda? aquí la capa es confinada).")
        h = gwf.output.head().get_data()[0]
        h = np.array(h, float)
        if not np.isfinite(h).all() or (np.abs(h) > 1e10).any():
            raise RuntimeError("MODFLOW devolvió cargas no válidas (celdas secas o no convergencia).")
        return h
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def simular(K: np.ndarray, g: Grilla, par: ParametrosFlujo, exe: str) -> dict:
    """Dos corridas: sin bombeo (referencia) y con bombeo. Abatimiento = h_ref − h_bombeo."""
    h_ref = correr_modelo(K, g, par, exe, con_bombeo=False)
    h_pump = correr_modelo(K, g, par, exe, con_bombeo=True)
    return {"h_ref": h_ref, "h_bombeo": h_pump, "abatimiento": h_ref - h_pump}


def sensibilidades(construir_K, nombres: list[str], p0: np.ndarray, celdas_obs: dict, g: Grilla,
                   par: ParametrosFlujo, exe: str, dp: float = 0.1) -> dict:
    """Sensibilidad LOCAL ∂h/∂pᵢ en celdas de observación por diferencias finitas hacia adelante.

    construir_K(p) → campo K (m/día) interpolado con el vector de valores p (log10 K).
    No es una prueba formal de identificabilidad: solo muestra a qué parámetros responde cada
    observación en el entorno del estado actual."""
    base = correr_modelo(construir_K(p0), g, par, exe, con_bombeo=True)
    res = {etq: [] for etq in celdas_obs}
    for i in range(len(p0)):
        p1 = p0.copy()
        p1[i] += dp
        h1 = correr_modelo(construir_K(p1), g, par, exe, con_bombeo=True)
        for etq, (fi, co) in celdas_obs.items():
            res[etq].append((h1[fi, co] - base[fi, co]) / dp)
    return {etq: np.array(v) for etq, v in res.items()}
