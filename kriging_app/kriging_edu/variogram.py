"""Modelos de covarianza / variograma con definiciones matemáticas explícitas.

Convención usada en TODO el paquete (coincide con pyEMU):
  * Covarianza total en h=0:    C(0) = nugget + sill     (sill = «contribución» parcial)
  * Para h > 0:                 C(h) = sill · ρ(h_ef / a)
  * Semivariograma:             γ(h) = nugget + sill · (1 − ρ(h_ef / a))   (h>0),  γ(0)=0
  * Relación:                   γ(h) = C(0) − C(h)   (h>0)
  * ρ exponencial:  exp(−r)        → a = longitud e-folding; rango práctico ≈ 3a
  * ρ gaussiano:    exp(−r²)       → a = escala; rango práctico ≈ 1.73a
  * ρ esférico:     1 − 1.5r + 0.5r³ (r<1), 0 (r≥1) → a = rango verdadero
  * Anisotropía geométrica: h_ef = sqrt(u² + (R·v)²), con u a lo largo del eje mayor
    (azimut medido desde el norte, sentido horario) y v perpendicular; R = a_mayor/a_menor ≥ 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODELOS = ("exponencial", "esferico", "gaussiano")
ETIQUETAS_MODELO = {"exponencial": "Exponencial", "esferico": "Esférico", "gaussiano": "Gaussiano"}

_EXPLICACION_A = {
    "exponencial": "En el modelo **exponencial**, `a` es la longitud *e-folding*: a la distancia h = a la "
                   "correlación cae a e⁻¹ ≈ 0.37. NO es el rango práctico: la correlación llega a 0.05 "
                   "recién en h ≈ 3a.",
    "gaussiano": "En el modelo **gaussiano** la correlación es exp(−(h/a)²): `a` es un parámetro de escala; "
                 "a h = a la correlación es e⁻¹ ≈ 0.37 y llega a 0.05 en h ≈ 1.73a (rango práctico).",
    "esferico": "En el modelo **esférico**, `a` es el rango verdadero: la covarianza vale exactamente 0 "
                "para h ≥ a (los puntos más lejanos que a no se correlacionan).",
}


@dataclass(frozen=True)
class Variograma:
    modelo: str = "exponencial"
    sill: float = 1.0            # contribución (sill parcial), en unidades de p²
    nugget: float = 0.0          # efecto pepita, en unidades de p²
    a: float = 350.0             # parámetro de longitud (m) a lo largo del eje mayor
    razon_anisotropia: float = 1.0
    azimut: float = 0.0          # grados desde el norte, sentido horario (eje mayor)

    def __post_init__(self):
        if self.modelo not in MODELOS:
            raise ValueError(f"Modelo desconocido «{self.modelo}». Opciones: {MODELOS}")
        if not (self.sill > 0):
            raise ValueError("El sill debe ser > 0.")
        if self.nugget < 0:
            raise ValueError("El nugget no puede ser negativo.")
        if not (self.a > 0):
            raise ValueError("El rango/escala `a` debe ser > 0 (metros).")
        if self.razon_anisotropia < 1.0:
            raise ValueError("La razón de anisotropía a_mayor/a_menor debe ser ≥ 1.")

    @property
    def c0(self) -> float:
        """Covarianza total en h=0 (nugget + sill)."""
        return self.nugget + self.sill

    def distancia_efectiva(self, dx, dy) -> np.ndarray:
        dx, dy = np.asarray(dx, float), np.asarray(dy, float)
        if self.razon_anisotropia == 1.0:
            return np.hypot(dx, dy)
        th = np.deg2rad(90.0 - self.azimut)
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        return np.sqrt(u * u + (self.razon_anisotropia * v) ** 2)

    def correlacion(self, h) -> np.ndarray:
        """ρ(h): correlación (entre 0 y 1) a distancia efectiva h (m), sin nugget."""
        r = np.asarray(h, float) / self.a
        if self.modelo == "exponencial":
            return np.exp(-r)
        if self.modelo == "gaussiano":
            return np.exp(-r * r)
        return np.where(r < 1.0, 1.0 - 1.5 * r + 0.5 * r ** 3, 0.0)

    def covarianza_h(self, h) -> np.ndarray:
        """C(h) con h = distancia efectiva (incluye el salto del nugget en h=0)."""
        h = np.asarray(h, float)
        return np.where(h < 1e-9, self.c0, self.sill * self.correlacion(h))

    def covarianza(self, dx, dy) -> np.ndarray:
        return self.covarianza_h(self.distancia_efectiva(dx, dy))

    def semivariograma_h(self, h) -> np.ndarray:
        h = np.asarray(h, float)
        return np.where(h < 1e-9, 0.0, self.nugget + self.sill * (1.0 - self.correlacion(h)))

    def rango_practico(self, umbral: float = 0.05) -> float:
        """Distancia (eje mayor) a la que la correlación ρ cae a `umbral`."""
        if self.modelo == "exponencial":
            return self.a * np.log(1 / umbral)
        if self.modelo == "gaussiano":
            return self.a * np.sqrt(np.log(1 / umbral))
        return self.a

    def explicacion_a(self) -> str:
        return _EXPLICACION_A[self.modelo]
