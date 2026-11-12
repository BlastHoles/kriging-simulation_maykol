"""Contenido didáctico de la introducción guiada (se muestra en paneles desplegables)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .interpolation import ejemplo_minimo


def _simbolos(filas):
    st.table(pd.DataFrame(filas, columns=["Símbolo", "Significado", "Unidades"]).set_index("Símbolo"))


def render_intro():
    st.markdown(
        "Esta guía va de lo simple a lo detallado. Abre cada panel en orden; las ecuaciones aparecen "
        "siempre junto a la interpretación de cada símbolo. Todo lo que está marcado como "
        "**supuesto didáctico** no proviene del artículo ni del proyecto original.")

    with st.expander("1. Variable regionalizada, facies y propiedad continua", expanded=True):
        st.markdown(
            "- **Variable regionalizada**: una cantidad que tiene un valor en cada punto del espacio y cuyos "
            "valores cercanos se parecen más que los lejanos (por ejemplo, la conductividad hidráulica K). "
            "El kriging la trata como una realización de un campo aleatorio con cierta correlación espacial.\n"
            "- **Facies**: categoría geológica discreta (arena, arcilla, grava). Es una variable *categórica*.\n"
            "- **Propiedad continua (K)**: toma cualquier valor positivo dentro de una facies y puede variar "
            "órdenes de magnitud. Por eso se interpola **p = log10(K)** y luego se vuelve a K = 10^p.")
        st.latex(r"p = \log_{10}(K), \qquad K = 10^{p}")
        _simbolos([("K", "conductividad hidráulica", "m/día"), ("p", "log10 de K (con K en m/día)", "adimensional")])

    with st.expander("2. Punto piloto vs pozo de observación; qué es interpolar"):
        st.markdown(
            "- **Punto piloto**: lugar donde se *define* un valor del parámetro (aquí p). Es un **parámetro** que "
            "se ajusta en la calibración; el resto del campo se obtiene por interpolación entre ellos.\n"
            "- **Pozo de observación**: lugar donde se *mide* un dato (por ejemplo la carga hidráulica h). Es una "
            "**observación** contra la que se compara el modelo.\n"
            "- **Interpolar**: estimar el valor en lugares sin dato a partir de los puntos conocidos.\n\n"
            "Un punto piloto puede coincidir o no con un pozo; son conceptos distintos.")

    with st.expander("3. Covarianza, variograma, sill, nugget, rango y anisotropía"):
        st.markdown("La **covarianza** C(h) dice cuánto «se parecen» dos puntos a distancia h (en las mismas "
                    "unidades que p²). El **variograma** γ(h) mide lo contrario: cuánto *se diferencian*. "
                    "Con las definiciones usadas aquí están relacionados por:")
        st.latex(r"\gamma(h) = C(0) - C(h)\quad (h>0)")
        st.latex(r"C(h) = \text{sill}\cdot\rho(h/a),\qquad C(0)=\text{nugget}+\text{sill}")
        _simbolos([("sill", "contribución (varianza estructural)", "p²"),
                   ("nugget", "variabilidad a escala menor que la separación entre puntos / error", "p²"),
                   ("a", "longitud de correlación (su definición depende del modelo)", "m"),
                   ("ρ", "función de correlación del modelo (entre 0 y 1)", "adimensional")])
        st.markdown(
            "- Exponencial: ρ = exp(−h/a). `a` es la longitud *e-folding*; el rango práctico ≈ 3a.\n"
            "- Gaussiano: ρ = exp(−(h/a)²). Rango práctico ≈ 1.73a.\n"
            "- Esférico: `a` es el rango verdadero (C = 0 para h ≥ a).\n"
            "- **Anisotropía**: la correlación llega más lejos en una dirección (eje mayor, azimut desde el norte, "
            "sentido horario) que en la perpendicular; razón R = a_mayor/a_menor ≥ 1.")

    with st.expander("4. Kriging ordinario y su supuesto sobre la media"):
        st.markdown(
            "El **kriging ordinario** supone que la media local es *constante pero desconocida*. Busca los pesos "
            "w que minimizan la varianza del error entre todas las combinaciones lineales sin sesgo. "
            "El kriging **simple** supone la media *conocida*; el **universal** permite una tendencia "
            "(plano, cuadrática); el de **deriva externa** usa una variable auxiliar conocida en toda la grilla.")
        st.latex(r"\begin{bmatrix} \mathbf{C} & \mathbf{1}\\ \mathbf{1}^{T} & 0\end{bmatrix}"
                 r"\begin{bmatrix}\mathbf{w}\\ \lambda\end{bmatrix}="
                 r"\begin{bmatrix}\mathbf{c}\\ 1\end{bmatrix},\qquad \hat p(x)=\sum_i w_i\,p_i")
        _simbolos([("C", "covarianza entre cada par de puntos piloto", "p²"),
                   ("c", "covarianza entre cada punto piloto y la celda objetivo x", "p²"),
                   ("w", "pesos de kriging", "adimensional"),
                   ("λ", "multiplicador de Lagrange: «precio» de exigir Σw = 1", "p²"),
                   ("p̂(x)", "estimación de log10(K) en x", "adimensional")])

    with st.expander("5. Pesos que suman 1, pesos negativos y varianza de kriging"):
        st.markdown(
            "- **Σw = 1** garantiza que el estimador no tenga sesgo cuando la media es constante pero desconocida.\n"
            "- Los pesos pueden ser **negativos** por el *efecto pantalla*: un punto «escondido» detrás de otro "
            "recibe peso ≤ 0. Esto permite que p̂ salga **fuera del rango** de los datos.\n"
            "- **Varianza de kriging**:")
        st.latex(r"\sigma^2_{OK}(x)=C(0)-\sum_i w_i c_i-\lambda")
        st.markdown(
            "Mide la incertidumbre de *interpolación* bajo el modelo de covarianza elegido y depende solo de la "
            "geometría de los puntos, no de los valores. **No incluye**: error de medición de K, error del "
            "modelo conceptual, incertidumbre en los parámetros del variograma, incertidumbre de condiciones de "
            "borde, recarga o bombeo, ni la no-unicidad de la calibración. Además es la varianza de p = log10(K), "
            "no de K.")

    with st.expander("6. ¿Qué es `parval1` en el ejemplo pyEMU?"):
        st.markdown(
            "En pyEMU, `parval1` es el **valor inicial del parámetro** en las unidades físicas del modelo. En "
            "este proyecto contiene **K en m/día**. Para interpolar se aplica una transformación logarítmica: "
            "p = log10(parval1), se hace kriging sobre p y se vuelve a K = 10^p̂. Si se aplicaran los factores de "
            "kriging directamente a K, el resultado sería una interpolación aritmética distinta de la geométrica. "
            "(Descripción basada en el enunciado; no se inspeccionó el cuaderno original.)")

    with st.expander("7. Ejemplo numérico paso a paso (3 puntos alineados)"):
        v, xy, z, obj, r = ejemplo_minimo()
        st.markdown("Supuestos didácticos: puntos en x = 0, 100, 300 m (y = 0) con p = 1.0, 1.4, 0.8; objetivo en "
                    "x = 150 m; modelo exponencial, sill = 1, nugget = 0, a = 350 m.")
        d = np.abs(xy[:, 0][:, None] - xy[:, 0][None, :])
        st.markdown("**Paso 1 – covarianza entre puntos** C = exp(−d/a):")
        st.dataframe(pd.DataFrame(r.C, index=["P1", "P2", "P3"], columns=["P1", "P2", "P3"]).round(4))
        st.markdown("**Paso 2 – covarianza punto–objetivo** c:")
        st.dataframe(pd.DataFrame({"distancia (m)": np.abs(xy[:, 0] - obj[0, 0]), "c": r.c[:, 0]},
                                  index=["P1", "P2", "P3"]).round(4))
        w = r.pesos[0]
        st.markdown("**Paso 3 – resolver el sistema aumentado**: pesos w y multiplicador λ:")
        st.dataframe(pd.DataFrame({"w": w, "w·p": w * z}, index=["P1", "P2", "P3"]).round(4))
        st.markdown(f"Suma de pesos = **{w.sum():.6f}**; λ = {r.multiplicadores[0, 0]:.5f}.")
        st.markdown(f"**Paso 4 – estimación**: p̂ = Σ wᵢpᵢ = **{r.estimacion[0]:.4f}**, entonces "
                    f"K = 10^p̂ = **{10 ** r.estimacion[0]:.2f} m/día**.")
        st.markdown(f"**Paso 5 – varianza**: σ² = C(0) − Σwc − λ = **{r.varianza[0]:.4f}** p².")
