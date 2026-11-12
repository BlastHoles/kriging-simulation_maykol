"""Visualización con Plotly. Todas las funciones devuelven una go.Figure.

Reglas didácticas:
  * Campos de ESTIMACIÓN: escala secuencial (viridis), rango común al comparar.
  * Mapas de DIFERENCIA: escala divergente centrada en 0 (RdBu).
  * Mapas de INCERTIDUMBRE (varianza): escala distinta (magma) para no confundirlos con estimaciones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import Grilla, p_a_K
from .variogram import Variograma, ETIQUETAS_MODELO

ESC_EST = "Viridis"
ESC_VAR = "Magma"
ESC_DIF = "RdBu"


def rango_comun(arrays, simetrico: bool = False):
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays if a is not None] or [np.array([0.0])])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if simetrico:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return lo, hi


def _heatmap(grilla: Grilla, campo, titulo_cb, escala, zmin, zmax, zmid=None, hover="valor"):
    y_asc = grilla.y_centros_filas[::-1]
    return go.Heatmap(x=grilla.x_centros, y=y_asc, z=np.asarray(campo)[::-1, :], colorscale=escala,
                      zmin=zmin, zmax=zmax, zmid=zmid, colorbar=dict(title=titulo_cb, thickness=12, len=0.9),
                      hovertemplate=f"x=%{{x:.0f}} m<br>y=%{{y:.0f}} m<br>{hover}=%{{z:.4g}}<extra></extra>")


def _marcas_puntos(puntos: pd.DataFrame, texto_valor=True):
    textos = [f"{n}<br>p={p:.2f}" if texto_valor else str(n) for n, p in zip(puntos["nombre"], puntos["p"])]
    hover = [f"{n}<br>x={x:.0f} m, y={y:.0f} m<br>p=log10K={p:.3f}<br>K={10 ** p:.2f} m/día"
             for n, x, y, p in zip(puntos["nombre"], puntos["x"], puntos["y"], puntos["p"])]
    return go.Scatter(x=puntos["x"], y=puntos["y"], mode="markers+text", text=textos, textposition="top center",
                      textfont=dict(size=10, color="white"), name="Puntos piloto",
                      marker=dict(symbol="diamond", size=10, color="white", line=dict(color="black", width=1.5)),
                      hovertext=hover, hoverinfo="text", showlegend=False)


def _layout(fig, grilla: Grilla, titulo, alto=430):
    fig.update_xaxes(range=[0, grilla.Lx], title="x (m, este)", constrain="domain")
    fig.update_yaxes(range=[0, grilla.Ly], title="y (m, norte)", scaleanchor="x", scaleratio=1)
    fig.update_layout(title=titulo, height=alto, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def mapa(grilla, campo, titulo, titulo_cb, puntos=None, escala=ESC_EST, zmin=None, zmax=None,
         zmid=None, hover="valor", objetivo=None, alto=430):
    """Mapa simple con hover nativo del heatmap."""
    lo, hi = rango_comun([campo])
    fig = go.Figure(_heatmap(grilla, campo, titulo_cb, escala, zmin if zmin is not None else lo,
                             zmax if zmax is not None else hi, zmid, hover))
    if puntos is not None and len(puntos):
        fig.add_trace(_marcas_puntos(puntos))
    if objetivo is not None:
        x0, y0 = grilla.centro(*objetivo)
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(symbol="x", size=13, color="red",
                                 line=dict(width=2, color="white")), hoverinfo="skip", showlegend=False))
    return _layout(fig, grilla, titulo, alto)


def mapa_clicable(grilla: Grilla, p_campo, titulo, puntos=None, objetivo=None, alto=470):
    """Mapa de log10(K) con una capa invisible de marcadores para hover completo y selección por clic."""
    XX, YY = grilla.mallas()
    K = p_a_K(p_campo)
    fil, col = np.indices(XX.shape)
    lo, hi = rango_comun([p_campo])
    fig = go.Figure(_heatmap(grilla, p_campo, "log10(K)<br>[log10 m/día]", ESC_EST, lo, hi))
    fig.data[0].hoverinfo = "skip"
    fig.data[0].hovertemplate = None
    textos = [f"fila {i}, col {j}<br>x={x:.0f} m, y={y:.0f} m<br>log10K={pp:.3f}<br>K={k:.2f} m/día"
              for i, j, x, y, pp, k in zip(fil.ravel(), col.ravel(), XX.ravel(), YY.ravel(), p_campo.ravel(), K.ravel())]
    fig.add_trace(go.Scatter(x=XX.ravel(), y=YY.ravel(), mode="markers", hovertext=textos, hoverinfo="text",
                             marker=dict(size=14, opacity=0.001, symbol="square"), showlegend=False, name="celdas",
                             selected=dict(marker=dict(opacity=0.001)), unselected=dict(marker=dict(opacity=0.001))))
    if puntos is not None and len(puntos):
        fig.add_trace(_marcas_puntos(puntos))
    if objetivo is not None:
        x0, y0 = grilla.centro(*objetivo)
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(symbol="x", size=15, color="red",
                                 line=dict(width=2, color="white")), hoverinfo="skip", showlegend=False))
    fig.update_layout(clickmode="event+select", dragmode="select", hovermode="closest")
    return _layout(fig, grilla, titulo, alto)


def mapas_lado_a_lado(grilla, campos: dict, titulo_cb, escala=ESC_EST, puntos=None, comun=True,
                      simetrico=False, zmid=None, hover="valor", alto=380):
    """Varios mapas en columnas con la MISMA escala de color (si comun=True)."""
    n = len(campos)
    fig = make_subplots(rows=1, cols=n, subplot_titles=list(campos.keys()), horizontal_spacing=0.06)
    lo, hi = rango_comun(list(campos.values()), simetrico)
    for k, (nombre, campo) in enumerate(campos.items(), start=1):
        zl, zh = (lo, hi) if comun else rango_comun([campo], simetrico)
        h = _heatmap(grilla, campo, titulo_cb, escala, zl, zh, zmid, hover)
        h.showscale = (k == n)
        fig.add_trace(h, row=1, col=k)
        if puntos is not None and len(puntos):
            fig.add_trace(_marcas_puntos(puntos, texto_valor=False), row=1, col=k)
        fig.update_xaxes(range=[0, grilla.Lx], row=1, col=k, title="x (m)")
        fig.update_yaxes(range=[0, grilla.Ly], row=1, col=k, scaleanchor=f"x{'' if k == 1 else k}", scaleratio=1,
                         title="y (m)" if k == 1 else None)
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def perfiles(grilla: Grilla, campos: dict, eje: str, indice: int, puntos=None, etiqueta="log10(K)"):
    """Corte a lo largo de una fila (eje='fila', recorre x) o de una columna (eje='col', recorre y)."""
    fig = go.Figure()
    if eje == "fila":
        s = grilla.x_centros
        y_corte = grilla.centro(indice, 0)[1]
        for nombre, c in campos.items():
            fig.add_trace(go.Scatter(x=s, y=c[indice, :], mode="lines+markers", name=nombre))
        xlab = f"x (m) a lo largo de la fila {indice} (y = {y_corte:.0f} m)"
        if puntos is not None:
            m = (puntos["y"] - y_corte).abs() <= grilla.dy / 2
            fig.add_trace(go.Scatter(x=puntos.loc[m, "x"], y=puntos.loc[m, "p"], mode="markers+text",
                                     text=puntos.loc[m, "nombre"], textposition="top center", name="Puntos en el corte",
                                     marker=dict(symbol="diamond", size=11, color="black")))
    else:
        s = grilla.y_centros_filas
        x_corte = grilla.centro(0, indice)[0]
        for nombre, c in campos.items():
            fig.add_trace(go.Scatter(x=s, y=c[:, indice], mode="lines+markers", name=nombre))
        xlab = f"y (m) a lo largo de la columna {indice} (x = {x_corte:.0f} m)"
        if puntos is not None:
            m = (puntos["x"] - x_corte).abs() <= grilla.dx / 2
            fig.add_trace(go.Scatter(x=puntos.loc[m, "y"], y=puntos.loc[m, "p"], mode="markers+text",
                                     text=puntos.loc[m, "nombre"], textposition="top center", name="Puntos en el corte",
                                     marker=dict(symbol="diamond", size=11, color="black")))
    fig.update_layout(height=380, xaxis_title=xlab, yaxis_title=etiqueta, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def barras_pesos(tabla: pd.DataFrame, tabla_idw: pd.DataFrame | None = None, titulo="Pesos por punto piloto"):
    """Barras de pesos con línea de referencia en cero; los pesos negativos se resaltan en rojo."""
    colores = ["#d62728" if w < 0 else "#1f77b4" for w in tabla["peso w"]]
    fig = go.Figure(go.Bar(x=tabla["punto"], y=tabla["peso w"], name="Kriging", marker_color=colores,
                           text=[f"{w:.3f}" for w in tabla["peso w"]], textposition="outside"))
    if tabla_idw is not None:
        fig.add_trace(go.Bar(x=tabla_idw["punto"], y=tabla_idw["peso IDW"], name="IDW", marker_color="#999999",
                             opacity=0.8))
    fig.add_hline(y=0, line_color="black", line_width=1.5)
    fig.update_layout(title=titulo, barmode="group", height=380, yaxis_title="peso (adimensional)",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def distancia_vs_peso(tabla: pd.DataFrame):
    """Dispersión distancia–peso: muestra que el más cercano no siempre domina."""
    fig = go.Figure(go.Scatter(x=tabla["distancia al objetivo (m)"], y=tabla["peso w"], mode="markers+text",
                               text=tabla["punto"], textposition="top center",
                               marker=dict(size=11, color=["#d62728" if w < 0 else "#1f77b4" for w in tabla["peso w"]])))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(height=330, xaxis_title="distancia al objetivo (m)", yaxis_title="peso kriging",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def matriz_covarianza(C, nombres):
    fig = go.Figure(go.Heatmap(z=C, x=nombres, y=nombres, colorscale="Blues", colorbar=dict(title="C"),
                               hovertemplate="%{y}–%{x}: C=%{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def curvas_variograma(v: Variograma, exp_df: pd.DataFrame | None = None, comparar: bool = False):
    """Covarianza C(h) y semivariograma γ(h) en el mismo gráfico doble."""
    hmax = max(3.5 * v.a, 1.1 * v.rango_practico())
    h = np.linspace(0, hmax, 400)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Covarianza C(h)", "Semivariograma γ(h)"))
    modelos = ["exponencial", "esferico", "gaussiano"] if comparar else [v.modelo]
    for mod in modelos:
        vv = Variograma(mod, v.sill, v.nugget, v.a, v.razon_anisotropia, v.azimut)
        nombre = ETIQUETAS_MODELO[mod]
        fig.add_trace(go.Scatter(x=h, y=vv.covarianza_h(h), name=nombre, mode="lines", legendgroup=nombre), row=1, col=1)
        fig.add_trace(go.Scatter(x=h, y=vv.semivariograma_h(h), name=nombre, mode="lines", legendgroup=nombre,
                                 showlegend=False), row=1, col=2)
    if v.razon_anisotropia > 1 and not comparar:
        a_menor = v.a / v.razon_anisotropia
        vm = Variograma(v.modelo, v.sill, v.nugget, a_menor)
        fig.add_trace(go.Scatter(x=h, y=vm.covarianza_h(h), name=f"eje menor (a={a_menor:.0f} m)", mode="lines",
                                 line=dict(dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=h, y=vm.semivariograma_h(h), name="eje menor", mode="lines",
                                 line=dict(dash="dash"), showlegend=False), row=1, col=2)
    fig.add_hline(y=v.c0, line_dash="dot", line_color="gray", annotation_text="sill total (nugget+sill)", row=1, col=2)
    for k in (1, 2, 3):
        fig.add_vline(x=k * v.a, line_dash="dot", line_color="lightgray", row=1, col=1,
                      annotation_text=f"{k}a" if k else None)
    if exp_df is not None and len(exp_df):
        fig.add_trace(go.Scatter(x=exp_df["h (m)"], y=exp_df["gamma"], mode="markers+text",
                                 text=[f"n={n}" for n in exp_df["pares"]], textposition="top center",
                                 name="Experimental (Matheron)", marker=dict(size=9, color="black")), row=1, col=2)
    fig.update_xaxes(title_text="h (m)")
    fig.update_yaxes(title_text="C [p²]", row=1, col=1)
    fig.update_yaxes(title_text="γ [p²]", row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def mapa_covarianza_objetivo(grilla: Grilla, v: Variograma, objetivo, titulo):
    """C(celda, objetivo): muestra la forma (elipse si hay anisotropía) de la influencia del objetivo."""
    XX, YY = grilla.mallas()
    x0, y0 = grilla.centro(*objetivo)
    C = v.covarianza(XX - x0, YY - y0)
    return mapa(grilla, C, titulo, "C [p²]", escala="Blues", objetivo=objetivo, hover="C", alto=360)


def barras_sensibilidad(df: pd.DataFrame, titulo):
    fig = go.Figure()
    for col in df.columns[1:]:
        fig.add_trace(go.Bar(x=df[df.columns[0]], y=df[col], name=col))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(title=titulo, barmode="group", height=360, yaxis_title="∂h/∂p  (m por unidad de log10K)",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    return fig
