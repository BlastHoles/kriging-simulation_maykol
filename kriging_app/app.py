"""Aplicación didáctica: kriging con puntos piloto (Streamlit).

Ejecutar:  streamlit run app.py
Las pestañas van de lo conceptual a lo avanzado. Nada se escribe sobre los archivos originales
del proyecto: los archivos de MODFLOW se generan en workspace/ y se borran al terminar.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from kriging_edu import content, data, experiments, flow, plots, pyemu_compare
from kriging_edu.data import Grilla, p_a_K, K_a_p
from kriging_edu.interpolation import (ConfigInterp, ErrorKriging, METODOS, TENDENCIAS, detalle_celda,
                                       interpolar_campo, variograma_experimental)
from kriging_edu.variogram import ETIQUETAS_MODELO, MODELOS, Variograma

st.set_page_config(page_title="Kriging con puntos piloto", layout="wide")
G = Grilla()


# ------------------------------------------------------------------ estado --
def _init():
    ss = st.session_state
    if "puntos" not in ss:
        ss.puntos = data.puntos_proyecto(G)
        ss.modo_prev = "log10(K)"
        ss.ver = 0
        ss.pts_base = ss.puntos[["nombre", "x", "y"]].assign(valor=ss.puntos["p"])
        ss.objetivo = (7, 5)
        ss.click_pts = None
        ss.click_pesos = None
        ss.celda_clic = None
        ss.deriva = None
        ss.csv_id = None
        ss.flujo_res = {}


def vista(puntos: pd.DataFrame, modo: str) -> pd.DataFrame:
    v = puntos[["nombre", "x", "y"]].copy()
    v["valor"] = puntos["p"] if modo == "log10(K)" else p_a_K(puntos["p"])
    return v.reset_index(drop=True)


def bump():
    """Cambio programático de los puntos: reinicia el editor para que no reaplique ediciones viejas."""
    ss = st.session_state
    ss.pts_base = vista(ss.puntos, ss.modo_prev)
    ss.ver += 1


def calcular(cfg: ConfigInterp):
    try:
        return interpolar_campo(G, st.session_state.puntos, cfg, st.session_state.deriva), None
    except (ErrorKriging, ValueError) as e:
        return None, str(e)


def celda_de_evento(evento):
    try:
        pts = evento.selection.points
    except Exception:
        return None
    if not pts:
        return None
    p = pts[0]
    ij = G.celda_de_xy(float(p.get("x")), float(p.get("y")))
    return ij


_init()
ss = st.session_state

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Modelo de covarianza (global)")
    modelo = st.selectbox("Modelo", MODELOS, format_func=lambda m: ETIQUETAS_MODELO[m], key="sb_modelo")
    sill = st.number_input("Sill (p²)", 0.001, 100.0, 1.0, 0.1, key="sb_sill")
    nugget = st.number_input("Nugget (p²)", 0.0, 100.0, 0.0, 0.05, key="sb_nugget")
    a = st.number_input("Rango / escala a (m)", 1.0, 5000.0, 350.0, 25.0, key="sb_a")
    ani = st.number_input("Razón de anisotropía a_mayor/a_menor (≥ 1)", 1.0, 20.0, 1.0, 0.5, key="sb_ani")
    azi = st.number_input("Azimut del eje mayor (° desde el norte, horario)", 0.0, 180.0, 0.0, 5.0, key="sb_azi")
    try:
        VARIO = Variograma(modelo, sill, nugget, a, ani, azi)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.caption(VARIO.explicacion_a())
    st.divider()
    st.subheader("Zonas (opcional)")
    zonas = st.checkbox("Interpolar por zonas separadas", False, key="sb_zonas")
    x_corte = st.number_input("Frontera de zonas: x (m)", 0.0, G.Lx, 500.0, 50.0, key="sb_xcorte", disabled=not zonas)
    st.divider()
    st.subheader("Deriva externa (KED)")
    op = st.radio("Variable auxiliar", ["No disponible", "Sintética (didáctica)", "Subir CSV"], key="sb_deriva")
    if op == "Sintética (didáctica)":
        ss.deriva = data.deriva_didactica(G)
        st.caption("Supuesto didáctico: gradiente este–oeste sintético, no es un dato real.")
    elif op == "Subir CSV":
        f = st.file_uploader("CSV de 15×20 valores (fila 0 = norte)", type="csv", key="sb_dcsv")
        ss.deriva = None
        if f is not None:
            try:
                ss.deriva = data.leer_deriva(f, G)
            except Exception as e:
                st.error(str(e))
    else:
        ss.deriva = None
    st.divider()
    st.subheader("MODFLOW 6")
    ruta_mf6 = st.text_input("Ruta del ejecutable mf6 (opcional)", "", key="sb_mf6")
    EXE = flow.localizar_mf6(ruta_mf6 or None)
    ok_fp, ver_fp = flow.flopy_disponible()
    st.caption(f"FloPy: {'v' + ver_fp if ok_fp else 'no instalado'} · mf6: {EXE or 'no encontrado'}")

BASE = ConfigInterp(metodo="ordinario", vario=VARIO, separar_zonas=zonas, x_corte=x_corte)

st.title("Kriging con puntos piloto: aprende, experimenta y explica")
st.caption("Grilla 15 × 20 (celdas de 50 × 50 m, dominio 1000 × 750 m) · p = log10(K) con K en m/día · "
           "fila 0 = norte.")

tabs = st.tabs(["Introducción", "Puntos piloto", "Mapas", "Pesos", "Variograma", "pyEMU", "FloPy / MODFLOW 6",
                "Experimentos"])

# ============================================================ 0 · INTRODUCCIÓN
with tabs[0]:
    content.render_intro()

# ============================================================ 1 · PUNTOS PILOTO
with tabs[1]:
    st.subheader("Editor de puntos piloto")
    st.info("Escenario inicial: 9 puntos del proyecto con los valores del enunciado. **Supuesto didáctico:** las "
            "coordenadas no se entregaron; se usa una malla regular 3×3 en centros de celda. Cárgalas desde un CSV "
            "si las tienes.")
    modo = st.radio("Editar el valor en", ["log10(K)", "K (m/día)"], horizontal=True, key="modo_radio")
    modo = "log10(K)" if modo == "log10(K)" else "K (m/día)"
    if ss.modo_prev != modo:
        ss.modo_prev = modo
        bump()
    col_ed = "p = log10(K)" if modo == "log10(K)" else "K (m/día)"
    ed = st.data_editor(
        ss.pts_base, key=f"ed_{ss.ver}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"nombre": st.column_config.TextColumn("Nombre"),
                       "x": st.column_config.NumberColumn("x (m)", format="%.1f"),
                       "y": st.column_config.NumberColumn("y (m)", format="%.1f"),
                       "valor": st.column_config.NumberColumn(col_ed, format="%.4f")})
    ed = ed.dropna(how="all").reset_index(drop=True)
    ed["nombre"] = [str(n) if (isinstance(n, str) and n.strip()) else f"PP{i + 1}" for i, n in enumerate(ed["nombre"])]
    try:
        pv = ed["valor"].to_numpy(float)
        p_new = pv if modo == "log10(K)" else np.where(np.isfinite(pv), K_a_p(np.where(np.isfinite(pv), pv, 1.0)), np.nan)
        ss.puntos = pd.DataFrame({"nombre": ed["nombre"], "x": ed["x"].astype(float), "y": ed["y"].astype(float),
                                  "p": p_new})
    except ValueError as e:
        st.error(f"{e} Se mantiene la última tabla válida.")

    cA, cB, cC = st.columns(3)
    with cA:
        f = st.file_uploader("Importar CSV (x, y, p o K)", type="csv", key="up_pts")
        if f is not None and ss.csv_id != (f.name, f.size):
            try:
                ss.puntos = data.leer_csv(f)
                ss.csv_id = (f.name, f.size)
                bump()
                st.rerun()
            except Exception as e:
                ss.csv_id = (f.name, f.size)
                st.error(str(e))
    with cB:
        ruta = st.text_input("…o ruta de un CSV del proyecto", "", key="ruta_csv")
        if st.button("Cargar desde ruta") and ruta:
            try:
                ss.puntos = data.leer_csv(ruta)
                bump()
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with cC:
        if len(ss.puntos):
            st.download_button("Exportar CSV", data.a_csv(ss.puntos), "puntos_piloto.csv", "text/csv")
        if st.button("Restaurar escenario del proyecto"):
            ss.puntos = data.puntos_proyecto(G)
            bump()
            st.rerun()

    msgs = data.validar_puntos(ss.puntos, G, zonas, x_corte)
    for nivel, m in msgs:
        {"error": st.error, "advertencia": st.warning, "info": st.info}[nivel](m)
    PTS_OK = not any(n == "error" for n, _ in msgs)

    if PTS_OK:
        st.markdown("**Tabla completa (con conversiones y zona)**  —  K = 10^p")
        st.dataframe(data.tabla_completa(ss.puntos, zonas, x_corte), hide_index=True, width="stretch")
        st.markdown("**Mapa clicable** (kriging ordinario con el modelo global). Haz clic en una celda para agregar "
                    "o mover un punto. Si el clic no responde en tu navegador, usa la tabla.")
        res_map, err = calcular(BASE)
        if err:
            st.error(err)
        else:
            fig = plots.mapa_clicable(G, res_map.p, "log10(K) estimado", ss.puntos, objetivo=ss.celda_clic)
            ev = st.plotly_chart(fig, key="map_pts", on_select="rerun", selection_mode="points", width="stretch")
            ij = celda_de_evento(ev)
            if ij is not None and ij != ss.click_pts:
                ss.click_pts = ij
                ss.celda_clic = ij
                st.rerun()
            if ss.celda_clic:
                fi, co = ss.celda_clic
                x0, y0 = G.centro(fi, co)
                st.write(f"Celda elegida: fila {fi}, columna {co} (x = {x0:.0f} m, y = {y0:.0f} m). "
                         f"Estimación actual: log10K = {res_map.p[fi, co]:.3f}, K = {res_map.K[fi, co]:.2f} m/día.")
                b1, b2 = st.columns(2)
                if b1.button("Agregar un punto aquí (valor = estimación actual)"):
                    ss.puntos = pd.concat([ss.puntos, pd.DataFrame([{"nombre": f"PP{len(ss.puntos) + 1}", "x": x0,
                                                                     "y": y0, "p": float(res_map.p[fi, co])}])],
                                          ignore_index=True)
                    bump()
                    st.rerun()
                sel = b2.selectbox("Mover el punto…", ss.puntos["nombre"].tolist(), key="mover_sel")
                if b2.button("…a la celda elegida"):
                    idx = ss.puntos.index[ss.puntos["nombre"] == sel][0]
                    ss.puntos.loc[idx, ["x", "y"]] = [x0, y0]
                    bump()
                    st.rerun()
    else:
        res_map = None

PTS_OK = not any(n == "error" for n, _ in data.validar_puntos(ss.puntos, G, zonas, x_corte))
PUNTOS = ss.puntos


def sin_datos():
    st.warning("Corrige los errores de la pestaña «Puntos piloto» para usar esta sección.")


def selector_config(prefijo: str, titulo: str, metodo_def: str = "ordinario", modelo_def=None, a_def=None):
    """Widgets compactos para definir una configuración de interpolación."""
    st.markdown(f"**{titulo}**")
    metodo = st.selectbox("Método", list(METODOS), index=list(METODOS).index(metodo_def),
                          format_func=lambda m: METODOS[m], key=f"{prefijo}_m")
    v = VARIO
    kw = {}
    if metodo == "idw":
        kw["potencia_idw"] = st.number_input("Potencia IDW", 0.5, 8.0, 2.0, 0.5, key=f"{prefijo}_pot")
    else:
        mdl = st.selectbox("Modelo", MODELOS, index=MODELOS.index(modelo_def or VARIO.modelo),
                           format_func=lambda m: ETIQUETAS_MODELO[m], key=f"{prefijo}_mod")
        aa = st.number_input("a (m)", 1.0, 5000.0, float(a_def or VARIO.a), 25.0, key=f"{prefijo}_a")
        v = replace(VARIO, modelo=mdl, a=aa)
        if metodo == "simple":
            if st.checkbox("Media conocida manual", False, key=f"{prefijo}_mm"):
                kw["media_simple"] = st.number_input("Media (log10 K)", -5.0, 5.0, 1.0, 0.1, key=f"{prefijo}_mv")
        if metodo == "universal":
            kw["tendencia"] = st.selectbox("Tendencia", list(TENDENCIAS), index=1, format_func=lambda t: TENDENCIAS[t],
                                           key=f"{prefijo}_t")
        if metodo == "ordinario":
            kw["quitar_negativos"] = st.checkbox("Eliminar pesos negativos y renormalizar", False, key=f"{prefijo}_neg")
        if metodo == "ked" and ss.deriva is None:
            st.warning("Kriging con deriva externa desactivado: carga o activa una variable auxiliar en la barra lateral.")
    return ConfigInterp(metodo=metodo, vario=v, separar_zonas=zonas, x_corte=x_corte, **kw)


# ============================================================ 2 · MAPAS
with tabs[2]:
    st.subheader("Mapas y comparación de métodos")
    if not PTS_OK:
        sin_datos()
    else:
        cols = st.columns(3)
        defs = [("ordinario", None, None), ("idw", None, None), ("ordinario", "gaussiano", None)]
        cfgs = []
        for k, (c, (m, md, aa)) in enumerate(zip(cols, defs)):
            with c:
                usar = True if k < 2 else st.checkbox("Activar configuración C", False, key="usar_C")
                if usar:
                    cfgs.append((("A", "B", "C")[k], selector_config(f"cfg{k}", f"Configuración {'ABC'[k]}", m, md, aa)))
        res = {}
        for nombre, cfg in cfgs:
            if cfg.metodo == "ked" and ss.deriva is None:
                continue
            r, err = calcular(cfg)
            if err:
                st.error(f"{nombre}: {err}")
            else:
                res[f"{nombre}: {cfg.etiqueta()}"] = r
                for av in r.avisos:
                    st.warning(f"{nombre}: {av}")
        if res:
            vista_c = st.radio("Campo a mostrar", ["log10(K)", "K (m/día)", "Varianza de kriging", "Diferencia entre dos mapas",
                                                   "Perfil / corte"], horizontal=True)
            nombres = list(res)
            if vista_c == "log10(K)":
                st.plotly_chart(plots.mapas_lado_a_lado(G, {k: r.p for k, r in res.items()}, "log10(K)", puntos=PUNTOS),
                                width="stretch")
                st.caption("Misma escala de color en todos los mapas. Pasa el cursor para ver x, y y valor.")
            elif vista_c == "K (m/día)":
                st.plotly_chart(plots.mapas_lado_a_lado(G, {k: r.K for k, r in res.items()}, "K [m/día]", puntos=PUNTOS),
                                width="stretch")
            elif vista_c == "Varianza de kriging":
                con_var = {k: r.varianza for k, r in res.items() if r.varianza is not None}
                if not con_var:
                    st.info("El IDW no produce varianza.")
                else:
                    st.plotly_chart(plots.mapas_lado_a_lado(G, con_var, "σ² [p²]", plots.ESC_VAR, puntos=PUNTOS),
                                    width="stretch")
                    st.caption("Incertidumbre de interpolación bajo el modelo de covarianza; no es incertidumbre total "
                               "de calibración o de predicción.")
                    for k, r in res.items():
                        if r.n_varianza_recortada:
                            st.caption(f"{k}: {r.n_varianza_recortada} celda(s) con varianza ligeramente negativa por "
                                       "redondeo (≈ −1e-16) se recortaron a 0.")
            elif vista_c == "Diferencia entre dos mapas":
                if len(nombres) < 2:
                    st.info("Necesitas al menos dos configuraciones válidas.")
                else:
                    c1, c2, c3 = st.columns(3)
                    n1 = c1.selectbox("Mapa 1", nombres, 0)
                    n2 = c2.selectbox("Mapa 2", nombres, 1)
                    que = c3.radio("Variable", ["log10(K)", "K (m/día)"], horizontal=True)
                    f1, f2 = (res[n1].p, res[n2].p) if que == "log10(K)" else (res[n1].K, res[n2].K)
                    dif = f1 - f2
                    m = plots.rango_comun([dif], simetrico=True)[1]
                    st.plotly_chart(plots.mapa(G, dif, f"Diferencia (Mapa 1 − Mapa 2) en {que}", f"Δ {que}",
                                               PUNTOS, plots.ESC_DIF, -m, m, 0.0, "Δ"), width="stretch")
                    st.caption("Escala divergente centrada en 0: no es un campo de estimación.")
            else:
                c1, c2 = st.columns(2)
                eje = c1.radio("Cortar a lo largo de", ["fila", "col"], horizontal=True,
                               format_func=lambda e: "una fila (este–oeste)" if e == "fila" else "una columna (norte–sur)")
                idx = c2.slider("Índice", 0, (G.nfilas if eje == "fila" else G.ncols) - 1, 7)
                st.plotly_chart(plots.perfiles(G, {k: r.p for k, r in res.items()}, eje, idx, PUNTOS), width="stretch")

# ============================================================ 3 · PESOS
with tabs[3]:
    st.subheader("Pesos de kriging para una celda objetivo")
    if not PTS_OK:
        sin_datos()
    else:
        cfg_p = selector_config("pw", "Método y modelo", "ordinario")
        res_p, err = calcular(cfg_p)
        if err:
            st.error(err)
        else:
            fig = plots.mapa_clicable(G, res_p.p, "Haz clic en una celda para elegirla como objetivo", PUNTOS,
                                      objetivo=ss.objetivo)
            ev = st.plotly_chart(fig, key="map_pesos", on_select="rerun", selection_mode="points", width="stretch")
            ij = celda_de_evento(ev)
            if ij is not None and ij != ss.click_pesos:
                ss.click_pesos = ij
                ss.objetivo = ij
                ss.pop("pw_fila", None)
                ss.pop("pw_col", None)
                st.rerun()
            c1, c2 = st.columns(2)
            fila = c1.number_input("Fila", 0, G.nfilas - 1, ss.objetivo[0], key="pw_fila")
            col = c2.number_input("Columna", 0, G.ncols - 1, ss.objetivo[1], key="pw_col")
            ss.objetivo = (int(fila), int(col))
            try:
                det = detalle_celda(G, PUNTOS, cfg_p, int(fila), int(col), ss.deriva)
            except ErrorKriging as e:
                st.error(str(e))
                det = None
            if det is not None:
                x0, y0 = G.centro(int(fila), int(col))
                st.markdown(f"Celda objetivo: fila {fila}, columna {col} → x = {x0:.0f} m, y = {y0:.0f} m")
                for av in det.avisos:
                    st.warning(av)
                st.dataframe(det.tabla.round(4), hide_index=True, width="stretch")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Suma de pesos", f"{det.suma_pesos:.6f}")
                m2.metric("p̂ = Σ wᵢpᵢ", f"{det.p_hat:.4f}")
                m3.metric("K = 10^p̂ (m/día)", f"{det.K_hat:.3f}")
                m4.metric("Varianza σ² (p²)", "—" if det.varianza is None else f"{det.varianza:.4f}")
                if cfg_p.metodo != "idw" and det.multiplicadores.size:
                    st.markdown("**Multiplicador(es) de Lagrange** (precio de imponer la restricción de insesgadez): "
                                + ", ".join(f"{m:.5f}" for m in det.multiplicadores))
                st.markdown("**Estimación paso a paso**")
                t = det.tabla
                terminos = " + ".join(f"({w:.3f})({p:.2f})" for w, p in zip(t["peso w"], t["p = log10(K)"]))
                st.code(f"p̂ = Σ wᵢ pᵢ = {terminos}\n  = {det.p_hat:.4f}\nK = 10^{det.p_hat:.4f} = {det.K_hat:.3f} m/día")
                cmp_ = st.columns(2)
                cmp_[0].plotly_chart(plots.barras_pesos(t, det.tabla_idw, "Pesos: kriging (color) vs IDW (gris)"),
                                     width="stretch")
                cmp_[1].plotly_chart(plots.distancia_vs_peso(t), width="stretch")
                i_cerca, i_max = int(t["distancia al objetivo (m)"].idxmin()), int(t["peso w"].idxmax())
                if i_cerca != i_max:
                    st.success(f"Aquí el punto más cercano ({t.loc[i_cerca, 'punto']}, peso {t.loc[i_cerca, 'peso w']:.3f}) "
                               f"NO tiene el mayor peso: lo tiene {t.loc[i_max, 'punto']} ({t.loc[i_max, 'peso w']:.3f}).")
                else:
                    st.info(f"En esta celda el punto más cercano ({t.loc[i_cerca, 'punto']}) sí tiene el mayor peso. "
                            "Prueba celdas entre puntos agrupados o detrás de otro punto (efecto pantalla), o el "
                            "experimento 8.")
                with st.expander("¿Por qué el más cercano no siempre domina?"):
                    st.markdown(
                        "El kriging no solo mira la distancia al objetivo (vector **c**): también mira cuánto se parecen "
                        "los puntos **entre sí** (matriz **C**). Si dos puntos son casi redundantes, comparten el peso; "
                        "si uno queda «detrás» de otro, su información ya está contenida en el más cercano y puede recibir "
                        "peso ≈ 0 o negativo (*efecto pantalla*). Matriz C de esta configuración:")
                    if det.C is not None:
                        st.plotly_chart(plots.matriz_covarianza(det.C, det.nombres), width="stretch")
                st.markdown("**Sistema aumentado:** [C 1; 1ᵀ 0][w; λ] = [c; 1] → los pesos suman 1 y λ es el multiplicador "
                            "de Lagrange asociado a esa restricción.")

# ============================================================ 4 · VARIOGRAMA
with tabs[4]:
    st.subheader("Variograma y covarianza")
    comp = st.checkbox("Comparar los tres modelos con los mismos parámetros", False)
    exp_df = None
    fuente = st.radio("Variograma experimental a partir de", ["Puntos piloto actuales", "CSV de datos adicionales"],
                      horizontal=True)
    xy_e = z_e = None
    if fuente == "Puntos piloto actuales":
        if PTS_OK:
            xy_e, z_e = PUNTOS[["x", "y"]].to_numpy(float), PUNTOS["p"].to_numpy(float)
    else:
        fe = st.file_uploader("CSV con x, y y p o K", type="csv", key="up_exp")
        if fe is not None:
            try:
                d_e = data.leer_csv(fe)
                xy_e, z_e = d_e[["x", "y"]].to_numpy(float), d_e["p"].to_numpy(float)
            except Exception as e:
                st.error(str(e))
    if xy_e is not None:
        if len(z_e) >= 8:
            exp_df = variograma_experimental(xy_e, z_e)
            st.caption(f"Variograma omnidireccional de Matheron con {len(z_e)} puntos ({len(z_e) * (len(z_e) - 1) // 2} "
                       "pares). Cada marcador muestra el número de pares; con pocos pares es poco fiable.")
            if len(exp_df) and exp_df["pares"].min() < 5:
                st.warning("Hay intervalos con menos de 5 pares: no ajustes el modelo a ojo con tan poca información.")
        else:
            st.info("Se necesitan al menos 8 puntos para un variograma experimental mínimo. "
                    f"Tienes {len(z_e)}; solo se muestra el modelo teórico.")
    st.plotly_chart(plots.curvas_variograma(VARIO, exp_df, comp), width="stretch")
    st.markdown(f"**Parámetros actuales:** modelo {ETIQUETAS_MODELO[VARIO.modelo]}, nugget = {VARIO.nugget:g} p², "
                f"sill = {VARIO.sill:g} p², sill total = {VARIO.c0:g} p², a = {VARIO.a:g} m, "
                f"razón de anisotropía = {VARIO.razon_anisotropia:g}, azimut = {VARIO.azimut:g}°.")
    st.info(VARIO.explicacion_a())
    filas = []
    for k in (1, 2, 3):
        h = k * VARIO.a
        filas.append({"distancia": f"{k}a = {h:.0f} m", "correlación ρ(h)": float(VARIO.correlacion(h)),
                      "covarianza C(h) (p²)": float(VARIO.covarianza_h(h))})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.caption(f"Rango práctico (ρ = 0.05) a lo largo del eje mayor: ≈ {VARIO.rango_practico():.0f} m. "
               "Se escribe «rango/escala `a`» siempre con su definición según el modelo.")

# ============================================================ 5 · pyEMU
with tabs[5]:
    st.subheader("Comparación con pyEMU")
    ok_py, ver_py = pyemu_compare.pyemu_disponible()
    if not ok_py:
        st.warning("pyEMU no está instalado; el resto de la app funciona con la implementación propia. "
                   "Para habilitar esta pestaña: `pip install pyemu` y reinicia la app.")
    elif not PTS_OK:
        sin_datos()
    elif zonas:
        st.info("Esta comparación usa un solo conjunto de puntos; desactiva «zonas separadas» en la barra lateral.")
    else:
        st.caption(f"pyEMU {ver_py}. Compara los factores de kriging ordinario en una celda con la implementación propia.")
        c1, c2, c3 = st.columns(3)
        fi = c1.number_input("Fila objetivo", 0, G.nfilas - 1, ss.objetivo[0], key="py_f")
        co = c2.number_input("Columna objetivo", 0, G.ncols - 1, ss.objetivo[1], key="py_c")
        qn = c3.checkbox("Eliminar pesos negativos y renormalizar (ambos)", True, key="py_neg")
        x0, y0 = G.centro(int(fi), int(co))
        try:
            tabla, d = pyemu_compare.comparar(PUNTOS, VARIO, x0, y0, qn)
        except Exception as e:
            st.error(f"No se pudo comparar: {e}")
            tabla = None
        if tabla is not None:
            st.dataframe(tabla.round(6), hide_index=True, width="stretch")
            st.metric("Máxima diferencia absoluta entre pesos", f"{d['max_dif']:.2e}")
            if d["hubo_negativos"]:
                txt = ", ".join(d["puntos_negativos"])
                st.warning(f"El kriging ordinario puro tenía pesos negativos en: {txt}. "
                           + ("Se ELIMINARON y renormalizaron en ambas implementaciones (Deutsch, 1996)."
                              if qn else "En esta comparación NO se eliminaron."))
            else:
                st.success("No había pesos negativos en esta celda; eliminar o no eliminar no cambia nada aquí.")
            if d["max_dif"] < 1e-8:
                st.success("Con los mismos supuestos (modelo, a, sill, nugget, anisotropía y eliminación de negativos), "
                           "los pesos coinciden a precisión de máquina.")
            else:
                st.error("Los pesos NO coinciden: revisa que el modelo, `a`, nugget, azimut y la eliminación de "
                         "negativos sean iguales en ambos lados.")
            st.markdown(f"**Estimación:** p̂ propio = {d['p_propio']:.4f}, p̂ pyEMU = {d['p_pyemu']:.4f} → "
                        f"K = {d['K_propio']:.3f} m/día (interpolando en log10). Si los factores de pyEMU se aplicaran "
                        f"directamente a K física: {d['K_pyemu_lineal']:.3f} m/día (interpolación aritmética, distinta).")
            st.markdown(
                f"**Varianza:** pyEMU `err_var` = {d['err_var_pyemu']:.4f}; nuestra σ² con los mismos pesos crudos = "
                f"{d['var_propia_cruda']:.4f}. En pyEMU {d['version']} `err_var = sill + μ − Σwc` (se verificó que "
                f"reproduce {d['err_var_pyemu_reproducida']:.4f} con μ = {d['mu_propio']:.5f}), mientras que la "
                "expresión estándar es σ² = C(0) − Σwc − μ, que coincide con el error cuadrático medio exacto. "
                "Difieren en 2μ. Además `err_var` se calcula antes de eliminar pesos negativos. Por eso no "
                "se afirma equivalencia de varianzas.")
            with st.expander("Otras fuentes de diferencia a vigilar"):
                st.markdown("- **Transformación logarítmica:** pyEMU no transforma; tú decides si aplicas los factores a "
                            "K o a log10 K.\n- **Radio de búsqueda y número máximo de puntos:** aquí se usan todos los puntos.\n"
                            "- **Zonas:** pyEMU puede kriging por zona (`pt_zone`); aquí se desactivó.\n"
                            "- **Orientación:** pyEMU usa `bearing` (azimut desde el norte, horario), igual que este paquete.\n"
                            "- **Convención de variograma:** `a` es e-folding (exp), escala (gauss) y rango (esf), igual aquí.")

# ============================================================ 6 · FLOPY / MODFLOW 6
with tabs[6]:
    st.subheader("Del campo de K al flujo subterráneo (opcional)")
    st.markdown(
        "**Modelo conceptual (valores DIDÁCTICOS):** capa confinada de espesor b, régimen estacionario; carga constante "
        "en las columnas oeste y este, fronteras norte y sur impermeables; recarga uniforme R (m/día) y un pozo de bombeo "
        "(Q < 0, m³/día). T = K·b (m²/día), h en metros; abatimiento = h sin bombeo − h con bombeo.")
    if not ok_fp:
        st.warning("FloPy no está instalado: `pip install flopy`. La parte geoestadística sigue funcionando.")
    if EXE is None:
        st.warning("No se encontró el ejecutable de MODFLOW 6. Instálalo con `get-modflow <carpeta>` (incluido en FloPy) "
                   "y añade esa carpeta al PATH, o escribe la ruta en la barra lateral.")
    if not PTS_OK:
        sin_datos()
    else:
        c1, c2, c3, c4 = st.columns(4)
        hw = c1.number_input("Carga oeste (m)", 0.0, 500.0, 30.0, 1.0)
        he = c2.number_input("Carga este (m)", 0.0, 500.0, 20.0, 1.0)
        rec = c3.number_input("Recarga R (m/día)", 0.0, 0.1, 0.001, 0.0005, format="%.4f")
        esp = c4.number_input("Espesor b (m)", 1.0, 1000.0, 50.0, 5.0)
        d1, d2, d3 = st.columns(3)
        qw = d1.number_input("Caudal del pozo Q (m³/día, negativo = bombeo)", -50000.0, 0.0, -1000.0, 100.0)
        fw = d2.number_input("Fila del pozo", 0, G.nfilas - 1, 7)
        cw = d3.number_input("Columna del pozo", 0, G.ncols - 1, 10)
        PAR = flow.ParametrosFlujo(hw, he, rec, qw, int(fw), int(cw), esp)
        opciones = {"IDW (potencia 2)": ConfigInterp("idw", VARIO, 2.0, separar_zonas=zonas, x_corte=x_corte),
                    "Kriging ordinario": ConfigInterp("ordinario", VARIO, separar_zonas=zonas, x_corte=x_corte),
                    "Kriging ordinario sin pesos negativos": ConfigInterp("ordinario", VARIO, quitar_negativos=True,
                                                                          separar_zonas=zonas, x_corte=x_corte),
                    "Kriging simple": ConfigInterp("simple", VARIO, separar_zonas=zonas, x_corte=x_corte),
                    "Kriging universal (plano)": ConfigInterp("universal", VARIO, tendencia="plano",
                                                              separar_zonas=zonas, x_corte=x_corte)}
        sel = st.multiselect("Campos de K a simular", list(opciones), ["IDW (potencia 2)", "Kriging ordinario"])
        listo = ok_fp and EXE is not None
        if st.button("Ejecutar simulación", disabled=not listo, type="primary"):
            ss.flujo_res = {}
            with st.spinner("Ejecutando MODFLOW 6…"):
                for nombre in sel:
                    r, err = calcular(opciones[nombre])
                    if err:
                        st.error(f"{nombre}: {err}")
                        continue
                    try:
                        out = flow.simular(r.K, G, PAR, EXE)
                        out["K"] = r.K
                        ss.flujo_res[nombre] = out
                    except Exception as e:
                        st.error(f"{nombre}: {e}")
        st.caption("La simulación solo corre al pulsar el botón; mover un control nunca la ejecuta.")
        R = ss.flujo_res
        if R:
            st.plotly_chart(plots.mapas_lado_a_lado(G, {k: v["K"] for k, v in R.items()}, "K [m/día]", comun=True),
                            width="stretch")
            st.plotly_chart(plots.mapas_lado_a_lado(G, {k: v["h_bombeo"] for k, v in R.items()}, "h [m]",
                                                    plots.ESC_EST), width="stretch")
            st.plotly_chart(plots.mapas_lado_a_lado(G, {k: v["abatimiento"] for k, v in R.items()}, "abatimiento [m]",
                                                    "Blues"), width="stretch")
            if len(R) >= 2:
                ks = list(R)
                x1 = st.selectbox("Comparar abatimiento: A", ks, 0, key="fl_a")
                x2 = st.selectbox("…con B", ks, 1, key="fl_b")
                dif = R[x1]["abatimiento"] - R[x2]["abatimiento"]
                m = plots.rango_comun([dif], True)[1]
                st.plotly_chart(plots.mapa(G, dif, "Diferencia de abatimiento (A − B)", "Δ [m]", None, plots.ESC_DIF,
                                           -m, m, 0.0, "Δ"), width="stretch")
                st.caption(f"Máxima diferencia absoluta de abatimiento: {np.abs(dif).max():.3f} m.")

# ============================================================ 7 · EXPERIMENTOS
with tabs[7]:
    st.subheader("Experimentos didácticos")
    ids = {f"{e.id}. {e.titulo}": e for e in experiments.ESCENARIOS}
    esc = ids[st.selectbox("Escenario", list(ids))]
    st.markdown(f"**Pregunta de aprendizaje:** {esc.pregunta}")
    st.markdown(f"**Parámetros que se modifican:** {esc.modifica}")
    if esc.usa_datos_propios:
        st.info("Este escenario usa un conjunto de datos didáctico propio (inventado), no los puntos del proyecto.")
    params = {}
    if esc.controles:
        cc = st.columns(len(esc.controles))
        for c, (k, (lab, lo, hi, df_, st_)) in zip(cc, esc.controles.items()):
            if isinstance(df_, int) and isinstance(st_, int):
                params[k] = c.number_input(lab, int(lo), int(hi), int(df_), int(st_), key=f"x{esc.id}_{k}")
            else:
                params[k] = c.number_input(lab, float(lo), float(hi), float(df_), float(st_), key=f"x{esc.id}_{k}")
    st.markdown(f"**Predice antes de ejecutar:** {esc.prediccion}")
    pred = st.radio("Tu predicción", esc.opciones, index=None, key=f"pred_{esc.id}", label_visibility="collapsed")
    bloqueo = (not PTS_OK and not esc.usa_datos_propios) or (esc.requiere_modflow and (EXE is None or not ok_fp))
    if esc.requiere_modflow and (EXE is None or not ok_fp):
        st.warning("Este escenario necesita FloPy y el ejecutable de MODFLOW 6 (ver barra lateral).")
    if st.button("Ejecutar escenario", disabled=bloqueo, type="primary", key=f"run_{esc.id}"):
        try:
            flujo_ = {"par": flow.ParametrosFlujo(), "exe": EXE} if esc.requiere_modflow else None
            ss[f"res_{esc.id}"] = (experiments.ejecutar(esc, G, PUNTOS, BASE, params, ss.deriva, flujo_), pred)
        except (ErrorKriging, ValueError, RuntimeError) as e:
            st.error(str(e))
            ss.pop(f"res_{esc.id}", None)
    guard = ss.get(f"res_{esc.id}")
    if guard:
        res, pred_run = guard
        if pred_run is None:
            st.caption("No registraste una predicción antes de ejecutar.")
        elif esc.opciones.index(pred_run) == esc.correcta:
            st.success("Tu predicción coincide con lo que ocurre.")
        else:
            st.error(f"Tu predicción no coincidió. Lo correcto: {esc.opciones[esc.correcta]}.")
        for av in res["avisos"]:
            st.caption(av)
        casos, P = res["casos"], res["puntos_usados"]
        st.markdown("**Resultado visual**")
        if esc.extra != "sensibilidad":
            st.plotly_chart(plots.mapas_lado_a_lado(G, {c["etiqueta"]: c["resultado"].p for c in casos}, "log10(K)",
                                                    puntos=P), width="stretch")
            vcs = {c["etiqueta"]: c["resultado"].varianza for c in casos if c["resultado"].varianza is not None}
            if vcs:
                st.plotly_chart(plots.mapas_lado_a_lado(G, vcs, "σ² [p²]", plots.ESC_VAR, puntos=P,
                                                        comun=esc.id != 2), width="stretch")
                if esc.id == 2:
                    st.caption("Cada mapa de varianza tiene su propia escala para que se vea su forma; compara las "
                               "magnitudes en la tabla de métricas.")
            if len(casos) >= 2:
                difs = {f"{c['etiqueta']} − {casos[0]['etiqueta']}": c["resultado"].p - casos[0]["resultado"].p
                        for c in casos[1:]}
                st.plotly_chart(plots.mapas_lado_a_lado(G, difs, "Δ log10(K)", plots.ESC_DIF, puntos=P, simetrico=True,
                                                        zmid=0.0), width="stretch")
                st.caption("Mapas de DIFERENCIA (escala divergente centrada en 0); no son campos de estimación.")
        st.dataframe(res["metricas"].round(4), hide_index=True, width="stretch")
        ex = res["extra"]
        if esc.extra == "curvas":
            st.plotly_chart(plots.curvas_variograma(BASE.vario, None, True), width="stretch")
        if esc.extra == "pesos" and "detalles" in ex:
            fi, co = res["objetivo"]
            st.markdown(f"**Pesos en la celda objetivo (fila {fi}, columna {co})**")
            cs = st.columns(len(ex["detalles"]))
            for c, dt, caso in zip(cs, ex["detalles"], casos):
                c.plotly_chart(plots.barras_pesos(dt.tabla, None, caso["etiqueta"]), width="stretch")
            if esc.id == 7:
                st.write(f"Peso mínimo en el dominio: {ex['peso_min']:.4f} (celda mostrada). "
                         f"Varianza con pesos recortados ≥ varianza óptima en todas las celdas: {ex['var_aumenta']}.")
            if esc.id == 8:
                gr = ex["grupo"]
                sumas = [float(dt.tabla.loc[dt.tabla["punto"].isin(gr), "peso w"].sum()) for dt in ex["detalles"]]
                st.write(f"Peso total del grupo {gr}: IDW = {sumas[0]:.3f}, kriging = {sumas[1]:.3f}.")
        if esc.extra == "razonK":
            m = ex["razonK"]
            st.plotly_chart(plots.mapa(G, m, f"Razón K_nuevo / K_base (punto {ex['punto']})", "razón",
                                       puntos=P, escala="RdBu", zmin=min(m.min(), 2 - m.max()),
                                       zmax=max(m.max(), 2 - m.min()), zmid=1.0, hover="razón"), width="stretch")
            st.write(f"Δlog10(K) máximo = {np.nanmax(ex['delta_p']):.4f} (esperado log10(factor) = "
                     f"{ex['delta_p_esperado']:.4f}). Razón mínima de K = {m.min():.3f}, máxima = {m.max():.3f}.")
        if esc.extra == "covmaps":
            cs = st.columns(len(casos))
            for c, caso in zip(cs, casos):
                c.plotly_chart(plots.mapa_covarianza_objetivo(G, caso["cfg"].vario, res["objetivo"], caso["etiqueta"]),
                               width="stretch")
        if esc.extra == "sensibilidad":
            st.plotly_chart(plots.barras_sensibilidad(ex["sens"], "Sensibilidad local ∂h/∂p por punto piloto"),
                            width="stretch")
            st.dataframe(ex["sens"].round(4), hide_index=True, width="stretch")
            st.caption("Diferencias finitas hacia adelante (Δp = 0.1) con los valores didácticos del panel de flujo "
                       "por defecto. El Jacobiano de cada observación es una sola fila (rango 1).")
        if esc.extra == "perfil":
            st.plotly_chart(plots.perfiles(G, {c["etiqueta"]: c["resultado"].p for c in casos}, "fila", ex["fila"], P),
                            width="stretch")
            st.caption("Corte este–oeste por la fila central; el contacto está en x = 500 m.")
        st.markdown("**Interpretación**")
        st.write(esc.interpretacion)
        st.markdown("**Para reflexionar**")
        for q in esc.reflexion:
            st.markdown(f"- {q}")
