# ctcxplorer V1
# -----------------------------------------------------------------------------
# Requiere: Python ≥ 3.9, duckdb, pandas, matplotlib, gradio (≥ 3.32 ó ≥ 4.*)
# Instalar rápidamente con:
#     pip install duckdb pandas matplotlib gradio
# -----------------------------------------------------------------------------

from __future__ import annotations

import gzip
import io
import inspect
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Union, Tuple

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import gradio as gr
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------

FileLike = Union[str, Path, "gradio.components.file.NamedString", bytes, io.BufferedIOBase]


def _read_text_from_fileobj(file_obj: FileLike) -> str:
    """Devuelve el texto del objeto subido, sea cual sea su tipo.

    * Acepta:
      - objetos con `.read()` (BytesIO, file handles…)
      - rutas (`pathlib.Path`, `str`)
      - `gradio.FileData` (dicts con claves `path`, `name`, `data`)
    * Reconoce automáticamente `.gz`.
    """
    # 🌟 Atajo: si *file_obj* es str/Path y apunta a un archivo existente, lo abrimos directamente
    if isinstance(file_obj, (str, Path)) and Path(file_obj).is_file():
        path = Path(file_obj)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        return path.read_text(encoding="utf-8", errors="ignore")
    # ------------------------------------------------------------------
    # 1) Soporte para FileData (dict) que devuelve gradio.components.Files
    # ------------------------------------------------------------------
    if isinstance(file_obj, dict):
        # Gradio FileData puede traer 'data' (bytes) o 'path'/'name'.
        data_bytes = file_obj.get("data")
        if isinstance(data_bytes, bytes) and data_bytes:
            try:
                return gzip.decompress(data_bytes).decode("utf-8", errors="ignore")
            except gzip.BadGzipFile:
                return data_bytes.decode("utf-8", errors="ignore")

        # Si no hay bytes, intentamos con la ruta siempre que sea un archivo.
        candidate_path = file_obj.get("path") or file_obj.get("name")
        if candidate_path and Path(candidate_path).is_file():
            file_obj = Path(candidate_path)
        else:
            raise ValueError(
                "FileData sin bytes y la ruta proporcionada no es un archivo: "
                f"{candidate_path}"
            )

    # ------------------------------------------------------------------
    # 2) Caso: `file_obj` tiene atributo `.name` que apunta a un archivo
    # ------------------------------------------------------------------
    _possible_path = getattr(file_obj, "name", None)
    if _possible_path and Path(_possible_path).is_file():
        file_obj = Path(_possible_path)

    # ------------------------------------------------------------------
    # 3) Flujo para objetos con `.read()` (ya abiertos)
    # ------------------------------------------------------------------
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        if isinstance(data, bytes):
            try:
                return gzip.decompress(data).decode("utf-8", errors="ignore")
            except gzip.BadGzipFile:
                return data.decode("utf-8", errors="ignore")
        return str(data)

    # ------------------------------------------------------------------
    # 4) Interpretar file_obj como ruta
    # ------------------------------------------------------------------
    path = Path(str(file_obj))
    if path.is_dir():
        raise IsADirectoryError(f"Se esperaba un archivo pero se recibió un directorio: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="ignore")


# -----------------------------------------------------------------------------
# PARSERS – #I → tabla circuitos; #T → tabla trenes
# -----------------------------------------------------------------------------

def _parse_circuitos(lines: List[str]) -> pd.DataFrame:
    """Extrae filas #I cuyo *name* (campo 7) contiene "_V" y devuelve (id, name)."""
    circuitos: Dict[int, str] = {}
    for line in lines:
        if not line.startswith("#I"):
            continue
        parts = line.rstrip().split(",")
        if len(parts) < 7:  # necesitamos al menos 7 columnas (índices 0‑6)
            continue
        try:
            cid = int(parts[4])  # campo 5 (1‑based)
        except ValueError:
            continue
        name = parts[6].strip()  # campo 7 (1‑based)
        if "_V" not in name:
            continue
        circuitos[cid] = name

    return pd.DataFrame(list(circuitos.items()), columns=["id", "name"])


def _safe_date(d: str) -> date:
    return datetime.strptime(d, "%d-%m-%y").date()


def _parse_trenes(lines: List[str]) -> pd.DataFrame:
    """Extrae todas las líneas #T y, para cada (train, tcid), conserva solo la primera hora."""
    rows: List[Tuple[datetime, int, int]] = []
    for line in lines:
        if not line.startswith("#T"):
            continue
        parts = line.rstrip().split(",")
        if len(parts) < 4:
            continue
        try:
            ts = datetime.combine(_safe_date(parts[1]), datetime.strptime(parts[2], "%H:%M:%S").time())
        except Exception:
            continue
        path_parts = parts[3].split("-")
        if len(path_parts) < 3:
            continue
        try:
            tcid = int(path_parts[1])
            train = int(path_parts[2])
        except ValueError:
            continue
        rows.append((ts, tcid, train))
    df = pd.DataFrame(rows, columns=["time", "tcid", "train"])
    # Mantener solo la primera vez que cada tren pasa por cada tcid
    df = df.sort_values("time").drop_duplicates(subset=["train", "tcid"], keep="first")
    return df

# -----------------------------------------------------------------------------
# MANDOS – parser
# -----------------------------------------------------------------------------
def _parse_mandos(lines: List[str]) -> pd.DataFrame:
    """Extrae todas las líneas #M (mandos).  
    Devuelve columnas: time (datetime), raw (línea completa)."""
    rows: List[Tuple[datetime, str]] = []
    for line in lines:
        if not line.startswith("#M"):
            continue
        parts = line.rstrip().split(",", 3)  # los tres primeros campos separan fecha + hora
        if len(parts) < 4:
            continue
        try:
            ts = datetime.combine(_safe_date(parts[1]), datetime.strptime(parts[2], "%H:%M:%S").time())
        except Exception:
            continue
        rows.append((ts, line.rstrip()))
    return pd.DataFrame(rows, columns=["time", "raw"])


# -----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE TABLAS (DuckDB)
# -----------------------------------------------------------------------------

def build_duckdb_tables(file_objs: Union[FileLike, List[FileLike]]) -> duckdb.DuckDBPyConnection:
    """Crea tablas en memoria combinando uno o varios ficheros de log.

    *Admite*:
      • Un único objeto (str, Path, dict FileData, BytesIO…)  
      • Una lista con cualquier combinación de los anteriores
    """
    # Normalizar a lista
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    dfs_circ = []
    dfs_tren = []
    dfs_mand = []
    for file_obj in file_objs:
        text = _read_text_from_fileobj(file_obj)
        if not text:
            continue
        lines = text.splitlines()
        dfs_circ.append(_parse_circuitos(lines))
        dfs_tren.append(_parse_trenes(lines))
        dfs_mand.append(_parse_mandos(lines))

    if not dfs_tren:
        raise ValueError("Los archivos están vacíos o no se pudieron leer.")

    df_circ = (
        pd.concat(dfs_circ, ignore_index=True)
        .drop_duplicates(subset=["id", "name"])
        .sort_values("id")
        .reset_index(drop=True)
    )
    df_tren = pd.concat(dfs_tren, ignore_index=True)
    df_mand = pd.concat(dfs_mand, ignore_index=True) if dfs_mand else pd.DataFrame(columns=["time", "raw"])

    con = duckdb.connect(database=":memory:")
    con.register("circuitos_df", df_circ)
    con.register("trenes_raw", df_tren)
    con.register("mandos_raw", df_mand)

    con.execute("CREATE TABLE circuitos AS SELECT * FROM circuitos_df;")
    con.execute(
        """
        CREATE TABLE trenes AS
        SELECT r.time, r.tcid, r.train, c.name AS tcname, c.id
        FROM trenes_raw AS r
        LEFT JOIN circuitos AS c ON r.tcid = c.id;
        """
    )
    con.execute(
        """
        CREATE TABLE mandos AS
        SELECT time, raw
        FROM mandos_raw;
        """
    )
    return con


 # -----------------------------------------------------------------------------
 # VELOCIDADES
 # -----------------------------------------------------------------------------
def _calc_velocidades(
    con: duckdb.DuckDBPyConnection,
    origen_tc: str,
    origen_km: float,
    destino_tc: str,
    destino_km: float,
) -> pd.DataFrame:
    """Devuelve la velocidad media (km/h) de cada tren entre dos tcname.

    * `origen_tc`, `destino_tc`: nombres de circuito (tcname)
    * `origen_km`, `destino_km`: posiciones en kilómetros (pueden incluir decimales)
    """
    if origen_tc == destino_tc:
        raise ValueError("El origen y el destino deben ser distintos.")
    distancia = abs(destino_km - origen_km)
    if distancia == 0:
        raise ValueError("Las distancias de origen y destino no pueden ser iguales.")

    query = '''
        SELECT
            train,
            MAX(CASE WHEN tcname = ? THEN time END) AS t_origen,
            MAX(CASE WHEN tcname = ? THEN time END) AS t_destino
        FROM trenes
        WHERE tcname IN (?, ?)
        GROUP BY train
        HAVING t_origen IS NOT NULL AND t_destino IS NOT NULL
    '''
    df = con.execute(query, [origen_tc, destino_tc, origen_tc, destino_tc]).fetchdf()
    if df.empty:
        return df

    df["horas"] = (df["t_destino"] - df["t_origen"]).dt.total_seconds() / 3600.0
    df["vel_kmh"] = distancia / df["horas"]
    return df[["train", "vel_kmh"]].sort_values("train")

# -----------------------------------------------------------------------------
# MANDOS – utilidades para colorear y agrupar
# -----------------------------------------------------------------------------
_COLOR_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]

def _colorize_log_line(line: str) -> str:
    """Devuelve la línea #M con cada campo coloreado en HTML."""
    fields = line.split(",")
    html_parts = []
    for i, fld in enumerate(fields):
        color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
        html_parts.append(f"<span style='color:{color};font-weight:bold'>{fld}</span>")
    # Coma separadora en gris
    sep = "<span style='color:#555'>,</span>"
    return sep.join(html_parts)

# -----------------------------------------------------------------------------
# Callback para la pestaña Mandos
# -----------------------------------------------------------------------------

def _build_mask(df: pd.DataFrame, expr: str) -> pd.Series:
    """Devuelve un boolean Series filtrando df['raw'] según *expr*.

    Sintaxis sencilla (sin paréntesis):
      - AND:  espacio o "&"   →  todos los términos deben aparecer
      - OR :  "|"             →  cualquiera de los grupos OR puede cumplirse
      - NOT:  "!" delante     →  el término NO debe aparecer

    Ej.:  'PUESTO5 & RA | !ERROR'
    """
    expr = expr.strip()
    if not expr:
        return pd.Series(True, index=df.index)

    import re
    or_parts = [p.strip() for p in re.split(r"\|", expr)]
    mask_total = pd.Series(False, index=df.index)

    for part in or_parts:
        if not part:
            continue
        and_terms = [t.strip() for t in re.split(r"&|\s+", part) if t.strip()]
        mask_and = pd.Series(True, index=df.index)
        for term in and_terms:
            neg = term.startswith("!")
            term_clean = term[1:] if neg else term
            term_clean = term_clean.strip('"\'')
            term_match = df["raw"].str.contains(term_clean, case=False, regex=False)
            mask_and &= ~term_match if neg else term_match
        mask_total |= mask_and

    return mask_total


def generar_mandos(log_files, filtro_txt):
    """Callback para la pestaña 'Mandos'."""
    if isinstance(log_files, (str, Path)):
        files = [log_files]
    elif isinstance(log_files, list) and log_files:
        files = log_files
    else:
        return "⚠️ Suba al menos un fichero de log."

    filtro = (filtro_txt or "").strip()
    try:
        con = build_duckdb_tables(files)
        df = con.execute("SELECT time, raw FROM mandos ORDER BY time").fetchdf()

        if df.empty:
            return "⚠️ No se encontraron líneas."

        # Aplicar filtro de expresión
        mask = _build_mask(df, filtro)
        df = df[mask]

        if df.empty:
            return "⚠️ Ninguna línea coincide con el filtro."

        # Colorear para salida HTML
        html_lines = [_colorize_log_line(r) for r in df["raw"]]
        html_out = "<br>".join(html_lines)
        return html_out
    except Exception as exc:
        return f"⚠️ {exc}"


# -----------------------------------------------------------------------------
# GRÁFICO
# -----------------------------------------------------------------------------

def plot_trenes(con: duckdb.DuckDBPyConnection, tcnames: List[str]):
    """Devuelve una figura Plotly interactiva (X = tiempo, Y = circuito).
    * Orden de Y = orden de selección.
    * Al pasar el ratón se muestra la hora exacta y el tren.
    """
    if not tcnames:
        raise ValueError("Debe elegir al menos un elemento.")

    placeholders = ",".join("?" * len(tcnames))
    df = con.execute(
        f"""
        SELECT time, train, tcname
        FROM trenes
        WHERE tcname IN ({placeholders})
        ORDER BY time
        """,
        tcnames,
    ).fetchdf()

    if df.empty:
        raise ValueError("No hay datos que coincidan con la selección.")

    # Mantener orden de selección en el eje Y
    df["tcname"] = pd.Categorical(df["tcname"], categories=tcnames, ordered=True)

    fig = px.line(
        df,
        x="time",
        y="tcname",
        color="train",
        markers=True,
        category_orders={"tcname": tcnames},
        labels={"time": "Hora", "tcname": "Circuito", "train": "Tren"},
        title="Paso de cada tren por circuito (interactivo)",
    )

    # Ajustar tamaño: ancho 100% (lo gestiona Gradio), altura proporcional
    height = max(400, int(60 * len(tcnames)))
    fig.update_layout(height=height, legend_title_text="Tren", hovermode="closest")
    fig.update_yaxes(autorange="reversed")  # Para que el primero aparezca arriba
    return fig


# -----------------------------------------------------------------------------
# INTERFAZ GRADIO
# -----------------------------------------------------------------------------

def _dropdown_update(choices: List[str]):
    try:
        return gr.update(choices=choices, value=[])
    except Exception:
        return {"choices": choices, "value": []}


def update_elementos_dropdown(log_files):
    # Normalizar: el componente Files devuelve lista cuando file_count="multiple"
    if isinstance(log_files, (str, Path)):
        file_path = log_files
    elif isinstance(log_files, list) and log_files:
        file_path = log_files[0]
    else:
        print("[DEBUG] Sin fichero cargado → choices vacíos")
        return _dropdown_update([])

    try:
        text = _read_text_from_fileobj(file_path)
        df_circ = _parse_circuitos(text.splitlines())
        names = df_circ["name"].tolist()
        print(f"[DEBUG] Elementos detectados: {len(names)}")
    except Exception as e:
        print("[ERROR] update_elementos_dropdown →", e)
        names = []

    return _dropdown_update(names)


def generar_grafico(log_files, seleccionados: List[str]):
    if isinstance(log_files, (str, Path)):
        files = [log_files]
    elif isinstance(log_files, list) and log_files:
        files = log_files
    else:
        return "⚠️ Suba al menos un fichero de log.", None
    try:
        con = build_duckdb_tables(files)
        fig = plot_trenes(con, seleccionados)
        return "", fig
    except Exception as exc:
        return f"⚠️ {exc}", None


def generar_velocidades(log_files, origen_tc, origen_km, destino_tc, destino_km):
    """Callback de Gradio para la pestaña 'velocidades'."""
    # Normalizar entrada de fichero igual que en generar_grafico
    if isinstance(log_files, (str, Path)):
        files = [log_files]
    elif isinstance(log_files, list) and log_files:
        files = log_files
    else:
        return "⚠️ Suba al menos un fichero de log.", None

    try:
        con = build_duckdb_tables(files)
        df = _calc_velocidades(con, origen_tc, float(origen_km), destino_tc, float(destino_km))
        if df.empty:
            return "⚠️ No se encontraron trenes que pasen por ambos puntos.", None
        return "", df
    except Exception as exc:
        return f"⚠️ {exc}", None



with gr.Blocks(title="Explorador de logs de CTC") as demo:
    gr.Markdown(
        "# Explorador de logs de CTC\n"
        "Suba un fichero de log y elija la funcionalidad."
    )

    # ---------- Entrada común ----------
    log_files = gr.Files(label="Ficheros de log (.out, .gz)", file_count="multiple")

    with gr.Tabs():
        # ==================== Pestaña 1: Trenes ====================
        with gr.TabItem("Trenes"):
            _dd_kwargs = dict(label="Elementos (tcname)", choices=[], multiselect=True)
            if "filterable" in inspect.signature(gr.Dropdown).parameters:
                _dd_kwargs["filterable"] = True
            elementos_dd = gr.Dropdown(**_dd_kwargs)
            log_files.change(update_elementos_dropdown, inputs=log_files, outputs=elementos_dd)

            generar_btn = gr.Button("Generar gráfico")
            salida_msg = gr.Markdown()
            salida_plot = gr.Plot()
            generar_btn.click(
                generar_grafico,
                inputs=[log_files, elementos_dd],
                outputs=[salida_msg, salida_plot],
            )

        # ==================== Pestaña 2: Velocidades ====================
        with gr.TabItem("Velocidades"):
            _drop_kwargs = dict(choices=[])
            if "filterable" in inspect.signature(gr.Dropdown).parameters:
                _drop_kwargs["filterable"] = True

            origen_dd = gr.Dropdown(label="Origen (tcname)", **_drop_kwargs)
            destino_dd = gr.Dropdown(label="Destino (tcname)", **_drop_kwargs)

            # Ambas listas se actualizan al cargar fichero
            log_files.change(update_elementos_dropdown, inputs=log_files, outputs=origen_dd)
            log_files.change(update_elementos_dropdown, inputs=log_files, outputs=destino_dd)

            with gr.Row():
                origen_km = gr.Number(label="Distancia origen (km)", value=0.0)
                destino_km = gr.Number(label="Distancia destino (km)", value=0.0)

            analizar_btn = gr.Button("Analizar")
            salida_vel_msg = gr.Markdown()
            salida_vel_df = gr.Dataframe(label="Velocidades (km/h)", interactive=False)

            analizar_btn.click(
                generar_velocidades,
                inputs=[log_files, origen_dd, origen_km, destino_dd, destino_km],
                outputs=[salida_vel_msg, salida_vel_df],
            )

        # ==================== Pestaña 3: Mandos ====================
        with gr.TabItem("Mandos"):
            filtro_txt = gr.Text(label="Cadena a buscar", placeholder="Ej.: PUESTO5 & RA | !ERROR")
            filtrar_btn = gr.Button("Filtrar")
            salida_mand_html = gr.HTML()

            filtrar_btn.click(
                generar_mandos,
                inputs=[log_files, filtro_txt],
                outputs=[salida_mand_html],
            )

    gr.Markdown(
        "---\n"
        "*La pestaña **Trenes** muestra el paso por circuito; "
        "la pestaña **Velocidades** estima la velocidad media entre dos posiciones; "
        "y la pestaña **Mandos** permite buscar eventos con expresiones simples "
        "(`AND`, `OR`, `NOT`).*"
    )

# Iniciar la aplicación
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
