"""
App: Ventas x Item – UR/UB robusto por día x sede (multi-item hasta 10)

Cambios clave:
- Parseo de fecha explícito para evitar warnings de Pandas.
- Nunca escribir '-' en columnas numéricas (se formatea en pantalla con Styler).
- Streamlit: usar width='stretch' en st.dataframe.
"""

import io
import re
import pandas as pd
import streamlit as st

# ===================== Config: nombres de sedes (y orden) =====================
SEDE_NAME_MAP = {
    "mercamio": {"1": "Calle 5ta","2": "La 39","3": "Plaza","4": "Jardín","5": "C. Sur","6": "Palmira"},
    "mtodo": {"1": "Floresta","2": "Floralia","3": "Guadua"},
    "bogota": {"1": "Calle 80","2": "Chía"},
}

MONTH_ABBR_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

# ===================== Aliases =====================
ALIASES = {
    "empresa": {"empresa","compania","compañia","company"},
    "fecha_dcto": {"fecha_dcto","fecha","fecha_doc","fecha documento","fecha_documento"},
    "id_co": {"id_co","sede","tienda","local","centro"},
    "id_item": {"id_item","item","codigo_item","sku","cod_item"},
    "und_dia": {"und_dia","und","unid","unidades","cantidad","cant"},
    "descripcion": {"descripcion","descripción","desc","detalle"},
    "ub_unit": {"ub_unit","unidad","unidad_medida","um","u.m.","u_m"},
    "ub_factor": {"ub_factor","factor","contenido","presentacion","presentación","unid_x","unidx","und_pack","unidades_por","pack","x"},
}

# ===================== Helpers de nombres/estandarización =====================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.to_series().astype(str).str.strip().str.lower()
    mapping = {}
    for canon, variants in ALIASES.items():
        for v in variants:
            hit = cols[cols == v]
            if not hit.empty:
                mapping[hit.index[0]] = canon
    cleaned = {old: (mapping.get(old, cols.loc[old])) for old in df.columns}
    return df.rename(columns=cleaned)

def unify_empresa(v: str) -> str:
    s = str(v).strip().lower()
    if s in ("metodo",): return "mtodo"
    if s in ("bogota","bogota dc","bogota d.c."): return "bogota"
    return s

def sede_key_to_name(sede_key: str) -> str:
    emp, co = sede_key.split("|", 1)
    emp = emp.strip().lower()
    co = co.strip()
    if emp in SEDE_NAME_MAP and co in SEDE_NAME_MAP[emp]:
        return SEDE_NAME_MAP[emp][co]
    return f"{emp}-{co}"

# ===================== Fechas robustas =====================
def parse_dates_strict(series: pd.Series) -> pd.Series:
    """Intenta formatos comunes; si no, fallback con dayfirst=True. Devuelve datetime (nullable)."""
    s = series.astype(str).str.strip()
    formats = ("%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%d/%m/%y","%Y/%m/%d")
    for fmt in formats:
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        if dt.notna().mean() >= 0.8:  # si 80%+ matchea, adoptamos ese formato
            return dt
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def extract_day_if_possible(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    day_str = raw.str.extract(r"^\s*(\d{1,2})")[0]
    day = pd.to_numeric(day_str, errors="coerce")
    return day.where((day >= 1) & (day <= 31)).astype("Int64")

# ===================== Parsing numérico =====================
def parse_und_dia_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    both = s.str.contains(r"\.") & s.str.contains(r",")
    s = s.where(~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    only_comma = s.str.contains(",") & ~s.str.contains(r"\.")
    s = s.where(~only_comma, s.str.replace(",", "", regex=False))
    out = pd.to_numeric(s, errors="coerce").fillna(0)
    # usamos Int64 (nullable) para evitar errores cuando haya NA en algún flujo
    return out.round().astype("Int64")

# ===================== Parsers UB: peso / volumen / conteo =====================
NUM = r"(\d+(?:[.,]\d+)?)"
WEIGHT_RE = re.compile(rf"{NUM}\s*(kg|kilo|kilogramo|kilogramos|kg\.)|{NUM}\s*(g|gr|gramo|gramos|g\.)", re.IGNORECASE)
VOL_RE    = re.compile(rf"{NUM}\s*(l|lt|litro|litros|l\.)|{NUM}\s*(cl|centilitro|centilitros)|{NUM}\s*(ml|mililitro|mililitros)", re.IGNORECASE)
COUNT_RE  = re.compile(r"(?:x\s*(\d{1,5}))|(?:\b(\d{1,5})\s*(?:u|un|und|uds|unid|unids|unidades|huevos?)\b)", re.IGNORECASE)

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

def extract_weight_grams(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = WEIGHT_RE.search(text)
    if not m: return None
    if m.group(1):  # kilos
        kg = _to_float(m.group(1))
        return int(round(kg * 1000)) if kg is not None else None
    if m.group(3):  # gramos
        g = _to_float(m.group(3))
        return int(round(g)) if g is not None else None
    return None

def extract_volume_ml(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = VOL_RE.search(text)
    if not m: return None
    if m.group(1):  # litros
        l = _to_float(m.group(1));  return int(round(l * 1000)) if l is not None else None
    if m.group(3):  # centilitros
        cl = _to_float(m.group(3)); return int(round(cl * 10)) if cl is not None else None
    if m.group(5):  # mililitros
        ml = _to_float(m.group(5)); return int(round(ml)) if ml is not None else None
    return None

def extract_count_units(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = COUNT_RE.search(text)
    if not m: return None
    for g in (m.group(1), m.group(2)):
        if g:
            try:
                n = int(g)
                return n if 1 <= n <= 5000 else None
            except:
                continue
    return None

def normalize_ub_unit(unit: str) -> str:
    if not isinstance(unit, str): return ""
    u = unit.strip().lower()
    if u in {"g","gr","gramo","gramos","g."}: return "g"
    if u in {"kg","kilo","kilogramo","kilogramos","kg."}: return "kg"
    if u in {"ml","mililitro","mililitros"}: return "ml"
    if u in {"l","lt","litro","litros","l."}: return "l"
    return "u"

# ===================== Preproceso =====================
def preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df_in.copy())

    required = {"empresa", "fecha_dcto", "id_co", "id_item", "und_dia"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas (después de normalizar): {missing}")

    # Fechas
    parsed = parse_dates_strict(df["fecha_dcto"])
    if parsed.notna().any():
        df["dia_mes"] = parsed.dt.day.astype("Int64")
        df["mes_num"] = parsed.dt.month.astype("Int64")
    else:
        day = extract_day_if_possible(df["fecha_dcto"])
        if day.isna().all():
            raise ValueError("No pude interpretar 'fecha_dcto'. Usa fecha completa (ej. 24/09/2025) o día 1..31.")
        df["dia_mes"] = day
        df["mes_num"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    # Normalizaciones
    df["empresa"] = df["empresa"].map(unify_empresa)
    idco_num = pd.to_numeric(df["id_co"], errors="coerce")
    df["id_co"] = (idco_num.round().astype("Int64").astype(str)
                   if idco_num.notna().any() else df["id_co"].astype(str))
    df["sede_key"] = df["empresa"].astype(str).str.lower().str.strip() + "|" + df["id_co"].astype(str).str.strip()
    df["id_item"] = df["id_item"].astype(str).str.strip()
    if "descripcion" in df.columns:
        df["descripcion"] = df["descripcion"].astype(str)

    # UR
    df["und_dia"] = parse_und_dia_series(df["und_dia"])

    # ===== UB por fila: prioridad peso > volumen > conteo =====
    # Explícitos (si existen)
    if "ub_factor" in df.columns:
        ef = pd.to_numeric(df["ub_factor"], errors="coerce").fillna(0)
        ef = ef.where(ef > 0, 0).astype(float)
    else:
        ef = pd.Series(0.0, index=df.index)

    if "ub_unit" in df.columns:
        eu = df["ub_unit"].map(normalize_ub_unit)
    else:
        eu = pd.Series([""]*len(df), index=df.index)

    # Desde descripción
    grams_desc = df["descripcion"].map(extract_weight_grams) if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    ml_desc    = df["descripcion"].map(extract_volume_ml)   if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    cnt_desc   = df["descripcion"].map(extract_count_units) if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)

    ub_factor_val = []
    ub_unit_type  = []
    for i in df.index:
        g = grams_desc[i]
        v = ml_desc[i]
        c = cnt_desc[i]
        ef_i = ef[i]
        eu_i = eu[i]

        # prioridad: peso > volumen > unitario
        if g is not None:
            ub_factor_val.append(float(g)); ub_unit_type.append("g")
        elif v is not None:
            ub_factor_val.append(float(v)); ub_unit_type.append("ml")
        elif eu_i in {"kg","g"} and ef_i > 0:
            # convertir kg a g
            grams = ef_i*1000.0 if eu_i == "kg" else ef_i
            ub_factor_val.append(float(grams)); ub_unit_type.append("g")
        elif eu_i in {"l","ml"} and ef_i > 0:
            # convertir litros a ml
            ml = ef_i*1000.0 if eu_i == "l" else ef_i
            ub_factor_val.append(float(ml)); ub_unit_type.append("ml")
        elif ef_i > 0:
            ub_factor_val.append(float(ef_i)); ub_unit_type.append("u")
        elif c is not None and c > 0:
            ub_factor_val.append(float(c));   ub_unit_type.append("u")
        else:
            ub_factor_val.append(1.0);        ub_unit_type.append("u")

    df["ub_factor_val"] = pd.Series(ub_factor_val, index=df.index)
    df["ub_unit_type"]  = pd.Series(ub_unit_type,  index=df.index)
    df["ub_unidades"]   = (df["und_dia"].astype("Float64") * pd.Series(ub_factor_val, index=df.index).astype("Float64")).round().astype("Int64")

    return df

# ===================== Tabla (UR / UB) =====================
def build_table_multi(df: pd.DataFrame, id_items_sel: list[str], metric: str = "UR") -> pd.DataFrame:
    if not id_items_sel: return pd.DataFrame()

    id_items_sel = [str(x).strip() for x in id_items_sel]
    dff = df[df["id_item"].isin(id_items_sel)].copy()
    if dff.empty: return pd.DataFrame()

    value_col = "und_dia" if metric == "UR" else "ub_unidades"

    pivot = dff.pivot_table(
        index="dia_mes", columns="sede_key", values=value_col,
        aggfunc="sum", fill_value=0, dropna=False,
    )

    # Días presentes
    all_days = sorted(dff["dia_mes"].dropna().unique())
    pivot = pivot.reindex(all_days, fill_value=0)

    # Columnas con nombres de sede (agrupar repetidas)
    new_cols = {col: sede_key_to_name(str(col)) for col in pivot.columns}
    pivot = pivot.rename(columns=new_cols)
    pivot = pivot.T.groupby(level=0).sum().T

    # Fecha visible
    pivot = pivot.reset_index().rename(columns={"dia_mes": "Fecha"})
    fecha_fmt = pivot["Fecha"].astype("Int64").astype(str)
    if "mes_num" in dff.columns and dff["mes_num"].notna().any():
        mes_por_dia = (dff.dropna(subset=["mes_num"]).groupby("dia_mes")["mes_num"]
                       .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
        fecha_fmt = pivot["Fecha"].astype(int).map(lambda d: f"{d}/{MONTH_ABBR_ES.get(int(mes_por_dia.get(d, pd.NA)), '')}".rstrip("/"))
    pivot["Fecha"] = fecha_fmt

    # Orden de sedes
    ordered_cols, extras = [], []
    for emp, mapping in SEDE_NAME_MAP.items():
        for _, nombre in mapping.items():
            if nombre in pivot.columns: ordered_cols.append(nombre)
    for c in pivot.columns:
        if c not in ordered_cols + ["Fecha"]: extras.append(c)
    sede_cols = ordered_cols + extras

    # Totales
    pivot = pivot[["Fecha"] + sede_cols].copy()
    # aseguramos ints con nullable
    for c in sede_cols:
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce").fillna(0).round().astype("Int64")
    pivot["T. Dia"] = pivot[sede_cols].sum(axis=1).astype("Int64")

    # Fila acumulado
    acum_values = [int(pivot[c].sum()) for c in sede_cols]
    acum_total  = int(pivot["T. Dia"].sum())
    acum_row = pd.DataFrame([["Acum. Mes:"] + acum_values + [acum_total]],
                            columns=["Fecha"] + sede_cols + ["T. Dia"])
    # convertir la fila acumulado a Int64
    for c in sede_cols + ["T. Dia"]:
        acum_row[c] = pd.to_numeric(acum_row[c], errors="coerce").astype("Int64")

    final = pd.concat([pivot, acum_row], ignore_index=True)
    return final.reset_index(drop=True)

# ===================== UI =====================
st.set_page_config(page_title="Ventas x Item – UR / UB (multi-item)", layout="wide")
st.title("📊 Ventas por día y acumulados por sede (UR / UB)")

with st.sidebar:
    st.header("Opciones")
    uploaded = st.file_uploader("Archivo (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    show_dash = st.checkbox("Mostrar '-' cuando sea 0 (solo visual)", value=True)
    debug = st.checkbox("Mostrar diagnóstico", value=False)

if not uploaded:
    st.info("⬅️ Sube un archivo con columnas: empresa, fecha_dcto, id_co, id_item, und_dia (opcional: descripcion, ub_factor, ub_unit)")
    st.stop()

# Leer archivo
name = uploaded.name.lower()
try:
    if name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"No pude leer el archivo: {e}")
    st.stop()

# Preprocesar
try:
    df = preprocess(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# Selector de ítems (1..10)
if "descripcion" in df.columns:
    sum_por_item = df.groupby(["id_item","descripcion"], as_index=True)["und_dia"].sum().sort_values(ascending=False)
    opts = [(i, d) for (i, d) in sum_por_item.index]
    def fmt(opt):
        i, desc = opt
        return f"{i} - {desc}  (UR: {int(sum_por_item.loc[(i, desc)]):,})".replace(",", ".")
    selected = st.multiselect("Items (selecciona 1 a 10)", opts, max_selections=10, format_func=fmt)
    id_items_sel = [i for (i, _) in selected]
else:
    sum_por_item = df.groupby("id_item", as_index=True)["und_dia"].sum().sort_values(ascending=False)
    opts = list(sum_por_item.index)
    def fmt(i):
        return f"{i}  (UR: {int(sum_por_item.loc[i]):,})".replace(",", ".")
    id_items_sel = st.multiselect("Items (selecciona 1 a 10)", opts, max_selections=10, format_func=fmt)

if not id_items_sel:
    st.warning("Selecciona al menos un item (máximo 10).")
    st.stop()

# Toggle UR / UB
if "metric" not in st.session_state:
    st.session_state.metric = "UR"

c1, c2 = st.columns([1,1])
with c1:
    if st.button("🔁 Cambiar vista (UR / UB)"):
        st.session_state.metric = "UB" if st.session_state.metric == "UR" else "UR"
with c2:
    st.write("Vista actual:", f"**{st.session_state.metric}**")

# Construcción de tabla (NUMÉRICA, sin guiones)
tabla_numeric = build_table_multi(df, id_items_sel, metric=st.session_state.metric)
if tabla_numeric.empty:
    st.error("No hay datos para los ítems seleccionados con la vista actual.")
    st.stop()

# ==== Render seguro para PyArrow: guiones SOLO en presentación ====
def render_table(df_num: pd.DataFrame, show_dash: bool):
    if not show_dash:
        st.dataframe(df_num, width="stretch")
        return
    numeric_cols = [c for c in df_num.columns if c != "Fecha"]
    def fmt_val(x):
        if pd.isna(x): return "-"
        try:
            xi = int(x)
            return "-" if xi == 0 else f"{xi:,}".replace(",", ".")
        except:
            return x
    styler = df_num.style.format({c: fmt_val for c in numeric_cols})
    st.dataframe(styler, width="stretch")

st.subheader(f"Tabla ({st.session_state.metric}) – items: {', '.join(map(str, id_items_sel))}")
render_table(tabla_numeric, show_dash)

# Diagnóstico opcional
if debug:
    with st.expander("🔎 Diagnóstico"):
        dff_dbg = df[df["id_item"].isin(id_items_sel)].copy()
        st.write("Días únicos:", sorted(dff_dbg["dia_mes"].dropna().unique()))
        if "mes_num" in dff_dbg.columns:
            st.write("Meses únicos (num):", sorted(pd.Series(dff_dbg["mes_num"]).dropna().unique()))
        st.write("Sedes únicas:", sorted(dff_dbg["sede_key"].unique()))
        st.write("Ejemplos UB (primeras 20 filas):")
        st.dataframe(dff_dbg[["id_item","descripcion","und_dia","ub_factor_val","ub_unit_type","ub_unidades"]].head(20), width="stretch")

# -------- Exportar a Excel (siempre datos numéricos, sin guiones) --------
def to_excel_bytes(df_out: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name="reporte", index=False)
        wb = writer.book
        ws = writer.sheets["reporte"]
        fmt_int = wb.add_format({"num_format": "#,##0"})
        for col_idx, col in enumerate(df_out.columns):
            ws.set_column(col_idx, col_idx, 12 if col != "Fecha" else 10, None if col=="Fecha" else fmt_int)
    return buf.getvalue()

excel_bytes = to_excel_bytes(tabla_numeric)
st.download_button(
    label=f"⬇️ Descargar Excel ({st.session_state.metric}).xlsx",
    data=excel_bytes,
    file_name=f"tabla_ventas_items_{st.session_state.metric.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
