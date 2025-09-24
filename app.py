"""
App: Ventas x Item – UR/UB robusto por día x sede (multi-item hasta 10)

Novedades:
- Título resumido e “inteligente” sin listar 10 ítems: agrupa por nombre base (2 tokens).
- Si todos pertenecen a un único grupo: muestra “(N variantes)”.
- Si hay varios grupos: muestra top 2 representativos por UR y “y N más”.
- Selección sigue siendo SOLO por ítems (sin filtros nuevos).
- Pre-proceso y pre-agregado cacheados con @st.cache_data para rendimiento.
- Render simultáneo en tabs (UR / UB).
"""

import io
import re
import string
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Config: nombres de sedes (y orden) =====================
SEDE_NAME_MAP = {
    "mercamio": {"1": "Calle 5ta","2": "La 39","3": "Plaza","4": "Jardín","5": "C. Sur","6": "Palmira"},
    "mtodo": {"1": "Floresta","2": "Floralia","3": "Guadua"},
    "bogota": {"1": "Calle 80","2": "Chía"},
}
MONTH_ABBR_ES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
MONTH_FULL_ES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}

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

# ===================== Normalización de nombres SOLO para el TÍTULO =====================
FILLER = {"de","la","el","los","las","con","sin","para","x","por","un","una","y","o","en","a"}
UNITS_RE = re.compile(r"""
    (\b\d+(?:[.,]\d+)?\s?(kg|kilo|kilogramos?|g|gr|gramos?|l|lt|litros?|ml|cl)\b)
  | (\bx\s*\d+\b)
  | (\b\d+\s*(u|un|und|uds|unid|unids|unidades)\b)
  | (\bpack\s*\d+\b)
  | (\b\d{1,2}%\b)
  | (\b\d+\b)
""", re.IGNORECASE | re.VERBOSE)

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _clean_for_base(s: str) -> str:
    s = _strip_accents(str(s)).lower().strip()
    s = UNITS_RE.sub(" ", s)                              # quita tamaños, packs, números
    s = s.translate(str.maketrans({p: " " for p in string.punctuation}))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_base_name(desc: str, max_tokens: int = 2) -> str:
    """
    Devuelve un nombre 'base' corto (agresivo: 2 tokens) para agrupar variantes similares.
    Ej: "Pan Seletti 500g x6" -> "Pan Seletti"
    """
    if not isinstance(desc, str) or not desc.strip():
        return ""
    txt = _clean_for_base(desc)
    tokens = [t for t in txt.split() if t not in FILLER]
    base = " ".join(tokens[:max_tokens]).strip()
    return base.title()

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
    s = series.astype(str).str.strip()
    formats = ("%d/%m/%Y","%Y-%m-%d","%d-%m-%Y","%d/%m/%y","%Y/%m/%d")
    for fmt in formats:
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        if dt.notna().mean() >= 0.8:
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
    return out.round().astype("Int64")

# ===================== Parsers UB =====================
NUM = r"(\d+(?:[.,]\d+)?)"
WEIGHT_RE = re.compile(rf"{NUM}\s*(kg|kilo|kilogramo|kilogramos|kg\.)|{NUM}\s*(g|gr|gramo|gramos|g\.)", re.IGNORECASE)
VOL_RE    = re.compile(rf"{NUM}\s*(l|lt|litro|litros|l\.)|{NUM}\s*(cl|centilitro|centilitros)|{NUM}\s*(ml|mililitro|mililitros)", re.IGNORECASE)
COUNT_RE  = re.compile(r"(?:x\s*(\d{1,5}))|(?:\b(\d{1,5})\s*(?:u|un|und|uds|unid|unids|unidades)\b)", re.IGNORECASE)

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

def extract_weight_grams(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = WEIGHT_RE.search(text)
    if not m: return None
    if m.group(1):
        kg = _to_float(m.group(1)); return int(round(kg * 1000)) if kg is not None else None
    if m.group(3):
        g = _to_float(m.group(3));  return int(round(g)) if g is not None else None
    return None

def extract_volume_ml(text: str):
    if not isinstance(text, str) or not text.strip(): return None
    m = VOL_RE.search(text)
    if not m: return None
    if m.group(1):
        l = _to_float(m.group(1));  return int(round(l * 1000)) if l is not None else None
    if m.group(3):
        cl = _to_float(m.group(3)); return int(round(cl * 10)) if cl is not None else None
    if m.group(5):
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

# ===================== Preproceso + agregado cacheados =====================
@st.cache_data(show_spinner=False)
def preprocess_cached(file_bytes: bytes, filename: str):
    # leer
    if filename.lower().endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))

    # normalizar columnas
    df = standardize_columns(df_raw)
    required = {"empresa", "fecha_dcto", "id_co", "id_item", "und_dia"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas (después de normalizar): {missing}")

    # fechas
    parsed = parse_dates_strict(df["fecha_dcto"])
    if parsed.notna().any():
        df["dia_mes"] = parsed.dt.day.astype("Int64")
        df["mes_num"] = parsed.dt.month.astype("Int64")
        df["anio"]    = parsed.dt.year.astype("Int64")
    else:
        day = extract_day_if_possible(df["fecha_dcto"])
        if day.isna().all():
            raise ValueError("No pude interpretar 'fecha_dcto'. Usa fecha completa (ej. 24/09/2025) o día 1..31.")
        df["dia_mes"] = day
        df["mes_num"] = pd.Series([pd.NA]*len(df), dtype="Int64")
        df["anio"]    = pd.Series([pd.NA]*len(df), dtype="Int64")

    # normalizaciones clave
    df["empresa"] = df["empresa"].map(unify_empresa)
    idco_num = pd.to_numeric(df["id_co"], errors="coerce")
    df["id_co"] = (idco_num.round().astype("Int64").astype(str)
                   if idco_num.notna().any() else df["id_co"].astype(str))
    df["sede_key"] = df["empresa"].astype(str).str.lower().str.strip() + "|" + df["id_co"].astype(str).str.strip()
    df["id_item"] = df["id_item"].astype(str).str.strip()
    if "descripcion" in df.columns:
        df["descripcion"] = df["descripcion"].astype(str)
        # calcular desc_base (2 tokens) para títulos
        df["desc_base"]   = df["descripcion"].map(make_base_name)
    else:
        df["desc_base"]   = ""

    # UR
    df["und_dia"] = parse_und_dia_series(df["und_dia"])

    # UB por fila (prioridad peso > volumen > unitario)
    if "ub_factor" in df.columns:
        ef = pd.to_numeric(df["ub_factor"], errors="coerce").fillna(0)
        ef = ef.where(ef > 0, 0).astype(float)
    else:
        ef = pd.Series(0.0, index=df.index)

    if "ub_unit" in df.columns:
        eu = df["ub_unit"].map(normalize_ub_unit)
    else:
        eu = pd.Series([""]*len(df), index=df.index)

    grams_desc = df["descripcion"].map(extract_weight_grams) if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    ml_desc    = df["descripcion"].map(extract_volume_ml)   if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)
    cnt_desc   = df["descripcion"].map(extract_count_units) if "descripcion" in df.columns else pd.Series([None]*len(df), index=df.index)

    ub_factor_val = np.empty(len(df), dtype=float)
    ub_unit_type  = np.empty(len(df), dtype=object)

    for i in range(len(df)):
        g = grams_desc.iloc[i]
        v = ml_desc.iloc[i]
        c = cnt_desc.iloc[i]
        ef_i = ef.iloc[i]
        eu_i = eu.iloc[i]
        if g is not None:
            ub_factor_val[i] = float(g);   ub_unit_type[i] = "g"
        elif v is not None:
            ub_factor_val[i] = float(v);   ub_unit_type[i] = "ml"
        elif eu_i in {"kg","g"} and ef_i > 0:
            grams = ef_i*1000.0 if eu_i == "kg" else ef_i
            ub_factor_val[i] = float(grams); ub_unit_type[i] = "g"
        elif eu_i in {"l","ml"} and ef_i > 0:
            ml = ef_i*1000.0 if eu_i == "l" else ef_i
            ub_factor_val[i] = float(ml);   ub_unit_type[i] = "ml"
        elif ef_i > 0:
            ub_factor_val[i] = float(ef_i); ub_unit_type[i] = "u"
        elif c is not None and c > 0:
            ub_factor_val[i] = float(c);    ub_unit_type[i] = "u"
        else:
            ub_factor_val[i] = 1.0;         ub_unit_type[i] = "u"

    df["ub_factor_val"] = ub_factor_val
    df["ub_unit_type"]  = ub_unit_type
    df["ub_unidades"]   = (df["und_dia"].astype("Float64") * pd.Series(ub_factor_val, index=df.index).astype("Float64")).round().astype("Int64")

    # ------- PRE-AGREGADO (UR y UB) cacheable -------
    agg = (df.groupby(["dia_mes","mes_num","anio","sede_key","id_item"], as_index=False)
             .agg(UR=("und_dia","sum"), UB=("ub_unidades","sum")))
    agg["sede_name"] = agg["sede_key"].map(lambda k: sede_key_to_name(str(k)))

    return df, agg

# ============== Helpers tabla ==============
def _order_sede_columns(cols):
    ordered, extras = [], []
    for emp, mapping in SEDE_NAME_MAP.items():
        for _, nombre in mapping.items():
            if nombre in cols: ordered.append(nombre)
    for c in cols:
        if c not in ordered: extras.append(c)
    return ordered + extras

def _fecha_label_from_group(dias: pd.Series, mes_map: dict) -> pd.Series:
    out = dias.astype("Int64").astype(str)
    if mes_map:
        as_int = dias.astype("Int64").fillna(0).astype(int)
        return as_int.map(lambda d: f"{d}/{MONTH_ABBR_ES.get(int(mes_map.get(d, 0)), '')}".rstrip("/"))
    return out

def build_table_from_agg(agg: pd.DataFrame, id_items_sel: list[str], metric: str) -> pd.DataFrame:
    if not id_items_sel:
        return pd.DataFrame()
    sids = [str(x).strip() for x in id_items_sel]
    dff = agg[agg["id_item"].isin(sids)]
    if dff.empty:
        return pd.DataFrame()

    # detectar mes por día (modo)
    m = (dff.dropna(subset=["mes_num"])
            .groupby("dia_mes")["mes_num"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
    # pivot liviano
    pv = dff.pivot_table(index="dia_mes", columns="sede_name", values=metric, aggfunc="sum", fill_value=0)
    all_days = sorted(dff["dia_mes"].dropna().unique())
    pv = pv.reindex(all_days, fill_value=0)

    # ordenar columnas de sede
    sede_cols = _order_sede_columns([c for c in pv.columns])
    pv = pv[sede_cols]

    # Fecha visible
    pv = pv.reset_index().rename(columns={"dia_mes":"Fecha"})
    pv["Fecha"] = _fecha_label_from_group(pv["Fecha"], m.to_dict())

    # Totales y fila de acumulado
    for c in sede_cols:
        pv[c] = pd.to_numeric(pv[c], errors="coerce").fillna(0).round().astype("Int64")
    pv["T. Dia"] = pv[sede_cols].sum(axis=1).astype("Int64")

    acum_vals = [int(pv[c].sum()) for c in sede_cols]
    acum_total = int(pv["T. Dia"].sum())
    acum_row = pd.DataFrame([["Acum. Mes:"] + acum_vals + [acum_total]], columns=["Fecha"] + sede_cols + ["T. Dia"])
    for c in sede_cols + ["T. Dia"]:
        acum_row[c] = pd.to_numeric(acum_row[c], errors="coerce").astype("Int64")

    final = pd.concat([pv, acum_row], ignore_index=True)
    return final

# ============== Título resumido por “nombre base” (2 tokens) ==============
def build_title_resumido(df: pd.DataFrame, id_items_sel: list[str], top_groups: int = 2) -> str:
    """
    - ÚNICO grupo base: “Mes Año – Vta por día y acumulada de “Base (N variantes)”” si N>1
    - VARIOS grupos:    “Mes Año – Vta por día y acumulada de k ítems (Base1, Base2 y X más)”
      Bases ordenadas por mayor UR total dentro de la selección.
    """
    if not id_items_sel:
        return "Vta por día y acumulada"

    ss = df[df["id_item"].astype(str).isin(map(str, id_items_sel))].copy()

    # Mes/Año por moda
    mes = int(ss["mes_num"].mode().iloc[0]) if "mes_num" in ss and ss["mes_num"].notna().any() else None
    anio = int(ss["anio"].mode().iloc[0]) if "anio" in ss and ss["anio"].notna().any() else None
    mes_txt  = MONTH_FULL_ES.get(mes, "") if mes else ""
    anio_txt = str(anio) if anio else ""

    # Si no hay descripciones, caer a conteo
    if "descripcion" not in df.columns or ss["descripcion"].isna().all():
        n = len(id_items_sel)
        return f"{mes_txt} {anio_txt} – Vta por día y acumulada de {n} ítems".strip()

    # Asegurar base: si no existe en DF (debería), calcular al vuelo
    if "desc_base" not in ss.columns or ss["desc_base"].isna().all():
        ss["desc_base"] = ss["descripcion"].map(lambda x: make_base_name(x, max_tokens=2))

    # Variantes por base (n° de id_item distintos) y UR total para ordenar
    variantes_por_base = (ss.groupby("desc_base")["id_item"].nunique().rename("variantes"))
    ur_por_base = (ss.groupby("desc_base")["und_dia"].sum().rename("UR_total")).sort_values(ascending=False)
    resumen = (pd.concat([variantes_por_base, ur_por_base], axis=1)
                 .sort_values("UR_total", ascending=False)
                 .reset_index())
    resumen = resumen[resumen["desc_base"].astype(str).str.strip() != ""]
    if resumen.empty:
        n = len(id_items_sel)
        return f"{mes_txt} {anio_txt} – Vta por día y acumulada de {n} ítems".strip()

    n_items = len(id_items_sel)
    n_bases = len(resumen)

    if n_bases == 1:
        base = resumen.loc[0, "desc_base"]
        n_var = int(resumen.loc[0, "variantes"])
        suf = "" if n_var <= 1 else f" ({n_var} variantes)"
        return f'{mes_txt} {anio_txt} – Vta por día y acumulada de “{base}{suf}”'.strip()

    # Varios grupos: top N bases representativas
    top = resumen.head(top_groups)["desc_base"].tolist()
    if len(top) == 1:
        listado = top[0]
    elif len(top) == 2:
        listado = f"{top[0]}, {top[1]}"
    else:
        listado = f'{", ".join(top[:-1])} y {top[-1]}'

    restantes = max(0, n_bases - len(top))
    sufijo = f" y {restantes} más" if restantes > 0 else ""
    return f"{mes_txt} {anio_txt} – Vta por día y acumulada de {n_items} ítems ({listado}{sufijo})".strip()

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

# Lectura y pre-agregado (cacheado por contenido de archivo)
file_bytes = uploaded.getvalue()
try:
    df, agg = preprocess_cached(file_bytes, uploaded.name)
except Exception as e:
    st.error(str(e))
    st.stop()

# ===== Selector de ítems (1..10) =====
if "descripcion" in df.columns:
    sum_por_item = (df.groupby(["id_item","descripcion"], as_index=True)["und_dia"]
                      .sum().sort_values(ascending=False))
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

# ===== Construir tablas desde el agregado (rápido) =====
tabla_UR = build_table_from_agg(agg, id_items_sel, "UR")
tabla_UB = build_table_from_agg(agg, id_items_sel, "UB")

if tabla_UR.empty and tabla_UB.empty:
    st.error("No hay datos para los ítems seleccionados.")
    st.stop()

# ===== Formateo vectorizado (sin Styler) =====
def format_df_fast(df_in: pd.DataFrame, dash_zero: bool) -> pd.DataFrame:
    dfv = df_in.copy()
    num_cols = [c for c in dfv.columns if c != "Fecha"]
    if dash_zero:
        for c in num_cols:
            s = pd.to_numeric(dfv[c], errors="coerce")
            dfv[c] = np.where(s.fillna(0).astype(int) == 0, "-", s.map(lambda x: f"{int(x):,}".replace(",", ".")))
    else:
        for c in num_cols:
            s = pd.to_numeric(dfv[c], errors="coerce")
            dfv[c] = s.map(lambda x: f"{int(x):,}".replace(",", "."))
    return dfv

df_UR_disp = format_df_fast(tabla_UR, show_dash) if not tabla_UR.empty else pd.DataFrame()
df_UB_disp = format_df_fast(tabla_UB, show_dash) if not tabla_UB.empty else pd.DataFrame()

# ===== Título resumido =====
titulo_tabla = build_title_resumido(df, id_items_sel, top_groups=2)

# ===== Render simultáneo en tabs =====
tab1, tab2 = st.tabs(["🔹 UR", "🔸 UB"])

with tab1:
    if not df_UR_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(df_UR_disp, width="stretch")
    else:
        st.info("Sin datos UR para la selección actual.")

with tab2:
    if not df_UB_disp.empty:
        st.subheader(titulo_tabla)
        st.dataframe(df_UB_disp, width="stretch")
    else:
        st.info("Sin datos UB para la selección actual.")

# ===== Diagnóstico opcional =====
if debug:
    with st.expander("🔎 Diagnóstico"):
        st.write("Agregado (primeras 20 filas):")
        st.dataframe(agg.head(20), width="stretch")
        st.write("Sedes únicas (agregado):", sorted(agg["sede_key"].unique()))
        st.write("Días únicos:", sorted(agg["dia_mes"].dropna().unique()))
        if "descripcion" in df.columns:
            st.write("Ejemplos desc_base (2 tokens):", df[["descripcion","desc_base"]].head(20))

# -------- Exportar a Excel (elige UR o UB) --------
@st.cache_data(show_spinner=False)
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

choice = st.radio("Descargar:", ["UR","UB"], horizontal=True)
df_to_save = tabla_UR if choice=="UR" else tabla_UB
excel_bytes = to_excel_bytes(df_to_save)
st.download_button(
    label=f"⬇️ Descargar Excel ({choice}).xlsx",
    data=excel_bytes,
    file_name=f"tabla_ventas_items_{choice.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
