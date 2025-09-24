# utils.py
import re, string, unicodedata
import numpy as np
import pandas as pd
from constants import ALIASES, SEDE_NAME_MAP, MONTH_ABBR_ES

# ---------------- Columnas y sedes ----------------
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
    if s in ("bogota","bogotá","bogota dc","bogotá dc","bogota d.c.","bogotá d.c."): return "bogota"
    return s

def sede_key_to_name(sede_key: str) -> str:
    emp, co = sede_key.split("|", 1)
    emp = emp.strip().lower()
    co = co.strip()
    if emp in SEDE_NAME_MAP and co in SEDE_NAME_MAP[emp]:
        return SEDE_NAME_MAP[emp][co]
    return f"{emp}-{co}"

# ---------------- Fechas ----------------
def parse_dates_strict(series: pd.Series) -> pd.Series:
    """
    Intenta parsear fechas con formatos comunes.
    Usa el primero que logre interpretar al menos 80% de los valores.
    Evita warnings de pandas al no depender de la inferencia automática.
    """
    s = series.astype(str).str.strip()
    formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            dt = pd.to_datetime(s, format=fmt, errors="coerce")
            if dt.notna().mean() >= 0.8:
                return dt
        except Exception:
            continue
    # fallback si hay mezcla rara de formatos
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def extract_day_if_possible(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    day_str = raw.str.extract(r"^\s*(\d{1,2})")[0]
    day = pd.to_numeric(day_str, errors="coerce")
    return day.where((day >= 1) & (day <= 31)).astype("Int64")

# ---------------- Números ----------------
def parse_und_dia_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    both = s.str.contains(r"\.") & s.str.contains(r",")
    s = s.where(~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    only_comma = s.str.contains(",") & ~s.str.contains(r"\.")
    s = s.where(~only_comma, s.str.replace(",", "", regex=False))
    out = pd.to_numeric(s, errors="coerce").fillna(0)
    return out.round().astype("Int64")

# ---------------- Normalización de nombres (para títulos) ----------------
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
    s = UNITS_RE.sub(" ", s)
    s = s.translate(str.maketrans({p: " " for p in string.punctuation}))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_base_name(desc: str, max_tokens: int = 2) -> str:
    if not isinstance(desc, str) or not desc.strip():
        return ""
    txt = _clean_for_base(desc)
    tokens = [t for t in txt.split() if t not in FILLER]
    base = " ".join(tokens[:max_tokens]).strip()
    return base.title()

# ---------------- Tablas: helpers ----------------
def _order_sede_columns(cols):
    expected = []
    for _, mapping in SEDE_NAME_MAP.items():
        for _, nombre in mapping.items():
            expected.append(nombre)
    extras = [c for c in cols if c not in expected]
    return expected + extras

def _fecha_label_from_group(dias: pd.Series, mes_map: dict) -> pd.Series:
    out = dias.astype("Int64").astype(str)
    if mes_map:
        as_int = dias.astype("Int64").fillna(0).astype(int)
        return as_int.map(lambda d: f"{d}/{MONTH_ABBR_ES.get(int(mes_map.get(d, 0)), '')}".rstrip("/"))
    return out

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
