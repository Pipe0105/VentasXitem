# titles.py
import pandas as pd
from constants import MONTH_FULL_ES
from utils import make_base_name

def build_title_resumido(df: pd.DataFrame, id_items_sel: list[str], top_groups: int = 2) -> str:
    """
    Genera un título compacto e “inteligente”:
    - Si todas las selecciones caen en UN solo grupo base → “Mes Año – ... de “Base (N variantes)”” (si N>1).
    - Si hay VARIOS grupos → “Mes Año – ... de k ítems (Base1, Base2 y X más)”.
    Los grupos base se ordenan por UR total dentro de la selección.
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

    # Asegurar desc_base (2 tokens). Si el DF no la trae, la calculamos al vuelo.
    if "desc_base" not in ss.columns or ss["desc_base"].isna().all():
        ss["desc_base"] = ss["descripcion"].map(lambda x: make_base_name(x, max_tokens=2))

    # Variantes por base y UR total para ordenar representatividad
    variantes_por_base = ss.groupby("desc_base")["id_item"].nunique().rename("variantes")
    ur_por_base = ss.groupby("desc_base")["und_dia"].sum().rename("UR_total").sort_values(ascending=False)
    resumen = (pd.concat([variantes_por_base, ur_por_base], axis=1)
                 .sort_values("UR_total", ascending=False)
                 .reset_index())
    resumen = resumen[resumen["desc_base"].astype(str).str.strip() != ""]
    if resumen.empty:
        n = len(id_items_sel)
        return f"{mes_txt} {anio_txt} – Vta por día y acumulada de {n} ítems".strip()

    n_items = len(id_items_sel)
    n_bases = len(resumen)

    # Único grupo
    if n_bases == 1:
        base = resumen.loc[0, "desc_base"]
        n_var = int(resumen.loc[0, "variantes"])
        suf = "" if n_var <= 1 else f" ({n_var} variantes)"
        return f'{mes_txt} {anio_txt} – Vta por día y acumulada de “{base}{suf}”'.strip()

    # Varios grupos: mostrar top N bases representativas
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
