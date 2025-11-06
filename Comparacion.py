import os
import re
import math
import pandas as pd
from difflib import get_close_matches
import streamlit as st

def leer_datos(path="precios.txt"):
    base = os.path.dirname(__file__)
    path = os.path.join(base, path)
    if not os.path.exists(path):
        st.error(f"Archivo '{path}' no encontrado.")
        return pd.DataFrame(columns=["supermercado", "producto", "precio", "unidad"])
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]
    df = None
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            try:
                df = pd.read_csv(path, encoding=enc, engine="python")
                break
            except Exception as e2:
                last_err = e2
                continue
    if df is None:
        st.error(f"No se pudo leer '{path}'. Último error: {last_err}")
        return pd.DataFrame(columns=["supermercado", "producto", "precio", "unidad"])
    expected = {"supermercado", "producto", "precio"}
    if not expected.issubset(set(df.columns)):
        st.error(f"El archivo debe contener al menos las columnas: {', '.join(expected)}")
        return pd.DataFrame(columns=["supermercado", "producto", "precio", "unidad"])
    df["producto"] = df["producto"].astype(str)
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    if "unidad" not in df.columns:
        df["unidad"] = ""
    df = df.dropna(subset=["precio"]).reset_index(drop=True)
    return df

def _normalize_text_for_clustering(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(\d+(\.\d+)?\s?(kg|g|lb|lbs|l|ml|oz|ozs|un|unidades|unidad|gr|grs))\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _convert_to_base(value: float, typ: str):
    typ = typ.lower()
    if typ in ("kg",):
        return (value, "mass")
    if typ in ("g","gr","grs"):
        return (value * 0.001, "mass")
    if typ in ("lb","lbs"):
        return (value * 0.45359237, "mass")
    if typ in ("oz","ozs"):
        return (value * 0.0283495, "mass")
    if typ in ("l",):
        return (value, "volume")
    if typ in ("ml",):
        return (value * 0.001, "volume")
    if typ in ("un", "unidad", "unidades"):
        return (value, "count")
    return (None, None)

def _parse_qty_and_type(unidad: str, producto: str):
    txt = (unidad or "").lower().strip()
    patterns = [
        (r"^(\d+(?:\.\d+)?)\s*kg$", "kg"),
        (r"^(\d+(?:\.\d+)?)\s*g$", "g"),
        (r"^(\d+(?:\.\d+)?)\s*gr", "g"),
        (r"^(\d+(?:\.\d+)?)\s*lb", "lb"),
        (r"^(\d+(?:\.\d+)?)\s*lbs", "lb"),
        (r"^(\d+(?:\.\d+)?)\s*l$", "l"),
        (r"^(\d+(?:\.\d+)?)\s*ml$", "ml"),
        (r"^(\d+(?:\.\d+)?)\s*oz$", "oz"),
        (r"^(\d+)\s*un", "un"),
        (r"^(\d+)\s*unidad", "un"),
    ]
    for pat, typ in patterns:
        m = re.search(pat, txt)
        if m:
            val = float(m.group(1))
            return _convert_to_base(val, typ)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|lb|lbs|l|ml|oz|ozs|un|unidad|unidades|gr|grs)", producto.lower())
    if m:
        val = float(m.group(1))
        typ = m.group(2)
        return _convert_to_base(val, typ)
    return (None, None)

def enriquecer_ofertas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["producto_norm"] = df["producto"].astype(str).apply(_normalize_text_for_clustering)
    qtys = df.apply(lambda r: _parse_qty_and_type(r.get("unidad", ""), r.get("producto", "")), axis=1)
    df["qty_base"] = [q[0] for q in qtys]
    df["unit_type"] = [q[1] for q in qtys]
    df["unit_type"] = df["unit_type"].fillna("each")
    # si no qty_base y unit_type == each, asumimos qty_base=1
    df["qty_base"] = df.apply(lambda r: r["qty_base"] if pd.notna(r["qty_base"]) else (1.0 if r["unit_type"] == "each" else None), axis=1)
    def calc_up(row):
        try:
            if pd.notna(row["qty_base"]) and row["qty_base"] > 0:
                return row["precio"] / row["qty_base"]
        except Exception:
            pass
        return None
    df["unit_price"] = df.apply(calc_up, axis=1)
    return df

def comparar_precios(datos: pd.DataFrame, lista_productos, cutoff=0.75, n_matches=3) -> pd.DataFrame:
    resultados = []
    if datos.empty:
        return pd.DataFrame()
    df = enriquecer_ofertas(datos)
    nombres_raw = df["producto"].astype(str).str.lower().tolist()
    for producto in lista_productos:
        q = producto.strip()
        if not q:
            continue
        q_norm = _normalize_text_for_clustering(q)
        mask = df["producto_norm"].str.contains(re.escape(q_norm), na=False)
        candidatas = df[mask].copy()
        if candidatas.empty and q_norm:
            q_tokens = set(q_norm.split())
            df["overlap"] = df["producto_norm"].apply(lambda s: len(q_tokens & set((s or "").split())))
            candidatas = df[df["overlap"]>0].copy().sort_values("overlap", ascending=False)
            df = df.drop(columns=["overlap"], errors="ignore")
        if candidatas.empty:
            similares = get_close_matches(q.lower(), nombres_raw, n=n_matches, cutoff=cutoff)
            if similares:
                candidatas = df[df["producto"].str.lower().isin(similares)].copy()
        if not candidatas.empty:
            candidatas["sort_key"] = candidatas["unit_price"].fillna(candidatas["precio"])
            candidatas = candidatas.sort_values("sort_key").reset_index(drop=True)
            mejor = candidatas.iloc[0]
            resultados.append({
                "query": q,
                "supermercado": mejor["supermercado"],
                "producto": mejor["producto"],
                "precio": float(mejor["precio"]),
                "unidad": mejor.get("unidad", ""),
                "unit_price": float(mejor["unit_price"]) if pd.notna(mejor.get("unit_price")) else None,
                "candidatas": candidatas[["supermercado","producto","precio","unidad","unit_price"]].to_dict(orient="records")
            })
        else:
            resultados.append({
                "query": q,
                "supermercado": None,
                "producto": None,
                "precio": None,
                "unidad": None,
                "unit_price": None,
                "candidatas": []
            })
    return pd.DataFrame(resultados)

def best_single_store(datos: pd.DataFrame, lista_productos) -> pd.DataFrame:
    """
    Calcula el costo total si se comprara toda la lista en cada tienda.
    Para cada tienda elige la mejor oferta disponible (por unit_price cuando exista, fallback precio).
    Si falta un producto en la tienda, la marca como no viable.
    Retorna DataFrame con: supermercado, feasible(bool), total, breakdown(lista de dicts)
    """
    results = []
    if datos.empty:
        return pd.DataFrame(results)
    df = enriquecer_ofertas(datos)
    stores = sorted(df["supermercado"].unique())
    for s in stores:
        store_df = df[df["supermercado"] == s].copy()
        feasible = True
        total = 0.0
        breakdown = []
        for q in lista_productos:
            q = q.strip()
            if not q:
                continue
            q_norm = _normalize_text_for_clustering(q)
            mask = store_df["producto_norm"].str.contains(re.escape(q_norm), na=False)
            candidates = store_df[mask].copy()
            if candidates.empty and q_norm:
                q_tokens = set(q_norm.split())
                store_df["overlap"] = store_df["producto_norm"].apply(lambda s2: len(q_tokens & set((s2 or "").split())))
                candidates = store_df[store_df["overlap"]>0].copy().sort_values("overlap", ascending=False)
                store_df = store_df.drop(columns=["overlap"], errors="ignore")
            if candidates.empty:
                similares = get_close_matches(q.lower(), store_df["producto"].astype(str).str.lower().tolist(), n=3, cutoff=0.8)
                if similares:
                    candidates = store_df[store_df["producto"].str.lower().isin(similares)].copy()
            if candidates.empty:
                feasible = False
                break
            candidates["sort_key"] = candidates["unit_price"].fillna(candidates["precio"])
            candidates = candidates.sort_values("sort_key").reset_index(drop=True)
            best = candidates.iloc[0]
            total += float(best["precio"])
            breakdown.append({
                "query": q,
                "producto": best["producto"],
                "precio": float(best["precio"]),
                "unidad": best.get("unidad",""),
                "unit_price": float(best["unit_price"]) if pd.notna(best.get("unit_price")) else None
            })
        results.append({
            "supermercado": s,
            "feasible": feasible,
            "total": total if feasible else None,
            "breakdown": breakdown
        })
    return pd.DataFrame(results)

def main():
    st.title("ComparaPR: Prototipo de agente comparador de precios")
    st.write("Ingresa los productos que desea revisar (separe con comas):")
    datos = leer_datos()
    entrada = st.text_input("Lista de productos (ejemplo: arroz, leche, huevos, aceite):")
    if entrada:
        lista = [p.strip() for p in entrada.split(",") if p.strip()]
        resultados = comparar_precios(datos, lista)
        st.subheader("Opción más barata por producto (selección por unit_price cuando posible)")
        for _, row in resultados.iterrows():
            q = row["query"]
            if pd.isna(row["precio"]):
                st.warning(f"No se encontró coincidencia para: '{q}'")
            else:
                precio_str = f"${row['precio']:.2f}"
                if row["unit_price"] is not None:
                    up_str = f"${row['unit_price']:.4f}"
                    st.write(f"• '{q}' → **{row['supermercado']}**: \"{row['producto']}\" — {precio_str} (unidad: {row['unidad']}) · unit_price: {up_str}")
                else:
                    st.write(f"• '{q}' → **{row['supermercado']}**: \"{row['producto']}\" — {precio_str} (unidad: {row['unidad']})")
            candidatas = row.get("candidatas", [])
            if candidatas:
                df_cand = pd.DataFrame(candidatas)
                df_cand["precio_str"] = df_cand["precio"].apply(lambda x: f"${float(x):.2f}")
                df_cand["unit_price_str"] = df_cand["unit_price"].apply(lambda x: f"${float(x):.4f}" if pd.notna(x) else "")
                st.caption(f"Candidatas para '{q}' (ordenadas por unit_price/precio):")
                st.table(df_cand[["supermercado","producto","precio_str","unit_price_str","unidad"]])
            else:
                st.info(f"No hubo candidatas para '{q}'.")
        # Comparacion comprando toda la lista en una sola tienda
        st.subheader("Costo si se compra toda la lista en una sola tienda")
        store_totals = best_single_store(datos, lista)
        if store_totals.empty:
            st.info("No hay datos para comparar por tienda.")
        else:
            # Mostrar tabla resumen por tienda
            display = []
            for _, r in store_totals.iterrows():
                display.append({
                    "supermercado": r["supermercado"],
                    "feasible": r["feasible"],
                    "total_str": f"${r['total']:.2f}" if r["feasible"] and r["total"] is not None else "N/A"
                })
            st.table(pd.DataFrame(display))
            # Mostrar tienda con menor total factible
            feasibles = store_totals[store_totals["feasible"] == True].copy()
            if not feasibles.empty:
                best = feasibles.loc[feasibles["total"].idxmin()]
                st.success(f"Mejor tienda para comprar toda la lista: {best['supermercado']} — {f'${best['total']:.2f}'}")
                st.write("Desglose de esa tienda:")
                st.table(pd.DataFrame(best["breakdown"]))
            else:
                st.info("Ninguna tienda tiene todas las coincidencias para la lista completa.")

if __name__ == "__main__":
    main()
