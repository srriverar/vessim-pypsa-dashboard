"""Editable "sheets": pandas tables that behave like the AdjieBrotot grids.

The browser tools expose every element as a spreadsheet with add/remove rows.
``to_sheet`` / ``from_sheet`` reproduce that contract in Streamlit:
the element name travels in a real ``name`` column (so new rows can be typed),
and the canonical dtypes are restored after every edit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..grid import empty_table
from .theme import wide

NUMERIC = ("float64", "int64", "float32", "int32", "int16")


def to_sheet(frame: pd.DataFrame, prefix: str = "el") -> pd.DataFrame:
    """Index -> ``name`` column, ready for ``st.data_editor(num_rows='dynamic')``."""
    sheet = frame.reset_index()
    first = sheet.columns[0]
    if first not in ("name", "index"):
        sheet = sheet.rename(columns={first: "name"})
    elif first == "index":
        sheet = sheet.rename(columns={"index": "name"})
    if sheet["name"].dtype.kind in "if":
        sheet["name"] = [f"{prefix}{i + 1}" for i in range(len(sheet))]
    sheet["name"] = sheet["name"].astype(str)
    return sheet


def from_sheet(sheet: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Edit -> canonical frame, dropping unnamed rows and coercing dtypes."""
    template = empty_table(kind)
    sheet = sheet.copy()
    if "name" not in sheet.columns:
        sheet.insert(0, "name", [f"el{i + 1}" for i in range(len(sheet))])
    sheet["name"] = sheet["name"].fillna("").astype(str).str.strip()
    sheet = sheet[sheet["name"] != ""]
    sheet = sheet.drop_duplicates(subset="name", keep="first").set_index("name")

    for column in template.columns:
        if column not in sheet.columns:
            sheet[column] = template[column].dtype.type(0) if template[column].dtype.kind in "if" else ""
        dtype = template[column].dtype
        if dtype.kind == "b":
            series = sheet[column]
            if series.dtype.kind != "b":
                series = series.astype(str).str.lower().isin(["true", "1", "yes"])
            sheet[column] = series.astype(bool)
        elif dtype.kind in "if":
            series = pd.to_numeric(sheet[column], errors="coerce")
            if dtype.kind == "i":
                sheet[column] = series.round().fillna(0).astype("int64")
            else:
                sheet[column] = series.fillna(0.0).astype("float64")
        else:
            sheet[column] = sheet[column].fillna("").astype(str)
    return sheet[list(template.columns)]


def editor(kind: str, frame: pd.DataFrame, key: str, columns: list[str] | None = None,
           column_config: dict | None = None, height: int = 260,
           prefix: str = "el") -> pd.DataFrame:
    """Render one editable sheet and return the canonical frame."""
    sheet = to_sheet(frame, prefix=prefix)
    order = ["name"] + [c for c in (columns or list(sheet.columns)) if c in sheet.columns]
    hidden = [c for c in sheet.columns if c not in order]
    view = sheet[order]
    edited = st.data_editor(
        view, key=key, num_rows="dynamic", **wide(), height=height,
        column_config=column_config or {},
        column_order=order,
    )
    # re-attach the columns the user did not see
    merged = edited.copy()
    for column in hidden:
        if len(merged) == len(sheet):
            merged[column] = sheet[column].to_numpy()
        else:
            merged[column] = np.nan
    return from_sheet(merged, kind)


def tabular(frame: pd.DataFrame, height: int = 300, formats: dict | None = None,
            highlight: str | None = None) -> None:
    view = frame
    if formats:
        view = view.style.format(formats, na_rep="—")
    st.dataframe(view, **wide(), height=height)


def net_editor(payload: dict, key_prefix: str, presets_info: dict | None = None) -> dict:
    """Full multi-sheet editor for a :class:`PowerNetwork` payload.

    Returns the updated payload.  The caller decides when to re-solve.
    """
    parts = payload["parts"]
    buses = list(parts["buses"].index)
    bus_options = buses or [""]
    line_types = ["", *__line_types()]

    tabs = st.tabs(["🟦 Buses", "➖ Lines", "🔁 Transformers", "⚡ Generators",
                    "🏭 Loads", "🔋 Storage", "🔗 Links", "🧲 Shunts"])

    with tabs[0]:
        parts["buses"] = editor(
            "buses", parts["buses"], f"{key_prefix}::buses",
            columns=["v_nom", "v_set", "type", "carrier"],
            column_config={
                "v_nom": st.column_config.NumberColumn("v_nom (kV)", min_value=0.1, step=1.0, format="%.2f"),
                "v_set": st.column_config.NumberColumn("v_set (p.u.)", min_value=0.5, max_value=1.2,
                                                       step=0.005, format="%.4f"),
                "type": st.column_config.SelectboxColumn("Type", options=["PQ", "PV", "Slack"]),
                "carrier": st.column_config.TextColumn("Carrier"),
            })

    with tabs[1]:
        parts["lines"] = editor(
            "lines", parts["lines"], f"{key_prefix}::lines",
            columns=["bus0", "bus1", "type", "length", "num_parallel", "r", "x", "b", "g", "s_nom"],
            column_config={
                "bus0": st.column_config.SelectboxColumn("From", options=bus_options),
                "bus1": st.column_config.SelectboxColumn("To", options=bus_options),
                "type": st.column_config.SelectboxColumn("Line type", options=line_types),
                "length": st.column_config.NumberColumn("Length (km)", min_value=0.0, step=1.0, format="%.2f"),
                "num_parallel": st.column_config.NumberColumn("Circuits", min_value=1, step=1),
                "r": st.column_config.NumberColumn("r (Ω)", min_value=0.0, step=0.01, format="%.4f"),
                "x": st.column_config.NumberColumn("x (Ω)", min_value=0.0, step=0.01, format="%.4f"),
                "b": st.column_config.NumberColumn("b (S)", step=1e-6, format="%.6f"),
                "g": st.column_config.NumberColumn("g (S)", step=1e-6, format="%.6f"),
                "s_nom": st.column_config.NumberColumn("Rating (MVA)", min_value=0.0, step=1.0, format="%.1f"),
            }, height=300)

    with tabs[2]:
        parts["transformers"] = editor(
            "transformers", parts["transformers"], f"{key_prefix}::transformers",
            columns=["bus0", "bus1", "s_nom", "r", "x", "tap_ratio", "phase_shift", "model"],
            column_config={
                "bus0": st.column_config.SelectboxColumn("From", options=bus_options),
                "bus1": st.column_config.SelectboxColumn("To", options=bus_options),
                "s_nom": st.column_config.NumberColumn("Rating (MVA)", step=1.0, format="%.2f"),
                "r": st.column_config.NumberColumn("r (p.u.)", step=0.001, format="%.4f"),
                "x": st.column_config.NumberColumn("x (p.u.)", step=0.001, format="%.4f"),
                "tap_ratio": st.column_config.NumberColumn("Tap", step=0.005, format="%.4f"),
                "phase_shift": st.column_config.NumberColumn("Phase shift (°)", step=1.0, format="%.2f"),
                "model": st.column_config.SelectboxColumn("Model", options=["t", "pi"]),
            }, height=240)

    with tabs[3]:
        parts["generators"] = editor(
            "generators", parts["generators"], f"{key_prefix}::generators",
            columns=["bus", "control", "p_set", "q_set", "p_nom", "carrier", "marginal_cost"],
            column_config={
                "bus": st.column_config.SelectboxColumn("Bus", options=bus_options),
                "control": st.column_config.SelectboxColumn("Control", options=["PQ", "PV", "Slack"]),
                "p_set": st.column_config.NumberColumn("P (MW)", step=1.0, format="%.2f"),
                "q_set": st.column_config.NumberColumn("Q (MVAr)", step=1.0, format="%.2f"),
                "p_nom": st.column_config.NumberColumn("Rating (MW)", min_value=0.0, step=1.0, format="%.2f"),
                "carrier": st.column_config.SelectboxColumn(
                    "Carrier", options=["", "gas", "coal", "diesel", "hydro", "solar", "wind", "battery", "grid"]),
                "marginal_cost": st.column_config.NumberColumn("Marginal cost (USD/MWh)", step=1.0, format="%.2f"),
            }, height=260)

    with tabs[4]:
        parts["loads"] = editor(
            "loads", parts["loads"], f"{key_prefix}::loads",
            columns=["bus", "p_set", "q_set"],
            column_config={
                "bus": st.column_config.SelectboxColumn("Bus", options=bus_options),
                "p_set": st.column_config.NumberColumn("P (MW)", min_value=0.0, step=1.0, format="%.2f"),
                "q_set": st.column_config.NumberColumn("Q (MVAr)", step=1.0, format="%.2f"),
            }, height=220)

    with tabs[5]:
        parts["storages"] = editor(
            "storages", parts["storages"], f"{key_prefix}::storages",
            columns=["bus", "p_set", "q_set", "p_nom", "max_hours", "efficiency_store",
                     "efficiency_dispatch", "soc0"],
            column_config={
                "bus": st.column_config.SelectboxColumn("Bus", options=bus_options),
                "p_set": st.column_config.NumberColumn("P (MW)", step=1.0, format="%.2f"),
                "q_set": st.column_config.NumberColumn("Q (MVAr)", step=1.0, format="%.2f"),
                "p_nom": st.column_config.NumberColumn("Rating (MW)", min_value=0.0, step=1.0, format="%.2f"),
                "max_hours": st.column_config.NumberColumn("Hours", min_value=0.0, step=0.5, format="%.2f"),
                "efficiency_store": st.column_config.NumberColumn("η charge", min_value=0.1, max_value=1.0,
                                                                  step=0.01, format="%.3f"),
                "efficiency_dispatch": st.column_config.NumberColumn("η discharge", min_value=0.1, max_value=1.0,
                                                                     step=0.01, format="%.3f"),
                "soc0": st.column_config.NumberColumn("SoC₀", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
            }, height=240)

    with tabs[6]:
        parts["links"] = editor(
            "links", parts["links"], f"{key_prefix}::links",
            columns=["bus0", "bus1", "p_set", "p_nom", "efficiency"],
            column_config={
                "bus0": st.column_config.SelectboxColumn("From", options=bus_options),
                "bus1": st.column_config.SelectboxColumn("To", options=bus_options),
                "p_set": st.column_config.NumberColumn("P (MW)", step=1.0, format="%.2f"),
                "p_nom": st.column_config.NumberColumn("Rating (MW)", min_value=0.0, step=1.0, format="%.2f"),
                "efficiency": st.column_config.NumberColumn("η", min_value=0.0, max_value=1.0, step=0.01,
                                                             format="%.3f"),
            }, height=200)

    with tabs[7]:
        parts["shunts"] = editor(
            "shunts", parts["shunts"], f"{key_prefix}::shunts",
            columns=["bus", "g", "b"],
            column_config={
                "bus": st.column_config.SelectboxColumn("Bus", options=bus_options),
                "g": st.column_config.NumberColumn("g (S)", step=1e-4, format="%.6f"),
                "b": st.column_config.NumberColumn("b (S)", step=1e-3, format="%.6f"),
            }, height=180)

    return payload


def __line_types() -> list[str]:
    from ..presets import line_type_names

    try:
        return line_type_names()
    except Exception:  # pragma: no cover
        return []
