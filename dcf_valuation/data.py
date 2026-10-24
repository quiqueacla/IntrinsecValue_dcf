"""Capa de acceso y adaptación de datos financieros.

Este módulo es la frontera entre Yahoo Finance y el resto de la aplicación:
descarga la información, normaliza los estados anuales y la entrega como un
objeto ``CompanyData``. El motor del modelo recibe ese objeto sin gestionar
directamente la conexión con Yahoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class CompanyData:
    """Contenedor con los datos de una empresa que consumirá el modelo DCF."""

    ticker: str
    info: dict
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    revenue_estimates: pd.Series


def _tidy_statement(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Devuelve un estado financiero con fechas ordenadas y formato uniforme."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = pd.to_datetime(out.columns).date
    return out.reindex(sorted(out.columns), axis=1)


def _find_row(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series | None:
    """Localiza una partida contable usando un conjunto de etiquetas posibles.

    La búsqueda normaliza las etiquetas para que diferencias de espacios,
    signos o mayúsculas entre respuestas de Yahoo no se propaguen al modelo.
    """
    if frame is None or frame.empty:
        return None
    normalize = lambda value: "".join(ch for ch in str(value).casefold() if ch.isalnum())
    by_normalized = {normalize(label): label for label in frame.index}
    for name in names:
        label = by_normalized.get(normalize(name))
        if label is not None:
            row = pd.to_numeric(frame.loc[label], errors="coerce").astype(float)
            row.index = pd.to_datetime(row.index).date
            return row
    return None


def _estimate_for_year(estimates: pd.DataFrame, key: str) -> float:
    """Extrae el valor medio de una fila de estimaciones de Yahoo."""
    if estimates is None or estimates.empty or key not in estimates.index:
        return np.nan
    row = estimates.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return float(pd.to_numeric(row.get("avg", np.nan), errors="coerce"))


def download_company_data(symbol: str) -> CompanyData:
    """Descarga y empaqueta los datos de entrada necesarios para una valoración.

    Agrupa la ficha de mercado, estados financieros y estimaciones en una sola
    estructura para que las capas de modelo y reporte trabajen con la misma
    representación de la empresa.
    """
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception as exc:
        raise RuntimeError(f"No se pudo obtener la ficha de {symbol}: {exc}") from exc

    try:
        income = _tidy_statement(ticker.financials)
        balance = _tidy_statement(ticker.balance_sheet)
        cashflow = _tidy_statement(ticker.cashflow)
        estimates = ticker.revenue_estimate
    except Exception as exc:
        raise RuntimeError(f"No se pudieron descargar los estados de {symbol}: {exc}") from exc

    estimate_series = pd.Series(
        {"0y": _estimate_for_year(estimates, "0y"), "+1y": _estimate_for_year(estimates, "+1y")},
        dtype=float,
    )
    return CompanyData(symbol, info, income, balance, cashflow, estimate_series)


def statement_row(company: CompanyData, statement: str, names: Iterable[str]) -> pd.Series | None:
    """Interfaz de consulta que usa el modelo para obtener una partida anual."""
    frame = getattr(company, statement)
    return _find_row(frame, names)
