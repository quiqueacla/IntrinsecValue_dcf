"""Capa de presentación del resultado DCF.

Este módulo recibe el ``ValuationResult`` producido por ``model.py`` y lo
convierte en tablas legibles para consola. No descarga datos ni ejecuta la
valoración; su responsabilidad es organizar y formatear la salida.
"""

from __future__ import annotations

import math

import pandas as pd

from .model import ValuationResult


def _number(value: object) -> float:
    """Convierte una entrada a número finito para las funciones de formato."""
    try:
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _money(value: object, currency: str) -> str:
    """Formatea un importe para mostrarlo en las tablas del informe."""
    number = _number(value)
    if pd.isna(number):
        return "—"
    formatted = f"{abs(number):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    sign = "-" if number < 0 else ""
    unit = "€" if currency == "EUR" else currency
    return f"{sign}{formatted} {unit}".strip()


def _percent(value: object) -> str:
    """Formatea un valor decimal como porcentaje localizado."""
    number = _number(value)
    return "—" if pd.isna(number) else f"{number:.2%}".replace(".", ",")


def _plain(value: object) -> str:
    """Formatea un número sin asignarle unidades monetarias."""
    number = _number(value)
    return "—" if pd.isna(number) else f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _year_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapta una tabla con periodos en filas al formato de años en columnas."""
    out = frame.T.copy()
    out.columns = [str(getattr(column, "year", column)) for column in out.columns]
    return out


def _print_year_table(
    title: str,
    frame: pd.DataFrame,
    rows: list[tuple[str, str, str]],
    currency: str,
) -> None:
    """Renderiza un conjunto seleccionado de métricas con años en columnas.

    Cada definición de fila contiene la etiqueta visible, la columna de origen
    y el formato que debe usar la capa de presentación.
    """
    source = _year_columns(frame)
    table = pd.DataFrame(index=[label for label, _, _ in rows], columns=source.columns, dtype=object)
    for label, column, kind in rows:
        if column not in source.index:
            continue
        for year in source.columns:
            value = source.loc[column, year]
            if kind == "money":
                formatted = _money(value, currency)
            elif kind == "percent":
                formatted = _percent(value)
            else:
                formatted = _plain(value)
            table.loc[label, year] = formatted
    print(f"\n{title}")
    print(table.to_string())


def _print_scenario_breakdowns(result: ValuationResult, currency: str) -> None:
    """Presenta el detalle anual y los valores calculados para cada escenario."""
    preferred_order = ["Alcista", "Neutro", "Bajista"]
    scenarios = [name for name in preferred_order if name in result.summary.columns]
    base_wacc = _number(result.summary.loc["WACC", "Neutro"]) if "Neutro" in result.summary else float("nan")
    base_growth = _number(result.summary.loc["TerminalGrowth", "Neutro"]) if "Neutro" in result.summary else float("nan")
    historical = _year_columns(result.historical)
    historical_periods = list(dict.fromkeys(historical.columns))
    forecast_periods = [str(period) for period in result.projections.index]
    periods = [*historical_periods, *forecast_periods, "Terminal Value"]
    # Agrupar visualmente los años proyectados, como en la tabla de referencia.
    grouped_periods = [
        ("", period) if period in historical_periods else ("Estimations", period)
        for period in [*historical_periods, *forecast_periods]
    ] + [("", "Terminal Value")]

    for scenario in scenarios:
        rows = result.scenarios.loc[result.scenarios["Scenario"] == scenario].copy()
        rows = rows.set_index("Period")
        table = pd.DataFrame(index=["Free Cashflow (FCFF)", "Discount Factor", "PV of future Cash Flow"], columns=periods, dtype=object)
        if "FCFF" in historical.index:
            for period in historical_periods:
                table.loc["Free Cashflow (FCFF)", period] = _money(historical.loc["FCFF", period], currency)
        for period in forecast_periods:
            if period not in rows.index:
                continue
            flow = rows.loc[period]
            table.loc["Free Cashflow (FCFF)", period] = _money(flow["FCFF"], currency)
            table.loc["Discount Factor", period] = _plain(1 / flow["DiscountFactor"] if flow["DiscountFactor"] else float("nan"))
            table.loc["PV of future Cash Flow", period] = _money(flow["PresentValue"], currency)
        if "Terminal" in rows.index:
            terminal = rows.loc["Terminal"]
            table.loc["Free Cashflow (FCFF)", "Terminal Value"] = _money(terminal["FCFF"], currency)
            table.loc["Discount Factor", "Terminal Value"] = _plain(1 / terminal["DiscountFactor"] if terminal["DiscountFactor"] else float("nan"))
            table.loc["PV of future Cash Flow", "Terminal Value"] = _money(terminal["PresentValue"], currency)

        scenario_values = result.summary[scenario]
        valuation = pd.DataFrame(
            [
                ("Required Return WACC", _percent(scenario_values["WACC"])),
                ("WACC Delta", _percent(scenario_values["WACC"] - base_wacc) if pd.notna(base_wacc) else "—"),
                ("Perpetual Growth", _percent(scenario_values["TerminalGrowth"])),
                ("Growth Delta", _percent(scenario_values["TerminalGrowth"] - base_growth) if pd.notna(base_growth) else "—"),
                ("Enterprise Value", _money(scenario_values["EnterpriseValue"], currency)),
                ("Net Debt", _money(scenario_values["NetDebt"], currency)),
                ("Today's Value (Equity)", _money(scenario_values["EquityValue"], currency)),
                ("Shares Out", f"{result.shares_outstanding:,.0f}"),
                ("Price Target", _money(scenario_values["IntrinsicValuePerShare"], currency)),
            ],
            columns=["VALUATION", "Value"],
        )
        table.columns = pd.MultiIndex.from_tuples(grouped_periods)
        print(f"\nESCENARIO: {scenario.upper()}")
        print(table.to_string())
        print(valuation.to_string(index=False))


def print_report(result: ValuationResult) -> None:
    """Imprime el informe completo a partir del resultado del modelo.

    Organiza la salida en información general, supuestos de tasas, métricas
    históricas, proyecciones, detalle por escenario y comparación final. La
    estructura de cálculo se mantiene en ``model.py``.
    """
    currency = result.diagnostics.get("currency", "") or ""
    diag = result.diagnostics
    market_cap = diag.get("market_cap")
    liabilities = diag.get("total_liabilities")
    equity_weight = diag.get("equity_weight")
    debt_weight = diag.get("debt_weight")

    # Datos de mercado y composición de capital usados como contexto del modelo.
    print(f"\nValoración DCF: {result.ticker}")
    print("=" * 92)
    main_info = pd.DataFrame(
        [
            ("Share Price", _money(result.price, currency)),
            ("Shares Out", _plain(result.shares_outstanding)),
            ("Market Cap", _money(market_cap, currency)),
            ("Total liabilities", _money(liabilities, currency)),
            ("% Equity", _percent(equity_weight)),
            ("% Debt", _percent(debt_weight)),
        ],
        columns=["MAIN INFORMATION", "Value"],
    )
    print("\n" + main_info.to_string(index=False))

    # Mostrar por separado los supuestos de rentabilidad exigida y financiación.
    cost_equity = pd.DataFrame(
        [
            ("Risk Free Rate (Rf)", _percent(diag.get("risk_free_rate"))),
            ("Beta Stock (β)", _plain(diag.get("beta"))),
            ("Stock Market return (Rm)", _percent(diag.get("market_return"))),
            ("ERP (Equity Risk Premium)", _percent(diag.get("equity_risk_premium"))),
            ("Cost of Equity", _percent(diag.get("cost_of_equity"))),
        ],
        columns=["COST OF EQUITY", "Value"],
    )
    print("\n" + cost_equity.to_string(index=False))

    cost_debt = pd.DataFrame(
        [
            ("Cost of loans", _percent(diag.get("cost_of_debt_before_tax"))),
            ("Tax Rate", _percent(diag.get("tax_rate_mean"))),
            ("Cost of Debt", _percent(diag.get("cost_of_debt_after_tax"))),
            ("WACC utilizado", _percent(diag.get("wacc_used"))),
        ],
        columns=["COST OF DEBT", "Value"],
    )
    print("\n" + cost_debt.to_string(index=False))

    # Resumir los datos históricos que sustentan el coste de deuda y los impuestos.
    years = _year_columns(result.historical)
    debt_rows = [
        ("Deuda Total", "TotalDebt", "money"),
        ("Gastos Intereses", "InterestExpense", "money"),
        ("Cost Loan (%)", "CostOfLoan", "percent"),
        ("EBIT", "EBIT", "money"),
        ("Provision Impuestos", "TaxProvision", "money"),
        ("Tax Rate", "TaxRate", "percent"),
    ]
    debt_table = pd.DataFrame(index=[label for label, _, _ in debt_rows], columns=["Promedio", *years.columns], dtype=object)
    for label, source_row, kind in debt_rows:
        if source_row not in years.index:
            continue
        for year in years.columns:
            value = years.loc[source_row, year]
            debt_table.loc[label, year] = _money(value, currency) if kind == "money" else _percent(value)
        if source_row in ("CostOfLoan", "TaxRate"):
            debt_table.loc[label, "Promedio"] = _percent(pd.to_numeric(years.loc[source_row], errors="coerce").mean())
        else:
            debt_table.loc[label, "Promedio"] = ""
    print("\nCoste de deuda y tipo impositivo calculados a partir de los históricos")
    print(debt_table.to_string())

    # Mantener una selección común de filas para las tablas anual histórica y proyectada.
    common_rows = [
        ("Revenues / Ventas", "Revenue", "money"),
        ("Growth %", "RevenueGrowth", "percent"),
        ("Net income / Beneficio neto", "NetIncome", "money"),
        ("Net income Growth", "NetIncomeGrowth", "percent"),
        ("Net income Margin", "NetIncomeMargin", "percent"),
        ("EBIT", "EBIT", "money"),
        ("Provision Impuestos", "TaxProvision", "money"),
        ("CFO", "CFO", "money"),
        ("Capex", "CapEx", "money"),
        ("FCF genérico", "OperatingFCF", "money"),
        ("Intereses", "InterestExpense", "money"),
        ("Free Cashflow (FCFF)", "FCFF", "money"),
        ("FCFF/Net Income", "FCFF_NI_Ratio", "percent"),
        ("Emision Deuda", "DebtIssued", "money"),
        ("Reembolso deuda", "DebtRepaid", "money"),
        ("Incremento Deuda Neto", "NetDebtChange", "money"),
        ("FCFE", "FCFE", "money"),
        ("FCFE/Net income", "FCFE_NI_Ratio", "percent"),
    ]

    # Las tablas anuales salen antes del detalle de escenarios para dar contexto
    # a las hipótesis y flujos que se usan en la valoración.
    historical = result.historical.copy()
    historical["RevenueGrowth"] = historical["Revenue"].pct_change(fill_method=None)
    historical["NetIncomeGrowth"] = historical["NetIncome"].pct_change(fill_method=None)
    historical["FCFF_NI_Ratio"] = historical["FCFF"] / historical["NetIncome"].replace(0, float("nan"))
    historical["FCFE_NI_Ratio"] = historical["FCFE"] / historical["NetIncome"].replace(0, float("nan"))
    _print_year_table("Históricos (años en columnas)", historical, common_rows, currency)

    projections = result.projections.copy()
    if "FCFE_NI_Ratio" not in projections:
        projections["FCFE_NI_Ratio"] = projections["FCFE"] / projections["NetIncome"].replace(0, float("nan"))
    projection_rows = common_rows + [
        ("Cost Loan (%)", "CostOfLoan", "percent"),
        ("Tax Rate", "TaxRate", "percent"),
    ]
    _print_year_table("Proyecciones (años en columnas)", projections, projection_rows, currency)

    # Cada escenario muestra su recorrido de flujos y valoración antes del
    # cuadro comparativo final.
    _print_scenario_breakdowns(result, currency)

    print("\nResumen de escenarios")
    cols = ["WACC", "TerminalGrowth", "Price", "IntrinsicValuePerShare", "MarginOfSafety"]
    # Esta copia mezcla números con cadenas formateadas (Margin of Safety).
    # Usar dtype object evita asignaciones de texto a filas float64 en pandas.
    summary = result.summary.loc[cols].copy().astype(object)
    for scenario in summary.columns:
        summary.loc["MarginOfSafety", scenario] = _percent(summary.loc["MarginOfSafety", scenario])
    print(summary.to_string())
    if pd.isna(diag.get("net_debt")):
        print("\nAviso: faltan deuda total o caja; se ha supuesto deuda neta cero para calcular el valor del capital.")
