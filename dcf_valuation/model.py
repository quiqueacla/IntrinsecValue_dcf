"""Motor de valoración DCF.

Este módulo recibe los datos ya descargados y normalizados por ``data.py``,
prepara las series históricas, construye las proyecciones y devuelve un
``ValuationResult`` para que ``report.py`` lo presente. La descarga y la
presentación en consola quedan fuera de este módulo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CompanyData, statement_row


@dataclass
class ModelConfig:
    """Parámetros que controlan el horizonte y los escenarios de valoración."""

    risk_free_rate: float = 0.029
    equity_risk_premium: float = 0.07
    wacc: float | None = None
    terminal_growth: float = 0.02
    downside_wacc_delta: float = 0.02
    downside_growth_delta: float = -0.01
    upside_wacc_delta: float = -0.02
    upside_growth_delta: float = 0.01
    forecast_years: int = 3


@dataclass
class ValuationResult:
    """Estructura común de salida que consume la capa de presentación."""

    ticker: str
    price: float
    shares_outstanding: float
    historical: pd.DataFrame
    projections: pd.DataFrame
    scenarios: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


def _scalar(mapping: dict, key: str) -> float:
    """Obtiene un dato escalar de la ficha de Yahoo y normaliza valores inválidos."""
    try:
        value = float(mapping.get(key, np.nan))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _series(company: CompanyData, statement: str, names: tuple[str, ...]) -> pd.Series:
    """Busca una partida en un estado financiero mediante sus nombres posibles."""
    value = statement_row(company, statement, names)
    return pd.Series(dtype=float) if value is None else value.dropna().astype(float)


def _mean_or_nan(values: pd.Series) -> float:
    """Calcula un promedio ignorando valores no numéricos o no finitos."""
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def run_valuation(company: CompanyData, config: ModelConfig | None = None) -> ValuationResult:
    """Ejecuta el flujo de valoración y devuelve sus datos, sin imprimirlos.

    El proceso se organiza en cuatro etapas: recopilar y alinear datos,
    completar las métricas históricas, generar las proyecciones y valorar cada
    escenario. El resultado mantiene separadas las tablas históricas,
    proyectadas, de escenarios y los diagnósticos para facilitar el reporte.
    """
    config = config or ModelConfig()
    if not isinstance(config.forecast_years, int) or config.forecast_years < 1:
        raise ValueError("forecast_years debe ser un entero mayor que cero.")
    # 1. Leer la ficha de mercado: precio, acciones y beta complementan los
    #    estados financieros que ya vienen agrupados en CompanyData.
    info = company.info
    price = _scalar(info, "currentPrice")
    if not np.isfinite(price):
        price = _scalar(info, "regularMarketPrice")
    shares = _scalar(info, "sharesOutstanding")
    beta = _scalar(info, "beta")

    # 2. Extraer las partidas requeridas de cada estado financiero. Los nombres
    #    alternativos permiten adaptarse a las etiquetas que publique Yahoo.
    revenue = _series(company, "income", ("Total Revenue", "Revenue"))
    net_income = _series(company, "income", ("Net Income", "Net Income Common Stockholders"))
    interest = _series(company, "income", ("Interest Expense", "Interest Expense Non Operating"))
    tax_provision = _series(company, "income", ("Tax Provision", "Income Tax Expense", "Tax Expense"))
    ebit = _series(company, "income", ("EBIT", "Operating Income"))
    cfo = _series(company, "cashflow", ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"))
    capex_raw = _series(company, "cashflow", ("Capital Expenditures", "Capital Expenditure"))
    debt = _series(company, "balance", ("Total Debt",))
    liabilities = _series(company, "balance", ("Total Liabilities Net Minority Interest", "Total Liabilities"))
    debt_issued = _series(company, "cashflow", ("Long Term Debt Issuance", "Issuance Of Debt", "Proceeds From Debt"))
    debt_repaid = _series(company, "cashflow", ("Long Term Debt Payments", "Repayment Of Debt", "Repayments Of Debt"))

    # 3. Unificar las series por fecha en una tabla histórica. Así cada fila
    #    representa un periodo y las columnas representan partidas contables.
    hist = pd.concat(
        {
            "Revenue": revenue,
            "NetIncome": net_income,
            "InterestExpense": interest,
            "TotalDebt": debt,
            "TotalLiabilities": liabilities,
            "EBIT": ebit,
            "TaxProvision": tax_provision,
            "CFO": cfo,
            "CapExRaw": capex_raw,
            "DebtIssued": debt_issued,
            "DebtRepaidRaw": debt_repaid,
        },
        axis=1,
    ).sort_index()
    # 4. Completar el conjunto histórico con métricas derivadas que luego se
    #    usan tanto en el reporte como en la estimación de promedios.
    hist["TaxRate"] = (hist["TaxProvision"] / hist["EBIT"]).replace([np.inf, -np.inf], np.nan)
    hist["RevenueGrowth"] = hist["Revenue"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    hist["NetIncomeGrowth"] = hist["NetIncome"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    tax_rate = _mean_or_nan(hist["TaxRate"])
    if not np.isfinite(tax_rate):
        tax_rate = 0.25

    # En yfinance, CapEx suele venir negativo; se expresa como coste positivo.
    hist["CapEx"] = hist["CapExRaw"].abs()
    hist["InterestExpense"] = hist["InterestExpense"].abs()
    hist["OperatingFCF"] = hist["CFO"] - hist["CapEx"]
    hist["FCFF"] = hist["OperatingFCF"] + hist["InterestExpense"] * (1 - hist["TaxRate"].fillna(tax_rate))
    hist["NetIncomeMargin"] = hist["NetIncome"] / hist["Revenue"]
    hist["CostOfLoan"] = (
        hist["InterestExpense"].abs() / hist["TotalDebt"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    # Los movimientos netos de deuda se usan para mostrar FCFE histórico.
    hist["DebtRepaid"] = hist["DebtRepaidRaw"].abs()
    hist["NetDebtChange"] = hist["DebtIssued"].fillna(0) - hist["DebtRepaid"].fillna(0)
    hist["FCFE"] = hist["OperatingFCF"] + hist["NetDebtChange"]
    cost_debt_before_tax = _mean_or_nan(hist["CostOfLoan"])
    if not np.isfinite(cost_debt_before_tax):
        cost_debt_before_tax = np.nan
    cost_equity = config.risk_free_rate + config.equity_risk_premium * beta if np.isfinite(beta) else np.nan
    # Mantener consistencia con los datos mostrados: capitalización = precio × acciones.
    market_cap = price * shares if np.isfinite(price) and np.isfinite(shares) else np.nan
    total_liabilities = _scalar(info, "totalLiabilities")
    if not np.isfinite(total_liabilities):
        total_liabilities = _mean_or_nan(hist["TotalLiabilities"].dropna().tail(1))
    debt_for_weights = total_liabilities if np.isfinite(total_liabilities) else _scalar(info, "totalDebt")
    capital_total = market_cap + debt_for_weights
    equity_weight = market_cap / capital_total if capital_total > 0 else np.nan
    debt_weight = debt_for_weights / capital_total if capital_total > 0 else np.nan
    cost_of_debt_after_tax = cost_debt_before_tax * (1 - tax_rate) if np.isfinite(cost_debt_before_tax) else np.nan
    wacc_calculated = (
        equity_weight * cost_equity + debt_weight * cost_of_debt_after_tax
        if np.isfinite(equity_weight) and np.isfinite(debt_weight)
        and np.isfinite(cost_equity) and np.isfinite(cost_of_debt_after_tax)
        else np.nan
    )
    wacc_used = config.wacc if config.wacc is not None else (wacc_calculated if np.isfinite(wacc_calculated) else 0.09)

    # 5. Detener el proceso pronto si faltan los datos mínimos para continuar.
    if revenue.empty or net_income.empty or hist["FCFF"].dropna().empty:
        raise ValueError(f"{company.ticker}: faltan ingresos, beneficio neto o flujo de caja para valorar.")
    if not np.isfinite(shares) or shares <= 0:
        raise ValueError(f"{company.ticker}: Yahoo Finance no ha proporcionado acciones en circulación válidas.")

    # 6. Preparar las series de referencia que alimentan las proyecciones.
    hist_revenue = hist["Revenue"].dropna()
    hist_net_income = hist["NetIncome"].dropna()
    hist_margin = (hist["NetIncome"] / hist["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna()
    hist_fcff_ratio = (hist["FCFF"] / hist["NetIncome"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    hist_cfo_margin = (hist["CFO"] / hist["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna()
    hist_capex_margin = (hist["CapEx"] / hist["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna()
    hist_ebit_margin = (hist["EBIT"] / hist["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna()
    if hist_margin.empty or hist_fcff_ratio.empty:
        raise ValueError(f"{company.ticker}: no hay suficiente historial para estimar márgenes y conversión a FCFF.")

    margin_mean = float(hist_margin.mean())
    fcff_ratio_mean = float(hist_fcff_ratio.mean())
    cfo_margin_mean = _mean_or_nan(hist_cfo_margin)
    capex_margin_mean = _mean_or_nan(hist_capex_margin)
    ebit_margin_mean = _mean_or_nan(hist_ebit_margin)
    growth_mean = _mean_or_nan(hist["RevenueGrowth"])
    net_income_growth_mean = _mean_or_nan(hist["NetIncomeGrowth"])
    if not np.isfinite(growth_mean):
        growth_mean = config.terminal_growth

    # 7. Construir el índice de años futuros y las series proyectadas con el
    #    horizonte indicado en ModelConfig.
    last_revenue = float(hist_revenue.iloc[-1])
    forecast_revenue = []
    base_revenue = last_revenue
    for _ in range(config.forecast_years):
        base_revenue *= 1 + growth_mean
        forecast_revenue.append(base_revenue)

    last_year = max(hist_revenue.index).year if hasattr(max(hist_revenue.index), "year") else int(str(max(hist_revenue.index))[:4])
    forecast_index = [str(last_year + i) for i in range(1, config.forecast_years + 1)]
    revenue_forecast = pd.Series(forecast_revenue, index=forecast_index, dtype=float)
    # Proyectar el beneficio con el margen neto medio histórico, no con el
    # crecimiento histórico del beneficio neto.
    net_income_forecast = revenue_forecast * margin_mean
    ebit_forecast = revenue_forecast * ebit_margin_mean if np.isfinite(ebit_margin_mean) else pd.Series(np.nan, index=forecast_index)
    tax_provision_forecast = ebit_forecast * tax_rate
    cfo_forecast = revenue_forecast * cfo_margin_mean if np.isfinite(cfo_margin_mean) else pd.Series(np.nan, index=forecast_index)
    capex_forecast = revenue_forecast * capex_margin_mean if np.isfinite(capex_margin_mean) else pd.Series(np.nan, index=forecast_index)
    starting_debt = _mean_or_nan(hist["TotalDebt"].dropna().tail(1))
    if not np.isfinite(starting_debt):
        starting_debt = 0.0
    projected_debt = pd.Series(starting_debt, index=forecast_index, dtype=float)
    projected_interest = projected_debt * cost_debt_before_tax if np.isfinite(cost_debt_before_tax) else pd.Series(np.nan, index=forecast_index)
    projected_operating_fcf = cfo_forecast - capex_forecast
    # Aplicar a cada año proyectado el ratio FCFF/NetIncome medio observado.
    projected_fcff = net_income_forecast * fcff_ratio_mean
    projected_net_debt_change = pd.Series(0.0, index=forecast_index)
    projected_fcfe = projected_operating_fcf + projected_net_debt_change
    projections = pd.DataFrame(
        {
            "Revenue": revenue_forecast,
            "RevenueGrowth": revenue_forecast.pct_change(fill_method=None).fillna(growth_mean),
            "NetIncome": net_income_forecast,
            "NetIncomeGrowth": net_income_forecast.pct_change(fill_method=None).fillna(
                net_income_forecast.iloc[0] / hist_net_income.iloc[-1] - 1
                if not hist_net_income.empty and hist_net_income.iloc[-1] != 0
                else np.nan
            ),
            "NetIncomeMargin": net_income_forecast / revenue_forecast,
            "EBIT": ebit_forecast,
            "TaxProvision": tax_provision_forecast,
            "CFO": cfo_forecast,
            "CapEx": capex_forecast,
            "OperatingFCF": projected_operating_fcf,
            "TotalDebt": projected_debt,
            "CostOfLoan": cost_debt_before_tax,
            "InterestExpense": projected_interest,
            "TaxRate": tax_rate,
            "FCFF": projected_fcff,
            "FCFF_NI_Ratio": fcff_ratio_mean,
            "DebtIssued": 0.0,
            "DebtRepaid": 0.0,
            "NetDebtChange": projected_net_debt_change,
            "FCFE": projected_fcfe,
            "FCFE_NI_Ratio": projected_fcfe / net_income_forecast.replace(0, np.nan),
        }
    )
    fcff_forecast = projections["FCFF"]

    # 8. Evaluar los mismos flujos proyectados bajo las hipótesis de cada
    #    escenario y guardar tanto los resultados agregados como el detalle
    #    anual que mostrará el reporte.
    scenarios = {
        "Bajista": (wacc_used + config.downside_wacc_delta, config.terminal_growth + config.downside_growth_delta),
        "Neutro": (wacc_used, config.terminal_growth),
        "Alcista": (wacc_used + config.upside_wacc_delta, config.terminal_growth + config.upside_growth_delta),
    }
    total_debt = _scalar(info, "totalDebt")
    total_cash = _scalar(info, "totalCash")
    net_debt_available = np.isfinite(total_debt) and np.isfinite(total_cash)
    net_debt = total_debt - total_cash if net_debt_available else 0.0
    scenario_rows = {}
    details = []
    for name, (wacc, terminal_growth) in scenarios.items():
        if wacc <= terminal_growth:
            raise ValueError(f"Escenario {name}: WACC ({wacc:.2%}) debe superar el crecimiento terminal ({terminal_growth:.2%}).")
        cashflows = fcff_forecast.to_numpy(dtype=float)
        discount = 1 / (1 + wacc) ** np.arange(1, config.forecast_years + 1)
        terminal_value = cashflows[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_forecasts = cashflows * discount
        pv_terminal = terminal_value * discount[-1]
        enterprise_value = pv_forecasts.sum() + pv_terminal
        equity_value = enterprise_value - net_debt  ##Si restamos net_debt estamos obteniendo el valor que queda a los accionistas, útil para valorar la empresa
        intrinsic_per_share = equity_value / shares
        scenario_rows[name] = {
            "WACC": wacc,
            "TerminalGrowth": terminal_growth,
            "Price": price,
            "EnterpriseValue": enterprise_value,
            "NetDebt": net_debt,
            "EquityValue": equity_value,
            "IntrinsicValuePerShare": intrinsic_per_share,
            "MarginOfSafety": intrinsic_per_share / price - 1 if np.isfinite(price) and price > 0 else np.nan,
        }
        for year, cf, df, pv in zip(forecast_index, cashflows, discount, pv_forecasts):
            details.append({"Scenario": name, "Period": year, "FCFF": cf, "DiscountFactor": df, "PresentValue": pv})
        details.append({"Scenario": name, "Period": "Terminal", "FCFF": terminal_value, "DiscountFactor": discount[-1], "PresentValue": pv_terminal})

    summary = pd.DataFrame(scenario_rows)
    summary["Mean"] = summary.mean(axis=1, numeric_only=True)
    # 9. Entregar un único objeto de salida a main.py/report.py. Mantener aquí
    #    los resultados en memoria evita acoplar el motor a un formato de salida.
    return ValuationResult(
        ticker=company.ticker,
        price=price,
        shares_outstanding=shares,
        historical=hist,
        projections=projections,
        scenarios=pd.DataFrame(details),
        summary=summary,
        diagnostics={
            "beta": beta,
            "tax_rate_mean": tax_rate,
            "cost_of_debt_before_tax": cost_debt_before_tax,
            "cost_of_equity": cost_equity,
            "wacc_calculated": wacc_calculated,
            "wacc_used": wacc_used,
            "risk_free_rate": config.risk_free_rate,
            "equity_risk_premium": config.equity_risk_premium,
            "market_return": config.risk_free_rate + config.equity_risk_premium,
            "cost_of_debt_after_tax": cost_of_debt_after_tax,
            "market_cap": market_cap,
            "total_liabilities": total_liabilities,
            "equity_weight": equity_weight,
            "debt_weight": debt_weight,
            "revenue_growth_mean": growth_mean,
            "net_income_growth_mean_historical": net_income_growth_mean,
            "net_income_margin_mean": margin_mean,
            "fcff_to_net_income_mean": fcff_ratio_mean,
            "net_debt": net_debt if net_debt_available else np.nan,
            "net_debt_adjustment_used": net_debt_available,
            "currency": info.get("currency", ""),
        },
    )
