"""Punto de entrada de la aplicación DCF.

Uso: ``python main.py REP.MC [opciones]``.

Este módulo coordina las capas de la aplicación: lee los argumentos de
consola, crea la configuración, solicita los datos a ``data.py``, ejecuta el
modelo de ``model.py`` y entrega el resultado a ``report.py`` para mostrarlo.
La lógica de acceso a datos, valoración y presentación vive en esos módulos.
"""

from __future__ import annotations

import argparse

from dcf_valuation import ModelConfig, download_company_data, run_valuation
from dcf_valuation.report import print_report


def parse_args() -> argparse.Namespace:
    """Define y procesa las opciones de ejecución de consola."""
    parser = argparse.ArgumentParser(description="Estima valor intrínseco FCFF/DCF para un ticker de Yahoo Finance.")
    parser.add_argument("ticker", help="Ticker de Yahoo Finance; por ejemplo REP.MC, IDR.MC o VOW.DE")
    parser.add_argument("--risk-free", type=float, default=0.029, help="Tipo libre de riesgo (por defecto: 0.029)")
    parser.add_argument("--equity-risk-premium", type=float, default=0.07, help="Prima de riesgo de mercado (por defecto: 0.07)")
    parser.add_argument("--wacc", type=float, default=0.071, help="WACC base usado (por defecto: 0.09)")
    parser.add_argument("--terminal-growth", type=float, default=0.03, help="Crecimiento perpetuo base (por defecto: 0.02)")
    return parser.parse_args()


def main() -> None:
    """Orquesta una ejecución completa para el ticker indicado.

    El flujo conecta los módulos de datos, modelo y reporte sin duplicar aquí
    sus responsabilidades internas.
    """
    args = parse_args()
    # Traducir las opciones de consola al objeto de configuración del modelo.
    config = ModelConfig(
        risk_free_rate=args.risk_free,
        equity_risk_premium=args.equity_risk_premium,
        wacc=args.wacc,
        terminal_growth=args.terminal_growth,
        forecast_years=4
    )
    # Ejecutar las capas en orden: obtener entradas, calcular y presentar.
    company = download_company_data(args.ticker)
    result = run_valuation(company, config)
    print_report(result)


if __name__ == "__main__":
    main()
