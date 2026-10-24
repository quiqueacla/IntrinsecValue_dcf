# Valoración intrínseca por DCF

Programa Python para estimar el valor intrínseco de una acción mediante un modelo FCFF/DCF con tres escenarios. Descarga datos anuales y estimaciones de ingresos con `yfinance`.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Desde esta carpeta:

```bash
python main.py REP.MC
python main.py IDR.MC --risk-free 0.029 --equity-risk-premium 0.07 --wacc 0.09 --terminal-growth 0.02
```

También se puede importar el paquete desde otro programa o notebook:

```python
from dcf_valuation import download_company_data, run_valuation

company = download_company_data("REP.MC")
result = run_valuation(company)
print(result.summary)
```

Los valores por defecto de Rf, prima de riesgo, WACC y crecimiento terminal son hipótesis de ejemplo, no datos de mercado. Revísalos antes de usar el resultado. `yfinance` depende de los datos y nombres de partidas que Yahoo Finance publique para cada empresa; algunas compañías pueden no tener todas las filas o estimaciones disponibles.

## Módulos

- `data.py`: descarga y normaliza estados financieros, datos generales y estimaciones.
- `model.py`: construye FCFF, proyecciones y valoración por escenarios.
- `report.py`: imprime tablas y resumen.
- `main.py`: punto de entrada de línea de comandos.

## Convenciones del modelo

- Los gastos de capital se tratan como salida de caja positiva en el modelo; `yfinance` suele presentarlos como importe negativo, por lo que se normalizan al descargar.
- Se aproxima FCFF como `CFO - CapEx + interés × (1 - tipo impositivo)`. El valor de CapEx de esta fórmula es positivo.
- Las cinco proyecciones (año actual estimado y cuatro años siguientes) se descuentan como periodos 1 a 5. El valor terminal se calcula al final del quinto periodo con Gordon-Shapiro.
- Se usa el WACC indicado (9 % por defecto). El programa calcula también un WACC ponderado orientativo, pero no lo sustituye automáticamente.
- Para pasar del valor de empresa a valor del capital se resta la deuda neta (deuda total menos caja) cuando Yahoo Finance proporciona ambos datos. Si falta alguno, se informa y se supone deuda neta cero; el resultado pierde precisión.
