# Fear & Greed × ES/NQ backtest

Pre-registered, look-ahead-free research on whether the CNN Fear & Greed index gives an executable edge
for long ES futures positions (NQ used for validation).

- `RESEARCH_PROTOCOL.md`: hypotheses, splits, timing rules, costs, pass/fail gates (frozen once approved)
- `DATA.md`: data sources, known data issues, TradingView export checklist
- `src/fgbt/`: library code; `tests/`: `pytest`; `scripts/`: one script per research phase; `reports/`: generated output

```bash
pip install -r requirements.txt
python -m pytest
```
