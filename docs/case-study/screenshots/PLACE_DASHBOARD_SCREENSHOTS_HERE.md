# Dashboard screenshots

Drop the following screenshots into this directory after running
`./scripts/run_dashboard.sh`:

- `dashboard-overview.png` — full-page screenshot showing the Cost Reduction headline
- `dashboard-routing.png` — close-up of the routing distribution donut
- `dashboard-verdicts.png` — close-up of the quality verdicts bar chart
- `dashboard-recent.png` — close-up of the recent requests table

The case study (`docs/case-study/CASE_STUDY.md`) and the top of the
project README already reference these exact filenames — drop them in
and they render automatically on GitHub.

To produce the data first:

```bash
uv run python scripts/simulate_load.py -n 500
./scripts/run_dashboard.sh
# then take screenshots from http://localhost:8501
```
