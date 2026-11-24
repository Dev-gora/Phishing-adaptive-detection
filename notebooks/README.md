# Notebooks

The pipeline is deliberately implemented as importable modules plus a staged CLI
(`main.py`), so that every number in the paper can be regenerated from a terminal
and reviewed as a diff — which is not possible with notebook-only code.

This folder is kept for exploratory work. If you want to inspect results
interactively, start Jupyter from the **project root** (not from this folder) so
that the `src` package is importable:

```bash
jupyter lab
```

then in a notebook:

```python
import pandas as pd
from src.config import METRICS_DIR, RANKINGS_DIR

pd.read_csv(METRICS_DIR / "final_test_metrics.csv").head()
pd.read_csv(RANKINGS_DIR / "feature_rankings.csv").head()
```

Anything worth keeping should end up in `src/` and be reachable from `main.py`;
exploratory notebooks are not part of the reported results.
