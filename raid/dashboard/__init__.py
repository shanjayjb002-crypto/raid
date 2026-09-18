"""Streamlit dashboard tying every RAID phase together.

The analysis itself lives in :mod:`raid.dashboard.pipeline` and imports no
Streamlit, so it is unit tested without a browser. :mod:`raid.dashboard.app` is
the rendering layer only.

Launch with::

    streamlit run raid/dashboard/app.py
"""

from .pipeline import Analysis, AnalysisOptions, analyse, rows_to_csv

__all__ = ["Analysis", "AnalysisOptions", "analyse", "rows_to_csv"]
