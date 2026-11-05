"""Overall statement of PROTOCOL_v2.md section 3 across the 8 datasets (4 regression + 4 classification), per backbone, split type and contrast.
Why this add-on exists: the frozen conformal_summary.py (digest in the protocol) tallies regression and classification separately, so its 'overall_statement' column applies
'at least 6 of 8' to at most 4 datasets and can never be met (journal note on the full run). The protocol text says 6 of 8 interpretable datasets over all eight datasets, so this
script only adds the two frozen tallies (summary_tally_reg.csv, summary_tally_cls.csv) and applies the rule as written. Nothing is recomputed from predictions.
  python conformal_overall.py [DIR]    default results_v2"""
import sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results_v2'
t = pd.concat([pd.read_csv(D / 'summary_tally_reg.csv'), pd.read_csv(D / 'summary_tally_cls.csv')], ignore_index=True)
num = ['datasets', 'narrower_all', 'wider_all', 'mixed_all', 'interpretable', 'narrower_interp', 'wider_interp']
o = t.groupby(['contrast', 'base', 'stype'], as_index=False)[num].sum()
o['overall_statement'] = ['narrower (>=6 of 8 interpretable)' if n >= 6 else 'wider (>=6 of 8 interpretable)' if w >= 6 else 'no overall statement' for n, w in zip(o.narrower_interp, o.wider_interp)]
o.to_csv(D / 'summary_overall_8datasets.csv', index=False)
pd.set_option('display.width', 250); print(o.to_string(index=False))
