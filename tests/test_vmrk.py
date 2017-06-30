import subprocess
import pandas as pd
from pandas.util.testing import assert_frame_equal

subprocess.run(["python", "../vmrk.py"])

a = []
for n in "results.csv", "expected_results.csv":
    d = pd.read_csv(n)
    a.append(d)

obs = a[0]
exp = a[1]

assert_frame_equal(obs, exp)
