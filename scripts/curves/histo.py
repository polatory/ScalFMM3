import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

# csv format :
# morton,particles
# 1,18
# 2,21
# ....
seq = pd.read_csv('distleaves.csv')

f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)

sns.histplot(
    data=seq,
    x='morton',
    y='particles',
    bins=50,
    color='#283764',
    edgecolor='.3',
    linewidth=.5,
)
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.title('Histo title')
plt.show()
ax.figure.savefig('histo.pdf')
