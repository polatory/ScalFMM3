import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

# csv format :
# groupsize,legende,time,type
# 100,farfield,3.66,experiment1
# 100,nearfield,3.66,experiment1
# 200,farfield,3.66,experiment1
# 200,nearfield,3.66,experiment1
# 300,farfield,3.66,experiment1
# 300,nearfield,3.66,experiment1
# 400,farfield,3.66,experiment1
# 400,nearfield,3.66,experiment1
# 100,farfield,3.66,experiment2
# 100,nearfield,3.66,experiment2
# 200,farfield,3.66,experiment2
# 200,nearfield,3.66,experiment2
# 300,farfield,3.66,experiment2
# 300,nearfield,3.66,experiment2
# 400,farfield,3.66,experiment2
# 400,nearfield,3.66,experiment2
#
seq = pd.read_csv('sphere.csv')

g = sns.relplot(
    data=seq,
    x="groupsize", y="time",
    hue="legende", 
    #col="dist",
    size="type",
    kind="line",
    palette='viridis',
    size_order=["views","normal"],
    height=5, aspect=.75, facet_kws=dict(sharex=False)
)
g.set_axis_labels("group size", "time in seconds")
g.legend.set_title("")
plt.title('Timings title')
#plt.yscale('log')
plt.show()
g.savefig('timings.pdf')
