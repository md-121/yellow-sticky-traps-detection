import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

array = [[1558, 0, 0, 1719],
         [0, 418, 0, 432],
         [0, 0, 197, 198],
         [1856, 562, 251, 0]]

df_cm = pd.DataFrame(array, ["WF", "MR", "NC", "Total"], ["WF", "MR", "NC", "Total"]).rename_axis("Actual")
df_cm = df_cm.rename_axis("Prediction", axis=1)
sn.set(font_scale=1.4)  # for label size
ax = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='d', annot_kws={"size": 16})  # font size
ax.tick_params(length=0)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()
