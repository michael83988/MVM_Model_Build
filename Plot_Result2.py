import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

df_alpha_1=pd.read_csv("alpha_1_change.csv")

print(df_alpha_1.info())

plt.figure(figsize=(7,6))
sns.lineplot(data=df_alpha_1,x=[i for i in range(len(df_alpha_1))],y="0",label="alpha^(-1)")
plt.plot([len(df_alpha_1)-1],[df_alpha_1.iloc[-1,0]],marker='o', markersize=10,color='red')
plt.annotate(f"({len(df_alpha_1)-1},{df_alpha_1.iloc[-1,0]:.7f})",(len(df_alpha_1)-1,df_alpha_1.iloc[-1,0]),(-10,20),textcoords="offset points",ha="center",fontsize=16)
plt.xticks([i for i in range(0,len(df_alpha_1),20)],rotation=45)
plt.grid(linestyle="dashed")
plt.legend(loc='upper right')
plt.ylabel("alpha^(-1)")
plt.xlabel("Batch")
plt.title("alpla^(-1) change with batches")
plt.show()