from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

df_result=pd.read_excel("LinearRegression\\2023-11-24 15_13_34_log.xlsx", sheet_name="result")
df_result2=pd.read_excel("LinearRegression\\2023-11-24 15_10_15_log.xlsx", sheet_name="result")
# print(df_result.info())


df_result_train=df_result[df_result["Type"]=="Train"]
df_result2_train=df_result2[df_result["Type"]=="Train"]
print("Get train data")

# sns.boxplot(x="Epoch",y="Loss",data=df_result_train)
# plt.show()

# avg_loss=[]
# for epoch in range(1,3):
#     df_temp=df_result_train[df_result_train["Epoch"]==epoch]
#     avg_loss.append(df_temp["Loss"].mean())

df_result_train_avg=df_result_train.groupby(["Type","Epoch"]).mean()
df_result2_train_avg=df_result2_train.groupby(["Type","Epoch"]).mean()
sns.lineplot(x="Epoch", y="Loss",data=df_result_train_avg,marker='o',markersize=10).set(title="Linear Regression with different time")
sns.lineplot(x="Epoch", y="Loss",data=df_result2_train_avg,marker='o',markersize=10)
plt.show()