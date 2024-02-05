from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

df_result=pd.read_excel("GeneralizedPoisson1\\2024-02-05 15_55_44_log_GP1.xlsx", sheet_name="result")
df_result2=pd.read_excel("GeneralizedPoisson1\\2024-02-05 16_06_48_log_GP1-1.xlsx", sheet_name="result")
df_result3=pd.read_excel("GeneralizedPoisson1\\2024-02-05 16_16_09_log_GP1-2.xlsx", sheet_name="result")

# df_result4=pd.read_excel("LinearXStack\\2023-12-05 16_48_22_log_LinearTanh_hidden450.xlsx", sheet_name="result")
# df_result5=pd.read_excel("LinearXStack\\2023-12-05 17_06_59_log_LinearTanh_hidden450-1.xlsx", sheet_name="result")
# df_result6=pd.read_excel("LinearXStack\\2023-12-06 08_57_21_log_LinearTanh_hidden450-2.xlsx", sheet_name="result")

# df_result7=pd.read_excel("LinearXStack\\2023-12-04 16_41_49_log_LinearXStack_HddenDim450_Stack4.xlsx", sheet_name="result")
# df_result8=pd.read_excel("LinearXStack\\2023-12-05 09_44_33_log_LinearXStack_HddenDim450_Stack4-1.xlsx", sheet_name="result")
# df_result9=pd.read_excel("LinearXStack\\2023-12-05 11_26_50_log_LinearXStack_HddenDim450_Stack4-2.xlsx", sheet_name="result")

# df_result10=pd.read_excel("LinearRegression\\2023-11-29 17_25_41_log_LinearRegression_feature_selection6_ADAM_Epoch30_lr1-3_batch60.xlsx", sheet_name="result")
# df_result11=pd.read_excel("LinearRegression\\2023-11-29 17_28_46_log_LinearRegression_feature_selection6_ADAM_Epoch30_lr1-3_batch60-1.xlsx", sheet_name="result")
# df_result12=pd.read_excel("LinearRegression\\2023-11-29 17_32_24_log_LinearRegression_feature_selection6_ADAM_Epoch30_lr1-3_batch60-2.xlsx", sheet_name="result")
# print(df_result.info())


df_result_train=df_result[df_result["Type"]=="Train"]
df_result2_train=df_result2[df_result2["Type"]=="Train"]
df_result3_train=df_result3[df_result3["Type"]=="Train"]
df_result_test=df_result[df_result["Type"]=="Test"]
df_result2_test=df_result2[df_result2["Type"]=="Test"]
df_result3_test=df_result3[df_result3["Type"]=="Test"]

# df_result4_train=df_result4[df_result4["Type"]=="Train"]
# df_result5_train=df_result5[df_result5["Type"]=="Train"]
# df_result6_train=df_result6[df_result6["Type"]=="Train"]
# df_result4_test=df_result4[df_result4["Type"]=="Test"]
# df_result5_test=df_result5[df_result5["Type"]=="Test"]
# df_result6_test=df_result6[df_result6["Type"]=="Test"]

# df_result7_train=df_result7[df_result7["Type"]=="Train"]
# df_result8_train=df_result8[df_result8["Type"]=="Train"]
# df_result9_train=df_result9[df_result9["Type"]=="Train"]
# df_result7_test=df_result7[df_result7["Type"]=="Test"]
# df_result8_test=df_result8[df_result8["Type"]=="Test"]
# df_result9_test=df_result9[df_result9["Type"]=="Test"]

# df_result10_train=df_result10[df_result10["Type"]=="Train"]
# df_result11_train=df_result11[df_result11["Type"]=="Train"]
# df_result12_train=df_result12[df_result12["Type"]=="Train"]
# df_result10_test=df_result10[df_result10["Type"]=="Test"]
# df_result11_test=df_result11[df_result11["Type"]=="Test"]
# df_result12_test=df_result12[df_result12["Type"]=="Test"]
print("Get train data")


# 盒狀圖
# sns.boxplot(x="Epoch",y="Loss",data=df_result_train)
# plt.show()

# avg_loss=[]
# for epoch in range(1,3):
#     df_temp=df_result_train[df_result_train["Epoch"]==epoch]
#     avg_loss.append(df_temp["Loss"].mean())


# 折線圖
# df_result_train_avg=df_result_train.groupby(["Type","Epoch"]).mean()
# # df_result2_train_avg=df_result2_train.groupby(["Type","Epoch"]).mean()
# sns.lineplot(x="Epoch", y="Loss",data=df_result_train_avg,marker='o',markersize=10,label="Epoch 30").set(title="Linear Regression Epoch 60")
# # sns.lineplot(x="Epoch", y="Loss",data=df_result2_train_avg,marker='o',markersize=10)
# plt.xlim(0,31)
# plt.legend(loc="lower right")
# plt.show()


# 多次實驗合併平均折線圖
df_concat=pd.concat([df_result_train,df_result2_train,df_result2_train],axis=0)
df_concat_avg=df_concat.groupby(["Type","Epoch"]).mean()
df_concat_test=pd.concat([df_result_test,df_result2_test,df_result3_test],axis=0)
df_concat_test_avg=df_concat_test.groupby(["Type","Epoch"]).mean()

# df_concat2=pd.concat([df_result4_train,df_result5_train,df_result6_train],axis=0)
# df_concat2_avg=df_concat2.groupby(["Type","Epoch"]).mean()
# df_concat2_test=pd.concat([df_result4_test,df_result5_test,df_result6_test],axis=0)
# df_concat2_test_avg=df_concat2_test.groupby(["Type","Epoch"]).mean()

# df_concat3=pd.concat([df_result7_train,df_result8_train,df_result9_train],axis=0)
# df_concat3_avg=df_concat3.groupby(["Type","Epoch"]).mean()
# df_concat3_test=pd.concat([df_result7_test,df_result8_test,df_result9_test],axis=0)
# df_concat3_test_avg=df_concat3_test.groupby(["Type","Epoch"]).mean()

# df_concat4=pd.concat([df_result10_train,df_result11_train,df_result12_train],axis=0)
# df_concat4_avg=df_concat4.groupby(["Type","Epoch"]).mean()
# df_concat4_test=pd.concat([df_result10_test,df_result11_test,df_result12_test],axis=0)
# df_concat4_test_avg=df_concat4_test.groupby(["Type","Epoch"]).mean()
print("Concat OK")

sns.set_style("whitegrid",{"grid.linestyle":"--"})
sns.lineplot(x="Epoch",y="R-squared score",data=df_concat_avg,marker='o',markersize=10,label="GP-1 (Train)",color="blue")
sns.lineplot(x="Epoch",y="R-squared score",data=df_concat_test_avg,marker='*',markersize=10,label="GP-1 (Validation)",linestyle='--',color="blue")
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat2_avg,marker='o',markersize=10,label="Tanh 450 (Train)",color="#f97306" )
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat2_test_avg,marker='*',markersize=10,label="Tanh 450 (Validation)",linestyle='--',color="#f97306")
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat3_avg,marker='o',markersize=10,label="Stack 4 (Train)",color="#3f9b0b" )
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat3_test_avg,marker='*',markersize=10,label="Stack 4 (Validation)",linestyle='--',color="#3f9b0b")
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat4_avg,marker='o',markersize=10,label="Dim 6 (Train)",color="#b5485d" )
# sns.lineplot(x="Epoch",y="R-squared score",data=df_concat4_test_avg,marker='*',markersize=10,label="Dim 6 (Validation)",linestyle='--',color="#b5485d")

plt.xlim(0,31)
plt.legend(loc="lower right")
plt.title("GP-1 Regression",fontsize=18)
plt.show()

print(f"df_concat_avg : {df_concat_avg.loc[('Train',30),'R-squared score']:.7f}")
print(f"df_concat_avg  (validation): {df_concat_test_avg.loc[('Test',30),'R-squared score']:.7f}")
# print(f"df_concat_avg Poisson {df_concat2_avg.loc[('Train',30),'R-squared score']:.7f}")
# print(f"df_concat_avg Poisson (validation): {df_concat2_test_avg.loc[('Test',30),'R-squared score']:.7f}")
# print(f"df_concat_avg stack 4: {df_concat3_avg.loc[('Train',30),'R-squared score']:.7f}")
# print(f"df_concat_avg stack 4 (validation): {df_concat3_test_avg.loc[('Test',30),'R-squared score']:.7f}")
# print(f"df_concat_avg 6: {df_concat4_avg.loc[('Train',30),'R-squared score']:.7f}")
# print(f"df_concat_avg 6 (validation): {df_concat4_test_avg.loc[('Test',30),'R-squared score']:.7f}")

# R-squared score
# https://www.biomooc.com/color/seabornColors.html
# https://www.baeldung.com/cs/training-validation-loss-deep-learning