
# 專案說明
# 113年自行研究計畫案--用於優化路檢聯稽績效之預測模型研究(以新竹所為例)
# 資料時間區間: 2022-10-01 00:00:00 ~ 2023-10-31 23:59:59
# 監理所站: 新竹所(50~54)
# 預測: 某時段某車種某地...(各項feature)的違規數量或通知臨檢數量


# 參考
# Datetime轉換: https://towardsdatascience.com/machine-learning-with-datetime-feature-engineering-predicting-healthcare-appointment-no-shows-5e4ca3a85f96


#=========================DATA PREPROCESSING=============================
#=========================DATA PREPROCESSING=============================

# Data cleaning
import pandas as pd
import os
from datetime import datetime
import csv

df_raw=pd.read_excel("113年自行研究計畫案_挑檔_raw.xlsx",sheet_name="raw")
# print(df_raw.info())
# 攔查總數: 24837
# 檢驗違規項目(違規項目->違規子項目=實際檢查的項目): 6981
# 檢驗通知臨檢項目(通知臨檢項目->通知臨檢子項目=實際檢查的項目): 3708
# 違規數: 3417
# 需通知臨檢數: 3708






# 去掉多餘的欄位
df_drop=df_raw.drop(["plate_no_main","exam_location_description","exam_staff_id","聯稽車種","檢驗違規項目名","檢驗違規子項目名","通知臨檢項目代號描述","通知臨檢子項目名稱"], axis=1)
# print(df_drop.info())


# 路稽時間轉換(排班的時間)
df_drop["exam_schedule_begin_time"]=pd.to_datetime(df_drop["exam_schedule_begin_time"], errors="coerce", format=f"%Y-%m-%d %H:%M:%S")
df_drop["exam_schedule_begin_year"]=df_drop["exam_schedule_begin_time"].dt.year
df_drop["exam_schedule_begin_month"]=df_drop["exam_schedule_begin_time"].dt.month
df_drop["exam_schedule_begin_day"]=df_drop["exam_schedule_begin_time"].dt.day
df_drop["exam_schedule_begin_hour"]=df_drop["exam_schedule_begin_time"].dt.hour
df_drop["exam_schedule_begin_minute"]=df_drop["exam_schedule_begin_time"].dt.minute


df_drop["exam_schedule_end_time"]=pd.to_datetime(df_drop["exam_schedule_end_time"], errors="coerce", format=f"%Y-%m-%d %H:%M:%S")
df_drop["exam_schedule_end_year"]=df_drop["exam_schedule_end_time"].dt.year
df_drop["exam_schedule_end_month"]=df_drop["exam_schedule_end_time"].dt.month
df_drop["exam_schedule_end_day"]=df_drop["exam_schedule_end_time"].dt.day
df_drop["exam_schedule_end_hour"]=df_drop["exam_schedule_end_time"].dt.hour
df_drop["exam_schedule_end_minute"]=df_drop["exam_schedule_end_time"].dt.minute

# Drop datetime columns(去掉年與raw datetime)
df_drop=df_drop.drop(["exam_schedule_begin_time","exam_schedule_end_time","exam_schedule_begin_year","exam_schedule_end_year"], axis=1)


# 填上null值(need_call_back)
df_drop["need_call_back"]=df_drop["need_call_back"].fillna(0)
# df_drop[["violation","need_call_back","rule_no","vil_source_rule","exam_item_no","exam_sub_class_no","exam_item_id","通檢子項目序號"]]=df_drop[["violation","need_call_back","rule_no","vil_source_rule","exam_item_no","exam_sub_class_no","exam_item_id","通檢子項目序號"]].fillna(0)
# # df_drop[["need_call_back"]]=df_drop[["need_call_back"]].fillna(0)
# df_drop[["violation"]]=df_drop[["violation"]].fillna(0)
# print(df_drop.info())


# 違規單數字轉成0/1; 0表示沒有違規, 1表示有違規
df_drop["violation"]=df_drop["violation"].apply(lambda x:1 if x>0 else 0)


# 將"類別"項feature做轉換
features_to_change=["exam_station_id","exam_location_id","rule_no","vil_source_rule","car_type","exam_item_no","exam_sub_class_no","exam_item_id","通檢子項目序號"]
df_dummies=pd.get_dummies(df_drop[features_to_change].astype("str"))
# print(df_dummies.info())


# 將dataframe合併
df_concat=pd.concat([df_drop.drop(features_to_change, axis=1), df_dummies], axis=1)
# print(df_concat)



# 衍伸欄位(違規數量、通知臨檢數量; 作為label項)
# 違規
vil_groupby_feature=[i for i in df_concat.columns if i not in ["violation"]] 
df_vil=df_concat.groupby(vil_groupby_feature).sum()  #用來作為違規數預測模型
df_vil=df_vil.reset_index(names=df_vil.index.names)  # MultiIndex轉成columns
# print(df_vil["violation"].describe())

# 通知臨檢
# callback_groupby_feature=[i for i in df_concat.columns if i not in ["need_call_back"]] 
# df_callback=df_concat.groupby(callback_groupby_feature).sum()  #用來作為違規數預測模型
# df_callback=df_callback.reset_index(names=df_callback.index.names)  # MultiIndex轉成columns
# print(df_callback["need_call_back"].describe())

# 呈現數據分布(histogram)
# print(df_callback["violation"].describe)
# from matplotlib import pyplot as plt
# # df_callback["need_call_back"].plot.hist()
# # plt.figure(figsize=(10,8))
# plt.figure().set_figwidth(10)
# _,_,bars=plt.hist(df_callback["need_call_back"],bins=35,edgecolor="black",align='left',color='#fac205')    # sky blue #75bbfd / goldenrod #fac205
# plt.bar_label(bars)
# plt.xticks([j for j in range(35)])

# plt.title("Histogram of callback")
# plt.xlabel("Callback count")
# plt.ylabel('Count')
# plt.show()
# print("Setp")

# Feature selection 評估哪些feature要用到model training
# Pearson's correlation
# 參考: https://chih-sheng-huang821.medium.com/%E7%B5%B1%E8%A8%88%E5%AD%B8-%E5%A4%A7%E5%AE%B6%E9%83%BD%E5%96%9C%E6%AD%A1%E5%95%8F%E7%9A%84%E7%B3%BB%E5%88%97-p%E5%80%BC%E6%98%AF%E4%BB%80%E9%BA%BC-2c03dbe8fddf
# https://pansci.asia/archives/115065
# https://study.com/academy/lesson/f-distribution-f-test-testing-hypothesis-definitions-example.html
# https://online.stat.psu.edu/stat501/lesson/6/6.2
# https://machinelearningmastery.com/feature-selection-for-regression-data/
# https://chih-sheng-huang821.medium.com/%E7%AC%A8%E8%9B%8B-%E5%95%8F%E9%A1%8C%E5%9C%A8data-2-%E5%BE%9E%E8%A8%8A%E6%81%AF%E7%90%86%E8%AB%96%E4%BE%86%E7%9C%8B%E8%B3%87%E6%96%99%E7%9A%84%E5%BD%B1%E9%9F%BF-9aa2e2b420c6
# ANOVA分析: https://belleaya.pixnet.net/blog/post/30754486
# F-statistic: https://courses.lumenlearning.com/introstats1/chapter/the-f-distribution-and-the-f-ratio/
# 從correlation coefficient 轉換到F-statistic (F-ratio): https://stats.stackexchange.com/questions/501342/f-regression-in-sklearn-how-is-a-correlation-converted-into-an-f-score

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression
# from sklearn.feature_selection import mutual_info_regression

# 留下前n個最高f-statistic/mutual-info的欄位
# SKB=SelectKBest(mutual_info_regression,k=6).fit(df_vil[[col for col in df_vil.columns if col != "violation"]],df_vil["violation"])
# look=sorted(SKB.scores_,reverse=True)
# # print(look)
# print(f">=0: {len(list(filter(lambda x:x>=0,look)))}, >0: {len(list(filter(lambda x:x>0,look)))}, >0.1: {len(list(filter(lambda x:x>0.1,look)))}")
# df_vil_X=SKB.transform(df_vil[[col for col in df_vil.columns if col != "violation"]])
# # print(type(df_vil_X))

# df_vil_X=pd.DataFrame(df_vil_X)
# df_vil=pd.concat([df_vil_X,df_vil["violation"]],axis=1)





# print(df_vil.index)
# df_vil=pd.concat([df_vil.index.to_frame(),df_vil["violation"]], axis=1)

# df_vil.reset_index()
# print(df_vil.index)
# df_test=df_vil.index.to_frame(index=False)
# df_test_violation=df_vil["violation"]
# print(df_vil.info())

# callback_feature=[i for i in df_concat.columns if i not in ["need_call_back"]]
# df_callback=df_concat.groupby(callback_feature).sum()
# df_callback=df_callback.reset_index(names=df_callback.index.names)
# print(df_callback.info())



# 隨機抽樣，當作測試集。剩下的當作訓練集
# 違規訓練與測試集
df_vil_test=df_vil.sample(frac=0.2,random_state=1)  # 用1當作random seed, for result's reproducibility
df_vil_train=df_vil.drop(index=df_vil_test.index, axis=0)
df_vil_test=df_vil_test.reset_index(drop=True)
df_vil_train=df_vil_train.reset_index(drop=True)



# Check dataframe has nan
# train_nan_num=df_vil_train.isna().sum().sum()
# print(f"train dataframe nan #: {train_nan_num}")

# df_vil_test=df_vil.sample(frac=0.2,random_state=1)  # 用1當作random seed, for result's reproducibility
# test_nan_num=df_vil_test.isna().sum().sum()
# print(f"test dataframe nan #: {test_nan_num}")


# print(df_vil_test.dtypes)
# print(f"df_vil_test.info(): {df_vil_test.info()}")


# 通知臨檢訓練與測試集
# df_callback_test=df_callback.sample(frac=0.2,random_state=1)  # 用1當作random seed, for result's reproducibility
# df_callback_train=df_callback.drop(index=df_callback_test.index, axis=0)
# df_callback_test.reset_index(drop=True)
# df_callback_train.reset_index(drop=True)
# df_vil_describe=df_vil.describe()
# print(df_vil.describe())
print("Data processing finished.")


# print(f"??: {df_vil_train.loc[7,'violation']}")
# print(f"df_callback_test.info(): {df_callback_test.info()}")

#======================DATA PREPROCESSING FINISHED======================
#======================DATA PREPROCESSING FINISHED======================





#============================MACHINE LEARNING=========================
#============================MACHINE LEARNING=========================

import torch 
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 違規預測
# 資料集包裝成DataLoader
# Custom dataset
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def label_to_tensor(val):
    return torch.tensor(val).float().to(device)

def feature_to_tensor(df:pd.DataFrame):
    # return nn.functional.normalize(torch.from_numpy(df.to_numpy()).float(),p=2,dim=0).to(device) normalize features
    return torch.from_numpy(df.to_numpy()).float().to(device)

class MVMDataset(Dataset):
    def __init__(self, data:pd.DataFrame, label_column, transform=None, target_transform=None):
        # Befor type changing
        # nan_num=data.isna().sum().sum()
        # print(f"Befor type changing nan #: {nan_num}")

        self.data=data.astype(float)
        self.label_column=label_column

        # nan_num=data.isna().sum().sum()
        # print(f"After type changing nan #: {nan_num}")

        # Normalize of each columns except the label column (violation in this case)
        # Make every features to have enough impact on the final result
        # This approach makes data contain lots of nan value!! -> Need modify
        # for column in self.data.columns:
        #     if column != label_column:
        #         # Apply min-max scaling -> 怪怪的? 會很依靠輸入批次的資料的均勻程度
        #         if (self.data[column].max()-self.data[column].min()) != 0:
        #             self.data[column]=(self.data[column]-self.data[column].min())/(self.data[column].max()-self.data[column].min())
        #         else:
        #             self.data[column]=1
                    

        self.transform=transform
        self.target_transform=target_transform


        # Check tensor has
        # nan_count=torch.isnan(torch.from_numpy(self.data.to_numpy())).sum()
        # print(f"Check self.data nan count: {nan_count}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label=self.data.loc[index, self.label_column]
        features=self.data.loc[index, self.data.columns != self.label_column]

        if self.transform:
            features=self.transform(features)

        if self.target_transform:
            label=self.target_transform(label)

        return features, label



print("MVMDataset definition completed.")


# Define hyperparameter for model training
input_dim= len(df_vil_train.columns)-1
output_dim=1
learning_rate=1e-3
epochs=30
batch_size=60
momentum=0.9


# DataLoader prepare
vil_training_dataset=MVMDataset(df_vil_train,label_column="violation", transform= feature_to_tensor, target_transform=label_to_tensor)
vil_training_dataloader=DataLoader(vil_training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
vil_test_dataset=MVMDataset(df_vil_test, "violation", feature_to_tensor, label_to_tensor)
vil_test_dataloader=DataLoader(vil_test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

# for idx, (features, violation) in enumerate(vil_training_dataset):
#     print(f"idx: {idx}, features: {features}, violation: {violation}")


# Test OK
# test_feature, test_num=next(iter(vil_training_dataloader))
# print(f"Vil features: {len(test_feature)}")
# print(f"Vil number: {len(test_num)}")



# Model definition
from torcheval.metrics import R2Score

# 1. Simple Linear Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim, output_dim)


    def forward(self, x):
        # print(f"weight shape: {self.linear.weight.shape}")
        # print(f"bias shape: {self.linear.bias.shape}")
        output=self.linear(x)
        return output


# 2. Polynomial regression model
class PolynomialRegression(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree=degree
        self.linear=nn.Linear(input_dim*degree,output_dim)

    def _polynomial_feature(self,x):
        return torch.cat([torch.pow(x,n) for n in range(1,self.degree+1)],1)
    
    def forward(self,x):
        return self.linear(self._polynomial_feature(x))

# 3. Linear ReLU stack (hidden dimension + stack number can modify)
class LinearReLUStack(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,stack_num):  # Linear-ReLU 堆疊次數
        super().__init__()
        self.start=nn.Linear(input_dim,hidden_dim)
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )
        self.end=nn.Linear(hidden_dim,output_dim)
        self.stack=nn.ModuleList([self.linear_relu_stack for _ in range(stack_num)])
        self.stack=self.stack.append(self.end)
        self.stack.insert(0,self.start)

    def forward(self,x):
        for module in self.stack:
            x=module(x)

        return x


# 4. Linear-X stack (hidden dimension + stack number modify)
class LinearXStack(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,stack_num):  
        super().__init__()
        self.start=nn.Linear(input_dim,hidden_dim)
        # self.start=nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size)  # batch size變成cin? 怪怪的

        # print(f"Test: {self.start.padding}")
        # Lout=((input_dim+2*self.start.padding[0]-self.start.dilation[0]*(kernel_size-1)-1)/self.start.stride[0])+1
        # Lout=int(Lout)
        # print(f"{Lout}")
        self.linear_x_stack=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
        self.end=nn.Linear(hidden_dim,output_dim)
        self.stack=nn.ModuleList([self.linear_x_stack for _ in range(stack_num)])
        self.stack=self.stack.append(self.end)
        self.stack.insert(0,self.start)

    def forward(self,x):
        for module in self.stack:
            x=module(x)

        return x


# 5. Poisson Regression model
class PoissonRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        # self.relu=nn.ReLU()
        
    def forward(self,x):
        return torch.poisson(torch.exp(self.linear(x)))


# 6. Negative binomial regression (NB(p) model)
# import numpy as np
alpha_1=torch.rand(1,requires_grad=True).to(device)
alpha_1_list=[]
# alpha=torch.rand(1,requires_grad=True).to(device)
alpha=torch.tensor(-0.3173,requires_grad=True)

def NB_pdf(y:torch.Tensor,rate:torch.Tensor,alpha_1:torch.Tensor):
    return torch.exp(torch.lgamma(y+alpha_1))/(torch.exp(torch.lgamma(y+1))*torch.exp(torch.lgamma(alpha_1)))*torch.pow((alpha_1/(alpha_1+rate)),alpha_1)*torch.pow((rate/(alpha_1+rate)),y)

def NB_NLL(rate,y):
    global alpha_1
    return -torch.sum(torch.log(NB_pdf(y,rate,alpha_1)))

def rejection_sampling(pdf,rate:torch.Tensor):
    # global alpha_1
    global alpha,alpha_1_list
    result=torch.empty(rate.shape).to(device)
    

    for idx in range(result.shape[0]):
        # print(f"mean: {mean[idx]}, sigma: {sigma[idx]}")
        # print(f"rate: {rate[idx]}")
        lock=0
        while True:
            lock+=1
            # sample_x=torch.normal(mean[idx],sigma[idx]).int().to(device)  # ?
            sample_x=torch.randint(0,50,(1,))  # uniform distribution from 0 to 49
            sample_y=torch.rand(1).to(device)
            # print(f"sample_x.shape: {sample_x.shape}")
            # print(f"sample_y.shape: {sample_y.shape}")
            # print(f"x: {sample_x}")
            # print(f"y: {sample_y}")

            threshold=pdf(sample_x,rate[idx],alpha)
            # print(f"threshold: {threshold}")
            # print(f"threshold.shape: {threshold.shape}")
            if sample_y <= threshold:
                result[idx]=sample_x
                break
            
            if lock>=1000:
                # print(f"mean: {mean[idx]}, sigma: {sigma[idx]}, threshold: {threshold}")
                # print(alpha_1_list)
                # file=pd.Series(alpha_1_list)
                # file.to_csv("alpha_change.csv",index=False)
                raise Exception("Infinite loop!")
    return result

class NegativeBinomial(nn.Module):
    def __init__(self,input_dim,output_dim): # p: var=mean + a*mean^p
        super().__init__()  
        self.linear=nn.Linear(input_dim,output_dim)
        # self.relu=nn.ReLU()
        # self.p=p
        # self.alpha_1=torch.rand(1,requires_grad=True)
        # self.mean=None

    def forward(self,x):
        # global alpha_1

        # mean=torch.exp(self.linear(x))
        # print(f"mean.shape: {mean.shape}")
        # sigma=torch.pow(self.relu(mean+torch.pow(alpha_1,-1)*torch.pow(mean,self.p)),0.5)    
        # print(f"sigma.shape: {sigma.shape}")

        # return rejection_sampling(NB_pdf,mean,sigma)  # 返回的應該要是一個rate! 非取樣的結果!
        return torch.exp(self.linear(x))


# 7. Generalized Poisson regression (GP-1)
def GP1_pdf(y:torch.Tensor,rate:torch.Tensor,alpha:torch.Tensor):
    return rate*torch.exp(-rate-alpha*y)*torch.pow(rate+alpha*y,y-1)/torch.exp(torch.lgamma(y+1))

def GP1_NLL(pred,y):
    global alpha
    return -torch.sum(torch.log(GP1_pdf(y,pred,alpha)))

class GeneralizedPoisson1(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        # self.relu=nn.ReLU()

    def forward(self,x):
        global alpha
        # lamb=torch.exp(self.linear(x))
        # mean=lamb/(1-alpha)
        # sigma=torch.pow(self.relu(lamb/torch.pow(1-alpha,3)),0.5)
        # return rejection_sampling(GP1_pdf,mean,sigma,lamb)
        return torch.exp(self.linear(x))


# 8. Zero-inflated Poisson Regression
class ZeroInflatedPoisson(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()


# Initialize model
# modelLinearRegression=LinearRegression(input_dim, output_dim).to(device)
# modelPolynomialRegression=PolynomialRegression(input_dim,output_dim,degree=5).to(device)
# modelLinearReLUStack=LinearReLUStack(input_dim,output_dim,hidden_dim=450,stack_num=4)
# modelLinearXStack=LinearXStack(input_dim,output_dim,450,stack_num=1)
# modelPoissonRegression=PoissonRegression(input_dim,output_dim).to(device)
# modelNegativeBinomial=NegativeBinomial(input_dim,output_dim)
modelGP1=GeneralizedPoisson1(input_dim,output_dim)


# Initialize loss function
# criterion=nn.MSELoss()
# criterion=nn.PoissonNLLLoss()
# criterion=NB_NLL
criterion=GP1_NLL

# Initialize optimizer
# optimizer=torch.optim.SGD(modelPolynomialRegression.parameters(), lr=learning_rate,momentum=momentum)
params=[it for it in modelGP1.parameters()]
# print(len(params))
# params.append(alpha_1)
params.append(alpha)
# print(len(params))
# def chain(iterable):
#     for it in iterable:
#         yield it

param_generator=(it for it in (params))        
# print(type(param_generator))
optimizer_adam=torch.optim.Adam(param_generator, lr=learning_rate)

# Define train loop and test loop methods
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size=len(dataloader.dataset)
    test_loss=0
    global data,alpha,alpha_1_list,alpha_1
    
    current=0

    # Set model to training mode
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        # print(len(X))
        # pred=model(X)
        rate=model(X)
        # print(f"Show the pred: {pred}")
        # for i in range(len(y)):
        #     print(f"pred[{i}]: {pred[i]}, y[{i}]: {y[i]}")
        # print(f"pred.dtype: {pred.dtype}")
        # print(f"y.dtype: {y.dtype}")
        y=y.reshape(-1,1)
        loss=loss_fn(rate, y)
        test_loss+=loss.item()
        # print(f"loss.dtype: {loss.dtype}")



        # Check parameters
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Batch {batch}, Name: {name}, Parameter: {param.data}")
                # 第一批做gradient就變成nan? -> OK! 因為normalize input導致nan產生。已解決



        # Back propagation
        loss.backward()

        # Check grad value
        # print(f"Parameter gradient: weight: {model.linear.weight.grad}")
        # print(f"Parameter gradient: bias: {model.linear.bias.grad}")
        optimizer.step()
        optimizer.zero_grad()


        r2score=R2Score()
        # r2score.update(pred,y)
        # mean=rate
        # mean=rate/(1-alpha)
        # sigma=torch.pow(mean+torch.pow(alpha_1,-1)*torch.pow(mean,2),0.5)
        # sigma=torch.pow(rate/torch.pow(1-alpha,3),0.5)
        pred=rejection_sampling(GP1_pdf,rate)
        r2score.update(pred,y)

        r2score_val=r2score.compute().item()


        if input_dim < batch_size-1:
            r2score_adj=R2Score(num_regressors=input_dim)
            r2score_adj.update(pred,y)
            r2score_adj_val=r2score_adj.compute().item()
        else:
            r2score_adj_val=None
        
        if True:  #batch%10==0:
            loss, processed=loss.item(), len(X)
            current+=processed
            print(f"loss: {loss:>7f}, R2-squared: {r2score_val}, Adjusted R2-squared: {r2score_adj_val}, [{current:>5d}/{size:>5d}]")
            print(f"alpha: {alpha}")
            # print(f"lambda: {}")
            alpha_1_list.append(alpha.item())


            
            data.append(["Train",epoch,batch,current,size,loss,r2score_val,r2score_adj_val])

    test_loss/=len(dataloader)
    print(f"Avg loss of train data: {test_loss:>8f}")


def test_loop(dataloader, model, loss_fn,epoch):
    global data,alpha_1,alpha
    # global data

    model.eval()
    # size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss=0
    current=0
    size=len(dataloader.dataset)


    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            # pred=model(X)
            rate=model(X)
            # print(rate)

            y=y.reshape(-1,1)
            loss=loss_fn(rate,y).item()
            test_loss+=loss
            current+=len(X)

            r2score=R2Score()
            # r2score.update(pred,y)
            
            # mean=rate
            # mean=rate/(1-alpha)
            # sigma=torch.pow(mean+torch.pow(alpha_1,-1)*torch.pow(mean,2),0.5)
            # sigma=torch.pow(rate/torch.pow(1-alpha,3),0.5)
            pred=rejection_sampling(GP1_pdf,rate)
            # print(pred)

            r2score.update(pred,y)
            r2score_val=r2score.compute().item()

            if input_dim < batch_size-1:
                r2score_adj=R2Score(num_regressors=input_dim)
                r2score_adj.update(pred,y)
                r2score_adj_val=r2score_adj.compute().item()
            else:
                r2score_adj_val=None

            data.append(["Test",epoch,batch,current,size,loss,r2score_val,r2score_adj_val])
            print(f"loss: {loss:>7f}, R2-squared: {r2score_val}, Adjusted R2-squared: {r2score_adj_val}, [{current:>5d}/{size:>5d}]")
            # print(f"Epoch {epoch} weight: {model.linear.weight}")
            # print(f"Epoch {epoch} bias: {model.linear.bias}")

    test_loss/=num_batches
    print(f"Avg loss of test data: {test_loss:>8f}")
    






# Export training and testing results
path = "D:\\Python workspace\\PyTorch\\"+modelGP1.__class__.__name__
if not os.path.isdir(path):
    os.mkdir(path)
    

# with open(os.path.join(path,datetime.now().strftime(r"%Y-%m-%d %H_%M_%S")+"_log.csv"),'w',encoding="utf8", newline="") as f:
with pd.ExcelWriter(os.path.join(path,datetime.now().strftime(r"%Y-%m-%d %H_%M_%S")+"_log.xlsx")) as writer:
    # test=["This","is","a","test"]
    # test2=["Hello","~~"]
    # writer=csv.writer(f)
    # writer.writerow(test)
    # writer.writerow(test2)

    data_info=[]
    data=[]
    data_column=["Type","Epoch","Batch","Accumulation","Total","Loss","R-squared score","Adjusted R-squared score"]

    data_info.append(["Datetime",datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")])
    data_info.append(["Train dataset","df_vil_train"])
    data_info.append(["Test dataset","df_vil_test"])
    data_info.append(["Model",modelGP1])
    data_info.append(["Loss function",criterion])
    data_info.append(["Optimizer",optimizer_adam])
    data_info.append(["Epoch number",epochs])
    data_info.append(["Learning rate",learning_rate])
    data_info.append(["Batch size",batch_size])
    # data_info.append(["Type","Epoch","Batch","Accumulation","Total","Loss","R-squared score","Adjusted R-squared score"])
    pd.DataFrame(data=data_info).to_excel(writer,sheet_name="info", header=None, index=False)


    # Start training model
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------")
        train_loop(vil_training_dataloader, modelGP1, criterion, optimizer_adam,t+1)
        test_loop(vil_test_dataloader, modelGP1, criterion,t+1)

    
    pd.DataFrame(data=data, columns=data_column).to_excel(writer, sheet_name="result", index=False)
    # writer.writerows(data)


print("Violation Regression Done!")


# Result
# Simple Linear Regression Model: loss值都是nan! (batch size=10) Why? Underdetermined system?
# Reference: https://www.collimator.ai/reference-guides/what-is-an-underdetermined-system
# Chatgpt advice:
# A NaN loss value in PyTorch linear regression model training can be caused by several reasons. One of the most common reasons is that the values in the dataset are too large or too small. This can cause the gradients to become too large or too small, which can lead to numerical instability and NaN values.
# To fix this issue, you can try normalizing the input data. You can do this by subtracting the mean and dividing by the standard deviation of the input data. Here is an example of how you can normalize the input data:
# Another reason for NaN loss values could be due to the batch size not matching the total number of inputs. You can use a batch size that is a factor of the total number of inputs. If you cannot use a batch size that is a factor of the total number of inputs, you can use the drop_last parameter in the DataLoader class to drop the last batch if it is smaller than the specified batch size.
# Information gain: https://machinelearningmastery.com/information-gain-and-mutual-information/
# Activation function選擇: 選gradient比較大的! https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
# Perceptron: 前端的神經元如何將輸入做總和(線性相加or其他, ex: convolution): https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-2%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E6%84%9F%E7%9F%A5%E5%99%A8-perceptron-%E4%BB%8B%E7%B4%B9-84d8b809f866

# 評估regression model的好壞
# R squared, Adjusted R squared, loss
# 參考: https://medium.com/qiubingcheng/%E5%9B%9E%E6%AD%B8%E5%88%86%E6%9E%90-regression-analysis-%E7%9A%84r%E5%B9%B3%E6%96%B9-r-squared-%E8%88%87%E8%AA%BF%E6%95%B4%E5%BE%8Cr%E5%B9%B3%E6%96%B9-adjusted-r-squared-f38ad733bc4e


# 其他模型
# Deep Neural Network: https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
# Zero Inflated Poisson Regression Model: https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
# Poission Regression Model: https://timeseriesreasoning.com/contents/poisson-regression-model/
# Negative binomial regression: https://timeseriesreasoning.com/contents/negative-binomial-regression-model/
# Negative binomial regression: https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf
# Negative binomial regression: https://github.com/meaneych/PyTorch_CountDistribution_Examples/blob/main/NegBin1_PyTorch_AutoGrad.ipynb
# Negative binomial regression: https://www.karlin.mff.cuni.cz/~pesta/NMFM404/NB.html

# GP1 model: https://www.sciencedirect.com/science/article/pii/S2213398421000828