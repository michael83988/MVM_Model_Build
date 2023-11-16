
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


# 路稽時間轉換
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
vil_groupby_feature=[i for i in df_concat.columns if i not in ["violation"]] 
# print(len(vil_groupby_feature))
df_vil=df_concat.groupby(vil_groupby_feature).sum()  #用來作為違規數預測模型
df_vil=df_vil.reset_index(names=df_vil.index.names)
# print(df_vil.index)
# df_vil=pd.concat([df_vil.index.to_frame(),df_vil["violation"]], axis=1)

# df_vil.reset_index()
# print(df_vil.index)
# df_test=df_vil.index.to_frame(index=False)
# df_test_violation=df_vil["violation"]
# print(df_vil.info())

callback_feature=[i for i in df_concat.columns if i not in ["need_call_back"]]
df_callback=df_concat.groupby(callback_feature).sum()
df_callback=df_callback.reset_index(names=df_callback.index.names)
# print(df_callback.info())



# 隨機抽樣，當作測試集。剩下的當作訓練集
# 違規訓練集
df_vil_test=df_vil.sample(frac=0.2,random_state=1)  # 用1當作random seed, for result's reproducibility
df_vil_train=df_vil.drop(index=df_vil_test.index, axis=0)
df_vil_test=df_vil_test.reset_index(drop=True)
df_vil_train=df_vil_train.reset_index(drop=True)
# print(df_vil_test.dtypes)
# print(f"df_vil_test.info(): {df_vil_test.info()}")


# 通知臨檢訓練集
df_callback_test=df_callback.sample(frac=0.2,random_state=1)  # 用1當作random seed, for result's reproducibility
df_callback_train=df_callback.drop(index=df_callback_test.index)
print("Data processing finished.")


print(f"??: {df_vil_train.loc[7,'violation']}")
# print(f"df_callback_test.info(): {df_callback_test.info()}")

#======================DATA PREPROCESSING FINISHED======================
#======================DATA PREPROCESSING FINISHED======================


#============================MACHINE LEARNING=========================
#============================MACHINE LEARNING=========================

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 違規預測
# 資料集包裝成DataLoader

def label_to_tensor(val):
    return torch.tensor(val)  #.float()

def feature_to_tensor(df:pd.DataFrame):
    return torch.from_numpy(df.to_numpy()).float()

class VilDataset(Dataset):
    def __init__(self, data:pd.DataFrame, transform=None, target_transform=None):
        self.data=data.astype(float)

        # Normalize of each columns except the label column (violation in this case)
        # Make every features to have enough impact on the final result
        for column in self.data.columns:
            if column != "violation":
                # Apply min-max scaling
                self.data[column]=(self.data[column]-self.data[column].min())/(self.data[column].max()-self.data[column].min())

        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        label=self.data.loc[index, "violation"]
        features=self.data.loc[index, self.data.columns != "violation"]

        if self.transform:
            features=self.transform(features)

        if self.target_transform:
            label=self.target_transform(label)

        return features, label



print("VilDataset completed.")
vil_training_dataset=VilDataset(df_vil_train,transform= feature_to_tensor, target_transform=label_to_tensor)
vil_training_dataloader=DataLoader(vil_training_dataset, batch_size=20)

# for idx, (features, violation) in enumerate(vil_training_dataset):
#     print(f"idx: {idx}, features: {features}, violation: {violation}")


test_feature, test_num=next(iter(vil_training_dataloader))
print(f"Vil features: {test_feature}")
print(f"Vil number: {test_num}")

















# # 將dataframe中非數字的feature換成0 ~ n的資料 -> 之後轉成one-hot encoded tensor所需
# # 管轄所站
# dmv_dict=dict()

# # 路檢聯稽地點
# location_dict=dict()

# # 違規條款
# rule_dict=dict()

# # 違規條款來源
# source_dict=dict()

# # 聯稽車種
# car_dict=dict()

# # 違規檢查項
# exam_item_no_dict=dict()

# # 違規檢查子項
# exam_sub_class_no_dict=dict()

# # 通知臨檢項
# exam_item_id_dict=dict()

# # 通知臨檢子項
# exam_sub_item_id_dict=dict()



# for idx in df_drop.index:
#     dmv_dict[df_drop.loc[idx, "exam_station_id"]]= dmv_dict.get(df_drop.loc[idx, "exam_station_id"], len(dmv_dict))
#     location_dict[df_drop.loc[idx, "exam_location_id"]]= location_dict.get(df_drop.loc[idx, "exam_location_id"], len(location_dict))
#     rule_dict[df_drop.loc[idx, "rule_no"]]= rule_dict.get(df_drop.loc[idx, "rule_no"], len(rule_dict))
#     source_dict[df_drop.loc[idx, "vil_source_rule"]]= source_dict.get(df_drop.loc[idx, "vil_source_rule"], len(source_dict))
#     car_dict[df_drop.loc[idx, "car_type"]]= car_dict.get(df_drop.loc[idx, "car_type"], len(car_dict))
#     exam_item_no_dict[df_drop.loc[idx, "exam_item_no"]]= exam_item_no_dict.get(df_drop.loc[idx, "exam_item_no"], len(exam_item_no_dict))
#     exam_sub_class_no_dict[df_drop.loc[idx, "exam_sub_class_no"]]= exam_sub_class_no_dict.get(df_drop.loc[idx, "exam_sub_class_no"], len(exam_sub_class_no_dict))
#     exam_item_id_dict[df_drop.loc[idx, "exam_item_id"]]= exam_item_id_dict.get(df_drop.loc[idx, "exam_item_id"], len(exam_item_id_dict))
#     exam_sub_item_id_dict[df_drop.loc[idx, "通檢子項目序號"]]= exam_sub_item_id_dict.get(df_drop.loc[idx, "通檢子項目序號"], len(exam_sub_item_id_dict))


    # 是否違規項: 0 & 1, 表示有無違規, 0=沒有, 1=有



# Replace value through dictionary
# df_drop[["exam_station_id"]]=df_drop[["exam_station_id"]].replace(dmv_dict)
# df_drop[["exam_location_id"]]=df_drop[["exam_location_id"]].replace(location_dict)
# df_drop[["rule_no"]]=df_drop[["rule_no"]].replace(rule_dict)
# df_drop[["vil_source_rule"]]=df_drop[["vil_source_rule"]].replace(source_dict)
# df_drop[["car_type"]]=df_drop[["car_type"]].replace(car_dict)
# df_drop[["exam_item_no"]]=df_drop[["exam_item_no"]].replace(exam_item_no_dict)
# df_drop[["exam_sub_class_no"]]=df_drop[["exam_sub_class_no"]].replace(exam_sub_class_no_dict)
# df_drop[["exam_item_id"]]=df_drop[["exam_item_id"]].replace(exam_item_id_dict)
# df_drop[["通檢子項目序號"]]=df_drop[["通檢子項目序號"]].replace(exam_sub_item_id_dict)
# df_drop[["violation"]]=df_drop[["violation"]].apply(lambda x:1 if x.item()>0 else 0)  #?

# print(df_drop.info())
# print(df_drop.size())