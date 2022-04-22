import pandas as pd
import csv

# def sort_target(x): # 1번째 data
#         return x[1]

if __name__ == '__main__':
### data 불러오기
    labels = []
    file_names = []
    csvPATH = "/home/ysy/AIHack2021_CC/specific_file_name.csv"  
    Data_list1 = pd.read_csv("/home/ysy/2021AI_data_Hackarthon/data_list_dev.csv")
    # print(type(Data_list1))  =  <class 'pandas.core.frame.DataFrame'>
    Data_list2 = pd.read_csv("/home/ysy/2021AI_data_Hackarthon/data_list_train.csv")

    
    data2num = {"Data_list1" : 1 , "Data_list2" : 2}
    num2Add = ["/home/ysy/dataset_malmoiDJ2021/dev/", "/home/ysy/dataset_malmoiDJ2021/train/"]
    # print(type(data2num))  =  <class 'dict'>
    # print(type(num2Add))  =  <class 'list'>

    Data_lists = [Data_list1,Data_list2]
    # print(str(Data_lists[0]))  =  pd
    # print(type(Data_lists))  =  <class 'list'>
    
    for j, Data_list in enumerate(Data_lists):
        for i in range(len(Data_list)):
            labels.append(Data_list.iloc[i].text)
            file_names.append(num2Add[j]+Data_list.iloc[i].file_name)
    # print(len(train_labels))
          
    # script 와 file csv에 쓰기
    with open(csvPATH, 'a', newline='') as f:
        wr = csv.writer(f)
        for Data_list in Data_lists:
            for i in range(len(Data_list)):
                # max_i = 635104
                wr.writerow([file_names[i],labels[i]])