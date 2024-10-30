import pandas as pd
import os

dataset = ['MFNet'] #['TNO', 'Roadscene', 'MSRS', 'M3FD_Fusion']

# 创建一个空的 Excel 文件作为结果
writer = pd.ExcelWriter('exp1_MFNet.xlsx', engine='xlsxwriter')
for name in dataset:
    excel_file = os.path.join('./', 'metric_exp1_epoch50_{}.xlsx'.format(name))
    # 创建一个空的 DataFrame 用于存储结果
    result_df = pd.DataFrame()
    # 遍历每个 sheet
    for sheet_name in pd.read_excel(excel_file, sheet_name=None):
        # 读取当前 sheet 的数据
        sheet_df = pd.read_excel(excel_file, skiprows=1, sheet_name=sheet_name)
        
        # 计算每一列的平均值
        sheet_means = sheet_df.mean()
        
        # 将平均值添加到结果 DataFrame 中
        result_df[sheet_name] = sheet_means

    # 将结果 DataFrame 写入到 result.xlsx 文件的对应 sheet 中
    result_df.to_excel(writer, index_label='Metric', sheet_name=name)
writer._save()