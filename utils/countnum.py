import os

def count_txt_files(folder_path):
    return len([f for f in os.listdir(folder_path) if f.endswith('.txt')])

def count_png_files(folder_path):
    return len([f for f in os.listdir(folder_path) if f.endswith('.png')])

def compare_txt_file_counts(folder1, folder2):
    count1 = count_png_files(folder1)
    count2 = count_txt_files(folder2)

    if count1 == count2:
        print(f"两个文件夹中的 .txt 文件数量相同：{count1} 个")
    else:
        print(f"两个文件夹中的 .txt 文件数量不同："
              f"文件夹 1 ({folder1}) 有 {count1} 个，"
              f"文件夹 2 ({folder2}) 有 {count2} 个")

folder1 = '/home/suguilin/myfusion/datasets/MFNet/Modal'  
folder2 = '/home/suguilin/myfusion/datasets/MFNet/Text/RGB' 

compare_txt_file_counts(folder1, folder2)