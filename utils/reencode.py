import os
import chardet

def convert_to_utf8(file_path):
    # 检测文件的原始编码
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        original_encoding = result['encoding']

    # 读取文件并重新编码为 UTF-8
    with open(file_path, 'r', encoding=original_encoding, errors='ignore') as f:
        content = f.read()

    # 以 UTF-8 编码保存文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def convert_directory_to_utf8(directory_path):
    # 遍历目录中的所有txt文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    convert_to_utf8(file_path)
                    print(f"Successfully converted {file_path} to UTF-8.")
                except Exception as e:
                    print(f"Failed to convert {file_path}: {e}")

# 指定要转换的目录路径
directory_path = '/home/suguilin/myfusion/datasets/MFNet/Text/Modal'

convert_directory_to_utf8(directory_path)
