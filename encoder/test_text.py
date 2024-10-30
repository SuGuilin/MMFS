from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载预训练的BLIP模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# 加载图像
image = Image.open("/home/suguilin/Graduation/myfusion/datasets/MFNet/RGB/00006N.png")

# 输入冗余/错误的文本描述
initial_text = "In the image with a resolution of 384X288, a car can be seen driving down a street. The dense caption reveals additional details, highlighting a parked silver car and a red and white sign. The region semantic further specifies various objects and their positions within the image. A black and white picture showcases a figure on a street, while a silver car is depicted against a black background. Close up shots feature a green figure with a white label, a white toothbrush with a long handle, and a white long stick with a long handle."
initial_text = "Generate a rich and detailed description of the image, including the environment, objects, and actions. Focus on creating a vivid scene."
# 处理输入的图像和文本
inputs = processor(image, return_tensors="pt")

# 生成优化后的文本描述
output = model.generate(**inputs, max_length=2000, num_beams=5, temperature=0.8)

# 输出最终的文本描述
print(processor.decode(output[0], skip_special_tokens=True))
description = processor.decode(output[0], skip_special_tokens=True)
cleaned_description = re.sub(r'\b\d+x\d+\b', '', description)
print(cleaned_description)



# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from PIL import Image
# import torch

# # 加载预训练的 BLIP-2 模型和处理器
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# # 加载图像
# image_path = "/home/suguilin/Graduation/myfusion/datasets/MFNet/RGB/00004N.png"  # 替换为你的图像路径
# image = Image.open(image_path)

# # 处理图像并生成描述
# inputs = processor(images=image, return_tensors="pt")

# # 使用 BLIP-2 生成描述
# with torch.no_grad():
#     generated_ids = model.generate(**inputs, max_length=100)

# # 解码生成的描述
# generated_description = processor.decode(generated_ids[0], skip_special_tokens=True)

# # 输出生成的图像描述
# print("Generated Description:")
# print(generated_description)

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 加载 BLIP-2 处理器和使用 OPT 2.7B 的模型
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# 加载图像
image_path = "/home/suguilin/Graduation/myfusion/datasets/MFNet/RGB/00004N.png"  # 替换为你的图像路径
image = Image.open(image_path)

# 处理图像并生成输入
inputs = processor(images=image, return_tensors="pt")

# 使用 BLIP-2 模型生成描述
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=2000, num_beams=5, temperature=0.8)

# 解码生成的描述
generated_description = processor.decode(generated_ids[0], skip_special_tokens=True)

# 输出生成的图像描述
print("Generated Description:")
print(generated_description)
