import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, AutoProcessor, LlavaConfig, LlavaForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import cv2
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

warnings.filterwarnings('ignore')

class FeatureExtractor(nn.Module):
    def __init__(self, device, model_id="openai/clip-vit-large-patch14"):#"openai/clip-vit-base-patch32"):
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_id).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)

    def forward(self, image):
        # use CLIP process image
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        # obtain the image feature of CLIP
        clip_features = self.clip_model.get_image_features(**inputs)
        return clip_features


class LLAVAModule(nn.Module):
    def __init__(self, device, model_id="llava-hf/llava-1.5-7b-hf"):
        super(LLAVAModule, self).__init__()
        self.device = device
        # load pretrained LLaVA model
        self.llava_processor = AutoProcessor.from_pretrained(model_id)
        # processor.vision_feature_select_strategy = YOUR_VISION_FEATURE_SELECT_STRATEGY
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,)

    def forward(self, image, queries):
        chat = []
        answers = []
        # generate text description by using LLaVa
        for query in queries:
            chat.append({"role": "user", "content": query})
            # print(chat)
            prompt = self.llava_processor.apply_chat_template(chat, add_generation_prompt=True)
            inputs = self.llava_processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            # with torch.no_grad():
            output = self.llava_model.generate(**inputs, max_new_tokens=200, do_sample=False)
            input_ids = inputs["input_ids"]
            cutoff = len(self.llava_processor.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        ))
            answer = self.llava_processor.decode(output[0], skip_special_tokens=True)[cutoff:]
            chat.pop()
            answers.append(answer)
        return answers


class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super(CrossAttentionFusion, self).__init__()
        # linear handle QKV
        self.query = nn.Linear(512, 512)
        self.key = nn.Linear(512, 512)
        self.value = nn.Linear(512, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, ir_features, vis_features, text_features):
        # Query is image feature，Key and Value is text feature
        Q = self.query(ir_features)
        K = self.key(text_features)
        V = self.value(text_features)
        # cross-attention
        fused_features, _ = self.attention(Q, K, V)
        return fused_features


def cosine_similarity(feature1, feature2):
    return torch.nn.functional.cosine_similarity(feature1, feature2)


class FullModel(nn.Module):
    def __init__(self, device, t5_model='t5-small'):  # 可以选择't5-small', 't5-base', 't5-large'
        super(FullModel, self).__init__()
        self.device = device
        # self.feature_extractor = FeatureExtractor(device=device)
        self.llava_module = LLAVAModule(device=device).to(device)
        self.clip_text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")#"openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.cross_attention_fusion = CrossAttentionFusion().to(device=device)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
    
    def summarize_text(self, text, max_length=77):
        input_text = f"summarize: {text}"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        summary_ids = self.t5_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4, 
            length_penalty=2.0, 
            early_stopping=True
        )
        
        # transfer generated ID to text
        summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def forward(self, image, queries):
        # 1. extract image feature
        inputs = self.clip_text_processor(images=image, return_tensors="pt").to(self.device)
        # image_features = self.feature_extractor(image)
        image_features = self.clip_model.get_image_features(**inputs)
        # print("img", image_features.shape)
        # 2. generate text answer
        answers = self.llava_module(image, queries)
        text_features = []
        # 3. extract text feature
        for answer in answers:
            if (len(answer) > 77):
                answer = self.summarize_text(answer)
            # print(answer)
            inputs_text = self.clip_text_processor(text=[answer], return_tensors="pt", padding=True, truncation=True, max_length=200).to(self.device)
            text_feature = self.clip_model.get_text_features(**inputs_text)
            text_features.append(text_feature)
            # print("text:", text_feature.shape)
        # # 4. compute cosine similarity
            # similarity = cosine_similarity(image_features, text_features)
            # print(similarity)
        combined_features = torch.cat([image_features, torch.cat(text_features, dim=0)], dim=0)
        # print(combined_features.shape)
        return answers, combined_features


def process_images(folder_path, model, queries, device):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).unsqueeze(0).to(device)

            answers, combined_features = model(image=image, queries=queries)
            # print(answers)

            filename = os.path.splitext(filename)[0]
            print(filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join('/home/suguilin/MMFS/datasets/MFNet/Text/RGB', txt_filename)
            npy_path = os.path.join('/home/suguilin/MMFS/datasets/MFNet/Text_CLIP/RGB', '{}.npy'.format(filename))
            npy_clip = combined_features.detach().cpu().numpy()
            np.save(npy_path, npy_clip)

            with open(txt_path, "w") as file:
                for answer in answers:
                    file.write(answer + "\n")

# 定义方向损失 L_d
def direction_loss(XT):  # {'vi': {'image': 'x', 'text': 'y'}, 'ir': {'image': 'z', 'text': 'w'}, 'fused': {'image': 'a', 'text': 'b'}}
    delta_V_vi = XT['fused']['image'] - XT['vi']['image']
    delta_V_ir = XT['fused']['image'] - XT['ir']['image']
    delta_T_vi = XT['fused']['text'] - XT['vi']['text']
    delta_T_ir = XT['fused']['text'] - XT['ir']['text']
    term1 = F.cosine_similarity(delta_V_vi, delta_T_vi, dim=-1)
    term2 = F.cosine_similarity(delta_V_ir, delta_T_ir, dim=-1)
    L_d = 1 - 0.5 * (term1 + term2).mean()
    return L_d


# 定义正则化项 Φ
def regularization_term(XT):
    V_f = XT['fused']['image']
    V_vi = XT['vi']['image']
    V_ir = XT['ir']['image']
    term1 = 1 - F.cosine_similarity(V_f, V_vi, dim=-1).mean()
    term2 = 1 - F.cosine_similarity(V_f, V_ir, dim=-1).mean()
    Phi = term1 + term2
    return Phi


# 定义总的语言驱动融合损失 L_g
def language_driven_fusion_loss(XT, lambda_reg=0.5):
    L_d = direction_loss(XT)
    Phi = regularization_term(XT)
    L_g = L_d + lambda_reg * Phi  
    return L_g



if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # image = cv2.imread("00093D.png", 1) # Image.open("00093D.png")
    # image = torch.from_numpy(np.array(image)).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # question = [
    #     {
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "text": "Describe the content of this image in detail."},
    #         {"type": "image"},
    #         ],
    #     },
    # ]

    queries = [
        #     [
        #     {"type": "text", "text": "What type of degradation has this image suffered?"},  # Please describe in 77 words or less which regions have higher contrast in this image.
        #     {"type": "image"},
        # ],

        #     [
        #     {"type": "text", "text": "Where is the blur in this image?"},  # Please describe in 77 words or less which regions have higher contrast in this image.
        #     {"type": "image"},
        # ],

        #     [
        #     {"type": "text", "text": "which regions have higher contrast in this image?"},  # Please describe in 77 words or less which regions have higher contrast in this image.
        #     {"type": "image"},
        # ],

            [
            {"type": "text", "text": "Can you describe the main scene depicted in the image?"},
            {"type": "image"},
        ],
        
            [
            {"type": "text", "text": "What are the key objects present in the image?"},
            {"type": "image"},
        ],

            [
            {"type": "text", "text": "Which regions in this image are visually significant or contain important details?"}, # Please describe in 77 words or less which regions should be noticed in this image.
            {"type": "image"},
        ],
            
        #     [
        #     {"type": "text", "text": "What targets are significant in this image?"}, # Please describe in 77 words or less what targets are significant in this image?
        #     {"type": "image"},
        # ],
        #     [
        #     {"type": "text", "text": "What actions are taking place in the image?"},
        #     {"type": "image"},
        # ],
            [
            {"type": "text", "text": "Finally, can you summarize the image in a single descriptive sentence?"},
            {"type": "image"},
        ],
        #     [
        #     {"type": "text", "text": "What emotions or atmosphere does the image convey?"},
        #     {"type": "image"},
        # ],
    ]

    # question = [
    #     [
    #         {"type": "text", "text": "describe the images in a sentence."},
    #         {"type": "image"},
    #     ],
    # ]
    # model = FullModel(device=device)
    # folder_path='/home/suguilin/MMFS/datasets/MFNet/RGB'
    # process_images(folder_path=folder_path, model=model, queries=queries, device=device)

    
    # inference
    # answer, similarity = model(image, queries)
    # print("The answer is:", answer)
    # print("Cosine Similarity:", similarity)
    # V_f = torch.randn(4, 768)
    # V_ir = torch.randn(4, 768)
    # term2 = 1 - F.cosine_similarity(V_f, V_ir, dim=-1).mean()
    # print(term2)


# model_id = "llava-hf/llava-1.5-7b-hf"

# prompt = "USER:\n Describe what is going on in image? <image>\nASSISTANT:"
# image_file = "/home/suguilin/LLM/00099D.png"

# model = LlavaForConditionalGeneration.from_pretrained(
#    model_id, 
#    torch_dtype=torch.float16, 
#    low_cpu_mem_usage=False, 
# ).to(1)

# processor = AutoProcessor.from_pretrained(model_id)

# raw_image = Image.open(image_file)

# # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# # image_2 = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(1, torch.float16)
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))


# inputs = processor(images=image_2, text=prompt, return_tensors='pt').to(1, torch.float16)
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))


# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(1, torch.float16)
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))


'''
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(1)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(1, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
'''