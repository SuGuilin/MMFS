import torch
from torch import nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, BertModel, BertTokenizer
import os
import cv2
import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class TextEncoder(nn.Module):
    def __init__(self, device, simplify=False, blip2_model_name="Salesforce/blip2-opt-2.7b", clip_model_name="openai/clip-vit-base-patch32", bert_model_name="bert-base-uncased"):
        super().__init__()
        self.simplify = simplify
        self.device = device
        if simplify:
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(blip2_model_name, device_map="auto")
            self.blip2_processor = Blip2Processor.from_pretrained(blip2_model_name)
            
            self.clip_model = CLIPTextModel.from_pretrained(clip_model_name)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, text_list, image_list=[]):
        simplified_texts = []
        clip_encoded_texts = []
        bert_encoded_texts = []

        if self.simplify:
            for text, image in zip(text_list, image_list):
                inputs = self.blip2_processor(text=text, images=image, return_tensors="pt", padding=True)
                with torch.no_grad():
                    blip_outputs = self.blip2_model.generate(
                        pixel_values=inputs["pixel_values"].cuda(), 
                        input_ids=inputs['input_ids'].cuda(), 
                        min_length=20, 
                        max_length=150, 
                        do_sample=False, 
                        num_beams=10,
                        repetition_penalty=2.0,
                    )
                removed = self.blip2_processor.batch_decode(blip_outputs, skip_special_tokens=True)
                simplified_texts.append(removed)

                clip_encodings = self.clip_tokenizer(removed, return_tensors="pt", padding=True, truncation=True)
                clip_encodings = clip_encodings.to(self.device)
                with torch.no_grad():
                    clip_text_embeddings = self.clip_model(**clip_encodings).pooler_output
                clip_encoded_texts.append(clip_text_embeddings)
        
        # for text, text_name in zip(text_list, image_list):
        for text in text_list:
            bert_encodings = self.bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
            input_ids = bert_encodings["input_ids"].to(self.device)
            attention_mask = bert_encodings["attention_mask"].to(self.device)
            with torch.no_grad():
                bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the [CLS] token's embeddings
            # bert_text_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            bert_text_embeddings = bert_outputs.last_hidden_state #.pooler_output
            # print("bert_last:", bert_text_embeddings.shape)
            # npy_path = os.path.join('/home/suguilin/myfusion/datasets/MFNet/Text_Bert/RGB', '{}.npy'.format(text_name))
            # npy_bert = bert_text_embeddings.cpu().numpy()
            # np.save(npy_path, npy_bert)
            # print('{} npy has saved!!!'.format(text_name))
            bert_encoded_texts.append(bert_text_embeddings.squeeze())
        bert_encoded_texts = torch.stack(bert_encoded_texts)
        # print(bert_encoded_texts)
        return simplified_texts, clip_encoded_texts, bert_encoded_texts


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = TextEncoder(device=device)
    
    text_batch = [
        ['In this image with a resolution of 384X288, we can see a person walking down a street at night. The dense caption provides further details about the various objects present. A woman can be seen walking on the road, positioned between coordinates [97, 76, 171, 248]. Another person is also depicted walking, located at coordinates [256, 98, 276, 163]. A grey backpack can be observed on a person, positioned between coordinates [123, 110, 164, 173]. Moreover, a car is parked on the side of the road, situated at coordinates [63, 116, 111, 164]. Additionally, the region semantic provides additional information about different elements in the image. A woman is depicted walking with a suitcase between coordinates [0, 149, 383, 137]. There is also a black and white photo of a man standing in front of a wall, positioned at [341, 50, 42, 95]. Furthermore, a black object on a black background can be seen at coordinates [101, 169, 65, 45]. Additionally, a black and white photo of a building can be found, wherein the lights are on. It is located at [0, 97, 72, 33]. Lastly, a white figure is depicted against a black background, positioned at [124, 101, 38, 71].'], 
        ['In this 384x288 resolution image, a street scene unfolds before our eyes. Cars are parked in front of a building, creating a bustling urban atmosphere. A white car can be observed parked on the side of the road, specifically positioned between coordinates (272, 142) and (343, 185). Another car, described as being white, can also be found parked along the roadside, with its positioning spanning from (251, 148) to (278, 175). As we shift our attention towards the road, we notice the presence of white lines, stretching from (2, 186) to (185, 288). These lines guide the flow of traffic and add structure to the scene. Adjacent to the road, trees are visible on the sidewalk, occupying an area defined by the coordinates (1, 1) and (230, 175). Additional white lines grace the road, spanning from (1, 168) to (380, 288), delineating the various lanes for vehicles. Lastly, we can identify a sole white line adorning the road, positioned between coordinates (276, 208) and (371, 231). This image paints a vivid picture of an urban street, emphasizing the presence of cars, road markings, and greenery along the sidewalk.'], 
        ['In this 384X288 resolution image, the captivating scene unfolds at night as two people gracefully walk down a dimly lit street. The dense caption brings additional insight, revealing a person confidently strolling on the sidewalk, a woman in a white shirt nearby, and another person standing nearby. In the foreground, a white line appears, marking the side of the road. The region semantics provide a different perspective, describing the image as two people traversing a dark road at night, with a man standing against a black background. Additionally, a long black object with a long handle is present, while a black and white photo showcases a person standing on a hill. Adding to the intriguing composition, a black piano sits upon a black background. These elements coalesce within the image, inviting viewers to appreciate the beauty and serenity of nature blending seamlessly with human presence in the enigmatic night.'], 
        ['In this image, the resolution is 384X288, capturing a large building. The dense caption reveals various elements within the scene: a person can be seen walking on the sidewalk, while another stands in the distance. A long asphalt road stretches across the frame, accompanied by lush green grass on one side. Trees line the street, adding a touch of nature to the urban setting. Switching to the region semantic, a dark hallway comes into focus, with a person walking down it. Additionally, a green bush catches the eye, featuring a lighted area at its center. A single green leaf stands out against a black background. Finally, a picture of a gold and black square and a yellow light shining on a black background add visual interest to the overall composition.']
        ]
    text_names = []
    text_contents = []
    dir = '/home/suguilin/myfusion/datasets/MFNet/Text/RGB'

    # there are 1444 files in total
    for filename in os.listdir(dir):
        if filename.endswith('.txt'):
            text_path = os.path.join(dir, filename)
            text_names.append(os.path.splitext(filename)[0])  

            with open(text_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                assert len(content) == 1 , '{} has more than one line'.format(filename)
                text_contents.append(content)  
    print(text_names)
    # print(text_contents)
    image_batch = []
    img1 = torch.from_numpy(cv2.resize(cv2.imread('00001D.png'), (384, 288))).float().unsqueeze(0).permute(0, 3, 1, 2)
    # print(img1)
    img2 = torch.from_numpy(cv2.resize(cv2.imread('00002D.png'), (384, 288))).float().unsqueeze(0).permute(0, 3, 1, 2)
    img3 = torch.from_numpy(cv2.resize(cv2.imread('00003N.png'), (384, 288))).float().unsqueeze(0).permute(0, 3, 1, 2)
    img4 = torch.from_numpy(cv2.resize(cv2.imread('00004N.png'), (384, 288))).float().unsqueeze(0).permute(0, 3, 1, 2)
    image_batch.extend([img1, img2, img3, img4])

    # simplified_text_batch, clip_encoded_text_batch, bert_encoded_text_batch = text_encoder(text_batch, image_batch)
    simplified_text_batch, clip_encoded_text_batch, bert_encoded_text_batch = text_encoder(text_contents, text_names)
    # print(clip_encoded_text_batch[0].shape)
    # print(bert_encoded_text_batch[0].shape)

    for idx, simplified_texts in enumerate(simplified_text_batch):
        print(f"简化后的文本批次 {idx + 1}:")
        for text in simplified_texts:
            print(text)
    
    for idx, encoded_texts in enumerate(clip_encoded_text_batch):
        print(f"clip编码批次 {idx + 1}:")
        print(encoded_texts.shape)

    for idx, encoded_texts in enumerate(bert_encoded_text_batch):
        print(f"bert编码批次 {idx + 1}:")
        print(encoded_texts.shape)

