from transformers import pipeline

# 加载预训练的文本摘要模型
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def simplify_text(text):
    # 使用预训练模型生成简洁的描述
    summarized_text = summarizer(text, max_length=100, min_length=50, do_sample=False)
    return summarized_text[0]['summary_text']

# 示例文本
text = """
In this image with a resolution of 384X288, we observe a street adorned with various elements. 
The dense caption informs us tobject there is a paved road present, measuring 1,163 pixels in width and 381 pixels in height. 
Additionally, we can spot a white sign with black letters spanning 314 pixels in width and 26 pixels in height. 
This sign is positioned at coordinates 382 pixels in width and 112 pixels in height. 
...
"""

simplified_text = simplify_text(text)
print(simplified_text)
