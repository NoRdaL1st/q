
# Use a pipeline as a high-level helper
from huggingface_hub import login
from transformers import pipeline

my_hf_token = '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'

login(my_hf_token)

pipe = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

text = "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales ."

x = pipe(text)

print(x)