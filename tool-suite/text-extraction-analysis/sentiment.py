import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load messages from Excel
file_path = "messages.csv"  # Replace with your file path
df = pd.read_csv(file_path)
messages = df['Messages'].tolist()

# Load the LLaMA model
model_name = "facebook/llama-7b"  # or the specific LLaMA variant you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize and predict
inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits

predictions = torch.argmax(logits, dim=-1)

# Interpret results
labels = ["negative", "neutral", "positive"]  # Adjust based on your model
results = [labels[prediction] for prediction in predictions]

for message, result in zip(messages, results):
    print(f"Message: {message} | Sentiment: {result}")
