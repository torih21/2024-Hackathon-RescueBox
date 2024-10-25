import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the CSV file
file_path = "messages.csv"  # Replace with your file path

# Load the CSV with a specified encoding
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin1')  # Fallback if needed


# Use a pipeline as a high-level helper

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# Load the LLaMA model
# Replace with the specific model you're using
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)

# Initialize a dictionary to store results
sentiment_results = {}

# Iterate over each column in the DataFrame
for column in df.columns:
    messages = df[column].dropna().tolist()  # Get non-null messages

    # Tokenize and predict sentiment
    inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=-1)

    # Interpret results
    labels = ["negative", "neutral", "positive"]  # Adjust based on your model
    results = [labels[prediction] for prediction in predictions]

    # Store results for the current conversation
    sentiment_results[column] = results

# Print sentiment analysis results for each conversation
for convo, results in sentiment_results.items():
    print(f"Conversation: {convo}")
    for message, result in zip(df[convo].dropna(), results):
        print(f"  Message: {message} | Sentiment: {result}")
