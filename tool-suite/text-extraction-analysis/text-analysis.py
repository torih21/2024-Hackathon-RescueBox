# Use a pipeline as a high-level helper
from transformers import pipeline
from llama import SentimentAnalyzer

messages = [
    {"role": "user", "content": "Who are you?"},    
]



pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
pipe(messages)