# Retrieval Augmented Generation (RAG) with TinyLlama
# Purpose: proof-of-concept for agentic-AI workflow

# imports
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForDocumentQuestionAnswering, DocumentQuestionAnsweringPipeline

# Global variables
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

##################################################################################################

# What is the biggest mining company in the world?
model_task = "text-generation"
pipe = pipeline(task=model_task, model=MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
messages = [
    {"role": "system","content": "You are a helpful assistant."},
    {"role":"user", "content": "What is the biggest mining company in the world?"}]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]['generated_text'])
##################################################################################################