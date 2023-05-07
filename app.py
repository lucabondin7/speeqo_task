import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import torch
from transformers import GPTJForCausalLM, AutoTokenizer

app = FastAPI()

# Define a model for the request body
class ChatCompletionRequest(BaseModel):
    text: str

# Define a model for the Hugging Face API response
class ChatCompletionResponse(BaseModel):
    completions: List[str]

# Define a model for the API key
class HuggingFaceAPIKey(BaseModel):
    api_key: str


async def call_huggingface_api(text: str) -> str:
    """
    Call the Hugging Face Chat Completion API and return the generated text.

    :param text: The text to complete.
    :return: The generated text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    prompt = text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    responses = []
    for i in range(0,5):
        generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)
        generated_text = tokenizer.decode(generated_ids[0])
        responses.append(generated_text)
    return responses


@app.post("/chat_completion", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Generate text using the Hugging Face Chat Completion API.

    :param request: The request containing the text to complete 
    :return: The generated text.
    """
    generated_text = await call_huggingface_api(request.text)
    return ChatCompletionResponse(text=generated_text)
