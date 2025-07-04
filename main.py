# main.py
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- Configuration ---
# Load the API key from an environment variable for security
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    # This error will be caught by Render's health check if the key is missing
    raise RuntimeError("GOOGLE_API_KEY environment variable not set!")

# --- Pydantic Models (to mimic OpenAI's structure) ---
# These models define the expected request and response formats.
# Janitor.ai sends requests in this format.

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemini-1.5-pro-latest" # The model name is passed but we'll use our own
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-fake-id"
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: List[ChatChoice]

# --- FastAPI Application ---
app = FastAPI()

# --- The Core Proxy Endpoint ---
# This is where Janitor.ai will send its requests.
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    The main endpoint that mimics OpenAI's chat completions API.
    """
    # --- Model and Safety Configuration ---
    # NOTE: You can change the model name here if you want to use a different one
    model_name = "gemini-1.5-pro-latest"
    
    # CRITICAL: This disables all safety filters.
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    generation_config = {
        "temperature": request.temperature,
        "max_output_tokens": request.max_tokens,
    }
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # --- Message Translation ---
    # Gemini requires a specific format. We convert the OpenAI-style
    # messages into what Gemini expects.
    gemini_messages = []
    system_instruction = None

    for msg in request.messages:
        # Gemini 1.5 Pro has a dedicated 'system' instruction
        if msg.role == "system":
            system_instruction = msg.content
            continue
        # Gemini uses 'model' for the 'assistant' role
        role = "model" if msg.role == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [msg.content]})

    try:
        # --- Calling the Gemini API ---
        chat_session = model.start_chat(
            history=gemini_messages
        )
        
        # We need to find the last user message to send to Gemini
        last_user_message = "..." # Default message if no user message found
        if gemini_messages and gemini_messages[-1]['role'] == 'user':
            last_user_message = gemini_messages.pop()['parts'][0] # Get content of last user message

        # Add system instruction to the model if it exists
        # This is a bit of a workaround to apply the system prompt effectively
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{last_user_message}"
            response = chat_session.send_message(full_prompt)
        else:
            response = chat_session.send_message(last_user_message)

        # --- Formatting the Response for Janitor.ai ---
        response_message = ChatMessage(role="assistant", content=response.text)
        response_choice = ChatChoice(message=response_message)
        
        return ChatCompletionResponse(
            model=model_name,
            choices=[response_choice]
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        # In case of an error with the Gemini API, return a server error
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint for Render
@app.get("/")
def health_check():
    return {"status": "ok"}