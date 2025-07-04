# main.py
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional

# --- Configuration ---
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set!")

# --- Pydantic Models (to mimic OpenAI's structure) ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
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

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- The Core Proxy Endpoint ---
@app.post("/")
async def chat_completions(request: ChatCompletionRequest):
    model_name = request.model
    
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
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        gemini_messages = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue
            role = "model" if msg.role == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [msg.content]})

        last_user_message = "..."
        if gemini_messages and gemini_messages[-1]['role'] == 'user':
            last_user_message = gemini_messages.pop()['parts']

        chat_session = model.start_chat(history=gemini_messages)
        
        full_prompt = last_user_message
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{last_user_message}"
        
        response = chat_session.send_message(full_prompt)

        response_message = ChatMessage(role="assistant", content=response.text)
        response_choice = ChatChoice(message=response_message)
        
        return ChatCompletionResponse(
            model=model_name,
            choices=[response_choice]
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint for Render
@app.get("/")
def health_check():
    return {"status": "ok"}