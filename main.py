# main.py
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORT THE NEW MODULE
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

# --- NEW: CORS (Cross-Origin Resource Sharing) Middleware ---
# This is the crucial part that allows Janitor.ai to talk to your proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# -------------------------------------------------------------

# --- The Core Proxy Endpoint ---
# This path is based on Janitor.ai's "Proxy" tab setting
@app.post("/")
async def chat_completions(request: ChatCompletionRequest):
    """
    The main endpoint that mimics OpenAI's chat completions API.
    Janitor AI's "Proxy" tab sends requests to the root path.
    """
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

        # We need the last user message to send to Gemini
        last_user_message = "..."
        if gemini_messages and gemini_messages[-1]['role'] == 'user':
            last_user_message = gemini_messages.pop()['parts'][0]

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
        # This makes the error visible in the Render logs
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint for Render
@app.get("/")
def health_check():
    return {"status": "ok"}
```***Note:*** *I also changed the endpoint from `/v1/chat/completions` to just `/` to better match the URL structure Janitor now uses in its "Proxy" tab. This makes our proxy more robust.*

---

#### **Part 2: Re-deploy the Final Code**

1.  Save the updated `main.py` file.
2.  Go to your **Git Bash** terminal.
3.  Run these commands to upload the fix:

    ```bash
    git add main.py
    git commit -m "Fix CORS and update endpoint path"
    git push origin main
    ```

---

#### **Part 3: Final Check**

1.  Wait for Render to finish deploying your update (the status will change to "Deploying" and then back to "Live").
2.  Go back to your Janitor.ai settings. The configuration you had in your last screenshot is **perfect**. Do not change it.
    *   **Proxy Tab**
    *   Model: `gemini-2.5-pro` (or `gemini-1.5-pro-latest` if you prefer)
    *   URL: `https://da1lizz-gemini-proxy.onrender.com`
    *   API Key: Any random text.
3.  Click **"Check API Key/Model"**.

It should now work. Congratulations in advance