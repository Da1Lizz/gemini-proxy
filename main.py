# main.py
import os
import json
import vertexai
# The new, correct imports
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationConfig, SafetySetting, HarmCategory
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import base64

# --- Configuration ---
try:
    gcp_creds_base64 = os.environ['GCP_SA_KEY_BASE64']
    gcp_creds_json = base64.b64decode(gcp_creds_base64).decode('utf-8')
    gcp_creds_dict = json.loads(gcp_creds_json)
    # Initialize Vertex AI with the project ID and credentials
    vertexai.init(project=gcp_creds_dict.get('project_id'), credentials=gcp_creds_dict)
except KeyError:
    raise RuntimeError("GCP_SA_KEY_BASE64 environment variable not set!")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

# --- Pydantic Models ---
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

# --- FastAPI App ---
app = FastAPI()

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
    # Translate model names to Vertex AI format
    if "gemini-1.5-pro" in model_name:
        model_name = "gemini-1.5-pro-001"
    elif "gemini-1.5-flash" in model_name:
        model_name = "gemini-1.5-flash-001"
    elif "gemini-2.5-pro" in model_name: # Future-proofing
        model_name = "gemini-2.5-pro-001"

    # Effective safety filter override for Vertex AI
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = GenerationConfig(
        temperature=request.temperature,
        max_output_tokens=request.max_tokens,
    )

    try:
        # --- THE 100% CORRECTED MESSAGE FORMATTING ---
        contents = []
        system_instruction = None
        for msg in request.messages:
            if msg.role == "system":
                system_instruction = Part.from_text(msg.content)
                continue
            
            role = "model" if msg.role == "assistant" else "user"
            # Create a proper 'Content' object as required by the library
            contents.append(Content(role=role, parts=[Part.from_text(msg.content)]))
        # ---------------------------------------------

        model = GenerativeModel(
            model_name,
            system_instruction=[system_instruction] if system_instruction else None
        )

        response = await model.generate_content_async(
            contents, # Pass the list of Content objects
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        # Check for safety blocks even with Vertex, just in case
        if not response.candidates or not response.candidates[0].content.parts:
            response_text = "[Response was fully blocked by Google Vertex AI's non-negotiable safety filters. This can happen in extreme cases. Please adjust the prompt.]"
        else:
            response_text = response.candidates[0].content.parts[0].text

        response_message = ChatMessage(role="assistant", content=response_text)
        response_choice = ChatChoice(message=response_message)
        
        return ChatCompletionResponse(model=request.model, choices=[response_choice])

    except Exception as e:
        print(f"An error occurred with Vertex AI: {e}")
        raise HTTPException(status_code=500, detail=f"Vertex AI Error: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok"}