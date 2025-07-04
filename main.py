# main.py
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, SafetySetting, HarmCategory
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import base64

# --- Configuration ---
# This setup is for Render and is more secure.
# It reads the base64-encoded JSON credentials from an environment variable.
try:
    # Get the base64-encoded string from environment variables
    gcp_creds_base64 = os.environ['GCP_SA_KEY_BASE64']
    # Decode the base64 string into a JSON string
    gcp_creds_json = base64.b64decode(gcp_creds_base64).decode('utf-8')
    # Parse the JSON string into a Python dictionary
    gcp_creds_dict = json.loads(gcp_creds_json)
    
    # Initialize Vertex AI
    vertexai.init(credentials=gcp_creds_dict)
    
except KeyError:
    raise RuntimeError("GCP_SA_KEY_BASE64 environment variable not set! Please provide your base64-encoded Service Account key.")
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
    # --- Vertex AI Specific Configuration ---
    # NOTE: Vertex AI uses a different model naming convention
    # We will translate common names to the Vertex format.
    model_name = request.model
    if "gemini-1.5-pro" in model_name:
        model_name = "gemini-1.5-pro-001"
    elif "gemini-1.5-flash" in model_name:
        model_name = "gemini-1.5-flash-001"
    # Add other translations as needed

    # THE CRITICAL PART: This is the Vertex AI way to disable safety filters.
    # It is more effective than the previous method.
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
        # Prepare messages for Vertex AI
        vertex_messages = []
        system_instruction = None
        for msg in request.messages:
            if msg.role == "system":
                # Vertex AI handles system instructions separately
                system_instruction = Part.from_text(msg.content)
                continue
            # Vertex AI uses 'user' and 'model' roles
            role = "model" if msg.role == "assistant" else "user"
            vertex_messages.append(Part.from_dict({'role': role, 'text': msg.content}))

        # Initialize the Vertex AI model
        model = GenerativeModel(
            model_name,
            system_instruction=[system_instruction] if system_instruction else None
        )

        # Send the request to Vertex AI
        response = await model.generate_content_async(
            vertex_messages,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Format the response back to OpenAI standard
        response_text = response.text
        response_message = ChatMessage(role="assistant", content=response_text)
        response_choice = ChatChoice(message=response_message)
        
        return ChatCompletionResponse(model=request.model, choices=[response_choice])

    except Exception as e:
        # Log the detailed error from Vertex AI
        print(f"An error occurred with Vertex AI: {e}")
        # Raise an HTTPException to send a clear error message back to the client
        raise HTTPException(status_code=500, detail=f"Vertex AI Error: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok"}