# main_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import core_logic
import uvicorn
import os
# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Initialization ---
app = FastAPI(
    title="NeurOm Chatbot API",
    description="API for the NeurOm chatbot and mindfulness guide.",
    version="1.0.0"
)

# <<< --- CORRECTED CORS MIDDLEWARE BLOCK --- >>>
# This is crucial for allowing your Flutter web app to talk to the API
# We will use a wildcard "*" to allow all origins during local development.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)
# <<< --- END OF CORRECTION --- >>>


# --- Pydantic Model for Request Body ---
class ChatQuery(BaseModel):
    query: str
    profile_data: Optional[dict] = None

# --- API Endpoint ---
@app.post("/chat", summary="Get a response from the NeurOm chatbot")
async def get_chat_response(chat_query: ChatQuery):
    print(f"main_api.py: Received query: {chat_query.query}")
    if chat_query.profile_data:
        print(f"main_api.py: Received profile data: {chat_query.profile_data}")

    if not core_logic.RESOURCES_INITIALIZED:
        core_logic.initialize_resources()
    if core_logic.LLM_INSTANCE is None or core_logic.RETRIEVER_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Chatbot service is not ready.")
            
    is_initial = False
    initial_options = ["i want to relax and de-stress", "i want a mental challenge",
                       "i want to boost my focus", "i want to lift my spirits or find some calm"]
    if chat_query.query.lower() in [q.lower() for q in initial_options]:
        is_initial = True
        
    try:
        bot_answer = core_logic.generate_llm_response(
            user_query=chat_query.query,
            profile_data_from_api=chat_query.profile_data,
            is_initial_mood_selection=is_initial
        )
        print(f"main_api.py: Sending response: {bot_answer}")
        return {"answer": bot_answer}
    except Exception as e:
        print(f"main_api.py: Error during LLM response generation: {e}")
        raise HTTPException(status_code=500, detail="Error processing request.")

# --- For Local Testing: Run with Uvicorn ---
if __name__ == "__main__":
    # The check for Google Cloud Functions can remain, it won't affect local runs
    IS_GCF = os.getenv("K_SERVICE") is not None or os.getenv("FUNCTION_TARGET") is not None or os.getenv("FUNCTIONS_FRAMEWORK") is not None
    if not IS_GCF:
        print("main_api.py: Running API locally with Uvicorn...")
        if not core_logic.RESOURCES_INITIALIZED: core_logic.initialize_resources()
        if core_logic.LLM_INSTANCE and core_logic.RETRIEVER_INSTANCE:
            print("main_api.py: Starting Uvicorn on http://localhost:8000")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("main_api.py: LLM or Retriever not initialized. API cannot start.")
    else:
        print("main_api.py: Detected GCF environment. Uvicorn will not start.")
        if not core_logic.RESOURCES_INITIALIZED: core_logic.initialize_resources()