# core_logic.py
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import Optional, List # For type hinting

print("core_logic.py: Loading environment variables...")
load_dotenv()

# --- Global Instances (Initialized by initialize_resources) ---
LLM_INSTANCE = None
RETRIEVER_INSTANCE = None
RESOURCES_INITIALIZED = False # Flag to ensure initialization runs only once
# ---

# --- Configuration ---
PDF_FILES_CONFIG = [
    "CerboTech Chatbot doc (3).pdf",
    "The_Miracle_of_Mindfulness__An_Introductio_-_Thich_Nhat_Hanh.pdf",
    "zenmind.pdf",
    "Mindfulness_in_Plain_English.pdf",
    "Kathleen_McDonald_Robina_Courtin_How_to.pdf",
    "Daniel Goleman_ Richard J. Davidson - The Science of Meditation_ How to Change Your Brain, Mind and Body .pdf"
]
CHROMA_PERSIST_DIRECTORY = "chroma_db_api_neuroum" # Directory to persist ChromaDB
# --- End of Configuration ---

def get_current_time_info():
    now = datetime.now()
    current_time_str = now.strftime("%I:%M %p")
    current_hour = now.hour
    return current_time_str, current_hour

def initialize_resources():
    global LLM_INSTANCE, RETRIEVER_INSTANCE, RESOURCES_INITIALIZED

    if RESOURCES_INITIALIZED:
        print("core_logic.py: Resources already initialized.")
        return

    print("core_logic.py: Initializing LLM and Retriever for API...")

    # Initialize LLM
    try:
        LLM_INSTANCE = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", # or gemini-1.0-pro
            temperature=0.1,
        )
        print("core_logic.py: LLM Initialized Successfully.")
    except Exception as e:
        print(f"core_logic.py: CRITICAL Error - LLM failed to initialize: {e}")
        LLM_INSTANCE = None

    # Initialize Embeddings (crucial for Retriever)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Or "text-embedding-004"
        print("core_logic.py: Embeddings Initialized Successfully.")
    except Exception as e:
        print(f"core_logic.py: CRITICAL Error - Failed to initialize Embeddings: {e}")
        # If embeddings fail, retriever cannot be initialized
        RETRIEVER_INSTANCE = None
        RESOURCES_INITIALIZED = True # Mark as initialized to prevent retry loops if this is the core error
        if LLM_INSTANCE is None: print("core_logic.py: LLM also failed or was not initialized.") # Secondary check
        return # Exit early if embeddings fail

    # Initialize Retriever
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets directory where script is running
    db_path = os.path.join(script_dir, CHROMA_PERSIST_DIRECTORY)

    if os.path.exists(db_path) and os.listdir(db_path): # Check if dir exists and is not empty
        try:
            print(f"core_logic.py: Loading existing vector store from {db_path}...")
            vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
            RETRIEVER_INSTANCE = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})
            print("core_logic.py: Retriever loaded successfully from persisted directory.")
        except Exception as e:
            print(f"core_logic.py: Error loading persisted Chroma DB from {db_path}: {e}. Will try to recreate.")
            RETRIEVER_INSTANCE = None # Fallback to recreation if loading fails
    
    if RETRIEVER_INSTANCE is None: # If not loaded or loading failed, create/recreate it
        print(f"core_logic.py: Creating/Recreating vector store, then persisting to {db_path}...")
        all_pages = []
        loaded_one = False
        for pdf_filename in PDF_FILES_CONFIG:
            full_pdf_path = os.path.join(script_dir, pdf_filename)
            if os.path.exists(full_pdf_path):
                try:
                    loader = PyPDFLoader(full_pdf_path)
                    pages = loader.load()
                    all_pages.extend(pages)
                    loaded_one = True
                    print(f"core_logic.py: Loaded {len(pages)} pages from {pdf_filename}.")
                except Exception as e_load:
                    print(f"core_logic.py: Error loading {pdf_filename} from {full_pdf_path}: {e_load}")
            else:
                print(f"core_logic.py: PDF not found at: {full_pdf_path}. Ensure it's in the same directory as core_logic.py or update path in PDF_FILES_CONFIG.")
        
        if loaded_one and all_pages:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs_chunks = text_splitter.split_documents(all_pages)
            print(f"core_logic.py: Total documents after splitting: {len(docs_chunks)}")
            if docs_chunks:
                try:
                    vectorstore = Chroma.from_documents(
                        documents=docs_chunks,
                        embedding=embeddings,
                        persist_directory=db_path # Persist to the specified directory
                    )
                    # vectorstore.persist() # from_documents with persist_directory should handle initial save
                    RETRIEVER_INSTANCE = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})
                    print("core_logic.py: Retriever created and persisted successfully.")
                except Exception as e_chroma:
                    print(f"core_logic.py: Error creating Chroma DB: {e_chroma}")
                    RETRIEVER_INSTANCE = None
            else:
                print("core_logic.py: No chunks generated from PDFs. Retriever not created.")
                RETRIEVER_INSTANCE = None
        else:
            print("core_logic.py: No PDFs loaded or no content extracted. Retriever not created.")
            RETRIEVER_INSTANCE = None
    
    if RETRIEVER_INSTANCE is None:
        print("core_logic.py: CRITICAL ERROR - Retriever could not be initialized.")
    if LLM_INSTANCE is None: # Check again in case it failed but embeddings succeeded
        print("core_logic.py: CRITICAL ERROR - LLM could not be initialized.")
    
    RESOURCES_INITIALIZED = True

# --- Dummy format_profile_for_prompt (as Pydantic models are in main_api.py) ---
# This function expects a dictionary.
def format_profile_for_prompt(profile_data_dict: Optional[dict]) -> str:
    if not profile_data_dict:
        return "" # Return empty string if no profile data

    profile_summary_parts = []
    # Safely access nested keys using .get()
    if profile_data_dict.get("stress_coping"):
        sc = profile_data_dict["stress_coping"]
        if sc.get("stressors"): profile_summary_parts.append(f"- Main stressors: {', '.join(sc['stressors'])}")
        if sc.get("overwhelmed_response"): profile_summary_parts.append(f"- Typical response to overwhelm: {', '.join(sc['overwhelmed_response'])}")
        if sc.get("self_care_practices"): profile_summary_parts.append(f"- Regular self-care: {', '.join(sc['self_care_practices'])}")
    
    if profile_data_dict.get("hobbies"):
        h = profile_data_dict["hobbies"]
        if h.get("interests"): profile_summary_parts.append(f"- Hobbies/Interests: {', '.join(h['interests'])}")

    if profile_data_dict.get("mood_emotions"):
        me = profile_data_dict["mood_emotions"]
        if me.get("recent_mood_rating"): profile_summary_parts.append(f"- Recent mood rating: {me['recent_mood_rating']}")
        if me.get("unpleasant_emotions"): profile_summary_parts.append(f"- Often struggles with: {', '.join(me['unpleasant_emotions'])}")
        
    if profile_data_dict.get("mental_health"):
        mh = profile_data_dict["mental_health"]
        if mh.get("improvement_areas"): profile_summary_parts.append(f"- Wants to improve in: {', '.join(mh['improvement_areas'])}")
            
    if not profile_summary_parts:
        return "User has not provided detailed profile data for this interaction, or the provided sections were empty."
        
    return "\nSome information about the user's profile from their NeurOm assessment:\n" + "\n".join(profile_summary_parts) + "\n"


def generate_llm_response(user_query, profile_data_from_api: Optional[dict] = None, is_initial_mood_selection=False):
    if not RESOURCES_INITIALIZED:
        print("core_logic.py: generate_llm_response called before resources initialized. Attempting init.")
        initialize_resources()

    if RETRIEVER_INSTANCE is None:
        return "I'm sorry, my knowledge base isn't available right now. Please try again later or contact support if the issue persists."
    if LLM_INSTANCE is None:
        return "I'm sorry, the language model isn't available right now. Please try again later or contact support if the issue persists."

    user_query_lower = user_query.lower()
    severe_distress_keywords = [
        "i want to die", "suicidal", "thinking about hurting myself", "kill myself",
        "end my life", "no hope", "hopeless", "can't cope anymore", "crippling anxiety"
    ]
    is_severe_distress = any(keyword in user_query_lower for keyword in severe_distress_keywords)
    if "depressed" in user_query_lower and \
       ("can't function" in user_query_lower or "can't do anything" in user_query_lower or \
        "overwhelming sadness all the time" in user_query_lower or "want to give up" in user_query_lower):
        is_severe_distress = True

    if is_severe_distress:
        return (
            "I hear that you're going through a very difficult time, and I want you to know that your feelings are valid. "
            "While NeurOm offers tools for general well-being, for what you're describing, "
            "it's really important to talk to someone who can offer professional support. "
            "If you're feeling severely depressed or overwhelmed, or if you are in crisis, "
            "I strongly encourage you to reach out to a psychiatrist, therapist, or another qualified mental health professional immediately. "
            "Please remember, you don't have to go through this alone. There are people who can support you."
        )
    
    profile_prompt_text = format_profile_for_prompt(profile_data_from_api)

    current_time_string, current_hour_int = get_current_time_info()
    time_specific_guidance = ""
    prompt_instruction_for_mood = ""
    system_prompt_time_mention = ""

    general_meditation_alternative = "exploring the 'BreatheEasy' techniques in the NeurOm app, listening to some calming 'Music' from our selection, or perhaps trying a general mindfulness practice from the provided texts"
    general_music_alternative = "our general 'Music' selection in the NeurOm app for calm or focus (available any time), or perhaps a quiet moment for reflection as suggested in the mindfulness books"
    is_time_specific_feature_query = False

    if "morning meditation" in user_query_lower:
        is_time_specific_feature_query = True
        system_prompt_time_mention = f"The user's current time is {current_time_string}. "
        start_hour, end_hour = 4, 12
        if start_hour <= current_hour_int < end_hour:
            time_specific_guidance = f"This is a great time for NeurOm's Morning Meditation (best between 4 AM and 12 PM)!"
        else:
            time_specific_guidance = (f"Regarding NeurOm's Morning Meditation, "
                                      f"this activity is best performed between 4 AM and 12 PM. "
                                      f"Since it's outside this window, perhaps you'd like to try {general_meditation_alternative} instead?")
    elif "night music" in user_query_lower or ("sleep" in user_query_lower and "music" in user_query_lower and ("neurom" in user_query_lower or "app" in user_query_lower)):
        is_time_specific_feature_query = True
        system_prompt_time_mention = f"The user's current time is {current_time_string}. "
        start_hour, end_hour_next_day = 20, 3
        is_time = (current_hour_int >= start_hour) or (current_hour_int < end_hour_next_day)
        if is_time:
            time_specific_guidance = (f"This is a perfect time for NeurOm's Night Music (best between 8 PM and 3 AM) to help you unwind for sleep.")
        else:
            time_specific_guidance = (f"Regarding NeurOm's Night Music, "
                                      f"this audio is best listened to between 8 PM and 3 AM. "
                                      f"At this time, you might enjoy {general_music_alternative} instead?")

    if not is_time_specific_feature_query:
        mood_keywords = ["stressed", "anxious", "overwhelmed", "bored", "unfocused", "distracted", "down", "sad", "tired", "low energy", "depressed", "relax", "calm"]
        goal_keywords = ["improve memory", "logical thinking", "improve reflexes", "lung capacity", "panic"]
        if is_initial_mood_selection or \
           any(mood in user_query_lower for mood in mood_keywords) or \
           any(goal in user_query_lower for goal in goal_keywords):
            prompt_instruction_for_mood = ("The user expressed/selected a mood or goal: '{user_input_for_mood}'. "
                                           "Primarily consult the 'MOOD/GOAL TO ACTIVITY/GAME MAPPING' section from the 'CerboTech Chatbot doc' in the context "
                                           "to suggest one or two suitable NeurOm app games or activities. "
                                           "If the query is very general about mindfulness or meditation not specific to NeurOm, you can draw from other provided books. "
                                           "Briefly state why each suggestion might be helpful.")

    system_prompt_template_string = (
        "You are an assistant for the NeurOm mental well-being app and a guide to mindfulness practices based on provided texts. "
        "{profile_info_if_any}" # Profile info goes here
        "{time_mention_if_relevant}"
        "{time_guidance} {mood_instruction} "
        "Your primary role is to guide users to suitable games or activities within the NeurOm app OR suggest relevant mindfulness practices from the provided books, based on their query and the retrieved context. "
        "When suggesting for a mood/goal, prioritize NeurOm app features from the 'CerboTech Chatbot doc' using its 'MOOD/GOAL TO ACTIVITY/GAME MAPPING'. "
        "Only refer to general practices from other books if the query is explicitly about them or very general and not covered by NeurOm features. "
        "If you don't know the answer from the context, or if the context doesn't fully address the user's query, "
        "say that you don't know or provide the best information you can. "
        "Keep answers concise, aiming for a maximum of three to four sentences."
        "\n\n"
        "Context:\n{context}"
    )
    user_input_for_mood_placeholder = user_query if prompt_instruction_for_mood else ""
    final_system_prompt_message = system_prompt_template_string.format(
        profile_info_if_any=profile_prompt_text,
        time_mention_if_relevant=system_prompt_time_mention,
        time_guidance=time_specific_guidance,
        mood_instruction=prompt_instruction_for_mood.format(user_input_for_mood=user_input_for_mood_placeholder) if prompt_instruction_for_mood else "",
        context="{context}"
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [("system", final_system_prompt_message), ("human", "{input}")]
    )
    try:
        question_answer_chain = create_stuff_documents_chain(LLM_INSTANCE, chat_prompt_template)
        rag_chain = create_retrieval_chain(RETRIEVER_INSTANCE, question_answer_chain)
        response = rag_chain.invoke({"input": user_query})
        bot_answer = response.get("answer", "Sorry, I couldn't formulate a response.")
        return bot_answer
    except Exception as e:
        print(f"core_logic.py: Error in RAG chain execution: {e}")
        return "I'm sorry, I encountered an issue trying to answer that. Please try again."

# --- Initialization Call ---
if not RESOURCES_INITIALIZED:
    print("core_logic.py: Top-level script execution or import, calling initialize_resources().")
    initialize_resources()

# --- Optional: Block for direct testing of this script ---
if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("Running core_logic.py directly for testing/setup...")
    if LLM_INSTANCE and RETRIEVER_INSTANCE:
        print("SUCCESS: LLM and Retriever instances appear to be initialized in core_logic.")
    else:
        print("ERROR: LLM or Retriever instance is NOT initialized after running core_logic.py.")
        if not LLM_INSTANCE: print(" - LLM_INSTANCE is None")
        if not RETRIEVER_INSTANCE: print(" - RETRIEVER_INSTANCE is None")
    print("-----------------------------------------------------")