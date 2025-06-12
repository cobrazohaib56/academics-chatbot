import streamlit as st
import os
from dotenv import load_dotenv
import logging
from whisper import whisper_stt

# Import the base query classifier
from base.query_classifier import classify_query_sync

# Import RAG pipeline
from modules.classified_chatbot import rag_pipeline_simple

from Library.DB_endpoint import db_endpoint

# Import translation handler
from base.translation_handler import is_arabic, translate_to_english

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="wide"
)

# Define available models
AVAILABLE_MODELS = {
    "OpenAI GPT-4o Mini($0.15/M)": "gpt-4o-mini",
    "OpenAI GPT-4.1 Mini($0.40/M)": "gpt-4.1-mini",
    "OpenAI GPT-4.1-nano($0.10/M)": "gpt-4.1-nano",
    "Anthropic Claude 3 Haiku($0.25/M)": "claude-3-haiku",
    "Anthropic Claude 3 Sonnet($3/M)": "claude-3-sonnet",
    "Anthropic Claude 3 Opus($15/M)": "claude-3-opus",
    "Google Gemini 2.0 Flash($0.10/M)": "gemini-2.0-flash-001",
}

# Define which models should use OpenRouter instead of OpenAI API
OPENROUTER_MODELS = [
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "gemini-2.0-flash-001",
]

# Module map
MODULES = {
    "course_information": {
        "name": "Course Information",
        "collection_name": "admission_course_guide_json",
        "description": "Information about course content, prerequisites, credit hours, etc."
    },
    "class_schedules": {
        "name": "Class Schedules",
        "collection_name": "class_schedule_json",
        "description": "Details about when and where classes meet"
    },
    "exam_alerts": {
        "name": "Exam Data",
        "collection_name": "exam_data_json",
        "description": "Information about exam dates, deadlines, and assessments"
    },
    "study_resources": {
        "name": "Study Resources",
        "collection_name": "study_resource_json",
        "description": "Materials for studying including textbooks and online resources"
    },
    "professors": {
        "name": "Professors",
        "collection_name": "professor_data_json",
        "description": "Faculty information, office hours, and contact details"
    },
    "library": {
        "name": "Library",
        "collection_name": "library_data_json",
        "description":"Information about available books and library resources"
    }
}

def get_module_response(query: str, language: str = "English") -> str:
    """
    Route the query to the appropriate module and get a response using RAG pipeline
    
    Args:
        query: The user's question
        language: The language for the response
        
    Returns:
        Formatted response string
    """
    try:
        # Get message history for context
        message_history = []
        if "messages" in st.session_state and isinstance(st.session_state.messages, list):
            message_history = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
        
        # Check if query is in Arabic and translate if needed
        is_arabic_query = is_arabic(query)
        if is_arabic_query:
            translated_query, success = translate_to_english(query)
            if success:
                query = translated_query
                print("Translated Query for Classification: ", query)
        
        # First, classify the query to determine which module should handle it (WITH CONTEXT)
        classification, chat_summary = classify_query_sync(query, message_history)
        module_name = classification.module
        confidence = classification.confidence
        reasoning = classification.reasoning
        context_influence = classification.context_influence
        
        logger.info(f"Query classified as '{module_name}' with confidence {confidence}")
        if context_influence:
            logger.info(f"Classification was influenced by conversation context")
        
        # Special handling for library module
        if module_name == "library":
            logger.info("Processing query through library orchestrator")
            try:
                # Import the library orchestrator
                from Library import process_library_query
                
                # Process the library query through the orchestrator with the model and context
                result = process_library_query(query, chat_summary)

                if result is not None:
                    response = result["response"]
                    return response
            except Exception as e:
                logger.error(f"Error in library orchestrator: {e}", exc_info=True)
                return "Sorry, there was an error processing your library query."
        
        # Special handling for professors module - check if it's a meeting request
        if module_name == "professors":
            logger.info("Processing query through professors orchestrator")
            try:
                # Import the professor orchestrator
                from modules.professors import process_professor_query
                
                # Check if query is about scheduling a meeting
                if module_name in MODULES:
                    collection_name = MODULES[module_name]["collection_name"]
                else:
                    collection_name = "professor_data_json"
                
                result = process_professor_query(query, collection_name,chat_summary)
                
                # If it's a meeting request, return the generated response
                if result is not None and result.get("is_meeting_request", False):
                    response = result["response"]
                    
                    # Add debug info if in development
                    if os.getenv("APP_ENV") == "development":
                        debug_info = f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\n"
                        debug_info += f"Context influenced: {context_influence}\n"
                        debug_info += f"Meeting intent detected (confidence: {result.get('confidence', 0):.2f})\n"
                        debug_info += f"Reasoning: {result.get('reasoning', '')}"
                        response += debug_info
                    
                    return response
                
                # If not a meeting request, continue with RAG pipeline below
                logger.info("Not a meeting request, using regular RAG pipeline")
            except Exception as e:
                logger.error(f"Error in professors orchestrator: {e}", exc_info=True)
                # Continue with RAG pipeline if there's an error
        if module_name == "exam_alerts":
            logger.info("Processing query through exam orchestrator")
            try:
                # Import the professor orchestrator
                from modules.examdata.exam_data_orchestrator import process_exam_query
                
                # Check if query is about scheduling a meeting
                if module_name in MODULES:
                    collection_name = MODULES[module_name]["collection_name"]
                else:
                    collection_name = "exam_data_json"
                
                result = process_exam_query(query, collection_name,chat_summary)
                
                # If it's a meeting request, return the generated response
                if result is not None and result.get("is_notification_request", False):
                    response = result["response"]
                    
                    return response
                
                # If not a meeting request, continue with RAG pipeline below
                logger.info("Not a meeting request, using regular RAG pipeline")
            except Exception as e:
                logger.error(f"Error in exam orchestrator: {e}", exc_info=True)
                # Continue with RAG pipeline if there's an error
        # For other modules or non-meeting professor queries, use the RAG pipeline
        # Determine which collection to use
        if module_name in MODULES:
            collection_name = MODULES[module_name]["collection_name"]
        else:
            # Fallback to study resources if module not found
            logger.warning(f"Module '{module_name}' not found, falling back to study resources")
            collection_name = MODULES["study_resources"]["collection_name"]
        
        # Get the selected model from session state
        model = st.session_state.get("model", "openai/gpt-4o-mini")
        
        # Check if we need to prepend the provider name for OpenRouter models
        if st.session_state.get("use_openrouter", False):
            # For OpenRouter models that aren't OpenAI, need to prepend provider
            if "gpt" not in model:
                if "claude" in model:
                    model = f"anthropic/{model}"
                elif "gemini" in model:
                    model = f"google/{model}"
            else:
                model = f"openai/{model}"
        
        # Use RAG pipeline to get response (with context summary)
        result = rag_pipeline_simple(query, collection_name, model, message_history=message_history, chat_summary=chat_summary, is_arabic=is_arabic_query)
        response = result["response"]
        
        # Add debug info if in development
        debug_info = ""
        if os.getenv("APP_ENV") == "development":
            debug_info = f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\n"
            debug_info += f"Context influenced: {context_influence}\n"
            debug_info += f"Reasoning: {reasoning}"
        
        return response + debug_info
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى لاحقًا."
        else:
            return "I'm sorry, an error occurred while processing your question. Please try again later."

# Main app
def main():
    # Custom CSS to remove button gaps and align input fields
    st.markdown("""
    <style>
    /* Remove padding around buttons */
    div.stButton > button {
        margin: 0;
        padding: 0.4rem 1rem;
    }
    
    /* Adjust button container spacing */
    div.row-widget.stButton {
        padding: 0;
        margin: 0;
    }
    
    /* Align input fields vertically */
    .stChatInput, [data-testid="stChatInput"] {
        margin-bottom: 0;
    }
    
    /* Ensure consistent spacing for columns */
    div[data-testid="column"] {
        padding-top: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    
    /* Align whisper component */
    .whisper-stt-container {
        display: flex;
        align-items: flex-end;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 University Assistant")
    
    # Main content area (upper section)
    main_area = st.container()
    
    # Chat and controls area (bottom section)
    bottom_container = st.container()
    
    # Main content area with app information
    with main_area:
        # Sidebar with settings only (no audio input)
        with st.sidebar:
            st.subheader("Settings")
            language = st.selectbox("Language / اللغة", ["English", "Arabic"])
            
            # Model selection dropdown - default to first model
            model_options = list(AVAILABLE_MODELS.keys())
            selected_model_name = st.selectbox(
                "Select AI Model",
                model_options,
                index=0
            )
            
            selected_model_id = AVAILABLE_MODELS[selected_model_name]
            
            # Store model ID and whether it's an OpenRouter model in session state
            if "model" not in st.session_state or st.session_state.model != selected_model_id:
                st.session_state.model = selected_model_id
                st.session_state.use_openrouter = selected_model_id in OPENROUTER_MODELS
            
            # Clear conversation button
            if st.button("Clear Conversation"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        st.subheader("University Information Assistant")
        st.write("Ask me anything about courses, schedules, exams, faculty, library resources, admission, or tuition!")
        
        # Display model being used
        st.caption(f"Currently using: {selected_model_name}")
        
        # Initialize chat history in session state if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
    spinner_container = st.container()
    # Bottom container for chat history and input controls
    with bottom_container:
        # Display a separator line
        st.markdown("---")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Single unified input container with columns for text and audio
        input_container = st.container()
        
        with input_container:
            # Create columns for text input and audio input with better alignment
            col_text, col_audio = st.columns([9, 1])
            
            # Text input column
            with col_text:
                prompt = st.chat_input("Ask me a question...")
            
            # Audio input column
            with col_audio:
                
                # Create a container to help with alignment
                audio_wrapper = st.container()
                
                with audio_wrapper:
                    # Initialize audio processing state
                    if "audio_processed" not in st.session_state:
                        st.session_state.audio_processed = False
                    if "last_audio_text" not in st.session_state:
                        st.session_state.last_audio_text = ""
                    
                    # Determine language for whisper based on selected language
                    whisper_language = 'ar' if language.lower() == 'arabic' else 'en'
                    
                    # Use whisper_stt for audio transcription
                    transcribed_text = whisper_stt(language=whisper_language)
                    
                    # Only process if we have new audio text and haven't processed it yet
                    if (transcribed_text and 
                        transcribed_text != st.session_state.last_audio_text and 
                        not st.session_state.audio_processed):
                        
                        # Mark as processed and store the text
                        st.session_state.audio_processed = True
                        st.session_state.last_audio_text = transcribed_text
                        
                        # Add to session state message history
                        st.session_state.messages.append({"role": "user", "content": transcribed_text})
                        
                        # Use the shared spinner container for audio processing too
                        with spinner_container:
                            with st.spinner("Generating response..."):
                                response = get_module_response(transcribed_text, language=language)
                        
                        # Update message history with response
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Reset the processed flag after a short delay to allow new recordings
                        st.session_state.audio_processed = False
                        
                        # Rerun to show new messages
                        st.rerun()
    
        # Create a separate container for spinner outside the input area
        spinner_container = st.container()
    
    # Handle text input
    if prompt:
        # Add to session state message history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show spinner in the dedicated container outside input area
        with spinner_container:
            with st.spinner("Generating response..."):
                # Generate response
                response = get_module_response(prompt, language=language)
        
        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI with the new messages
        st.rerun()

if __name__ == "__main__":
    main()