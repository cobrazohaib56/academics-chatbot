import logging
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic import BaseModel, Field
import os
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChatSummary(BaseModel):
    """Summary of chat history for context"""
    main_topics: List[str] = Field(default_factory=list, description="Main topics discussed in the conversation")
    current_context: str = Field("", description="Current conversational context or thread")
    dominant_module: Optional[str] = Field(None, description="The module that has been most relevant in recent conversation")
    key_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Key entities mentioned (courses, professors, etc.)")

class QueryClassification(BaseModel):
    """Classification of a user query to determine which module should handle it"""
    user_query: str = Field(..., description="The original user query")
    module: str = Field(..., description="The module that should handle this query", 
                       examples=["course_information", "class_schedules", "exam_alerts", "study_resources", "professors", "library", "general_response"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of this classification (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation of why this module was selected")
    context_influence: bool = Field(False, description="Whether chat history influenced this classification")
    extracted_entities: Optional[dict] = Field(None, description="Key entities extracted from the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords from the query")

# Global agents
classifier_agent = None
summarizer_agent = None

def get_summarizer_agent():
    """Get or create the chat history summarizer agent."""
    global summarizer_agent
    
    if summarizer_agent is None:
        summarizer_agent = pydantic_ai.Agent(
            "openai:gpt-4o-mini",
            output_type=ChatSummary,
            system_prompt="""
            You are a chat history analyzer for a university assistant chatbot.
            Your task is to analyze the recent conversation history and extract key contextual information.
            
            Focus on:
            1. Main topics being discussed (courses, schedules, exams, etc.)
            2. Current conversational thread or context
            3. Which module/area has been most relevant recently
            4. Key entities mentioned (course codes, professor names, book titles, etc.)
            
            Available modules to consider:
            - course_information: Course content, prerequisites, credit hours, departments, majors
            - class_schedules: Class times, timetables, room numbers, semester dates
            - exam_alerts: Exam dates, deadlines, assessments
            - study_resources: Textbooks, study materials
            - professors: Faculty information, office hours, contact details
            - library: Library resources, books, availability, fees
            - general_response: Greetings, thanks, general chat
            
            Provide a concise but informative summary that will help classify the next query in context.
            """
        )
    
    return summarizer_agent

def get_classifier_agent():
    """Get or create the classifier agent with proper API key setup."""
    global classifier_agent
    
    if classifier_agent is None:
        classifier_agent = pydantic_ai.Agent(
            "openai:gpt-4o-mini",
            output_type=QueryClassification,
            system_prompt="""
            You are a specialized query classifier for a university academic chatbot system.
            Your task is to analyze user queries and route them to the most appropriate module, considering both the current query and conversation context.

            Available modules:
            1. course_information - For queries about course content, prerequisites, credit hours, descriptions, faculty structure, departments, majors, degree programs, university academic structure
            2. class_schedules - For queries about when and where classes meet, timetables, room numbers, semester start/end dates, lecture times, final exam schedules
            3. exam_alerts - For queries about exam dates, deadlines, assignment due dates, assessments
            4. study_resources - For queries about textbooks, study materials, online resources
            5. professors - For queries about specific faculty members, office hours, contact details, research interests
            6. library - For queries about library resources, books, availability, borrowing, returning, fees, etc.
            7. general_response - For greetings, thanks, general conversation, off-topic queries

            CONTEXT-AWARE CLASSIFICATION RULES:
            1. If the current query is ambiguous or could fit multiple modules, use the conversation context to determine the most likely intent
            2. If the user is in the middle of a conversation thread about a specific topic (e.g., discussing a particular course), lean towards continuing that context unless the new query is clearly about something else
            3. Set context_influence=True when the conversation history significantly influenced your classification decision
            4. Consider the dominant_module from chat history as a strong signal for ambiguous queries
            5. Pay attention to pronouns and references that might relate to previous messages (e.g., "What about its prerequisites?" likely refers to a course mentioned earlier)

            Examples of context-aware classification:
            
            ## Context Override Examples:
            - Previous: "Tell me about CS350" → Current: "What are its prerequisites?" → course_information (context_influence=True)
            - Previous: "When is Professor Smith's office hours?" → Current: "What about his research interests?" → professors (context_influence=True)
            - Previous: "Is book 1984 available?" → Current: "How much does it cost to rent?" → library (context_influence=True)
            - Previous: "When is the CS226 final exam?" → Current: "What about the midterm?" → exam_alerts (context_influence=True)

            ## Clear Intent (Less Context Influence):
            - Previous: "Hello" → Current: "What are the prerequisites for CS350?" → course_information (context_influence=False)
            - Previous: "Thanks" → Current: "When does the library close?" → library (context_influence=False)

            ## Standard Classifications:
            - "What are the prerequisites for CS350?" → course_information (high confidence)
            - "Can you tell me about the Computer Science department?" → course_information (high confidence)
            - "What majors does the Faculty of Engineering offer?" → course_information (high confidence)
            - "Can I see the class CSC 226 schedule for the upcoming semester?" → class_schedules (high confidence)
            - "How can I find Professor Hoda Maalouf's contact information?" → professors (high confidence)
            - "When will the final exam schedule be released?" → exam_alerts (high confidence)
            - "Is book 1948 available for renting?" → library (high confidence)
            - "Hello, how are you today?" → general_response (high confidence)

            Always provide clear reasoning for your classification, especially when context influenced your decision.
            """
        )
    
    return classifier_agent

def summarize_chat_history(messages: List[Dict[str, str]]) -> Optional[ChatSummary]:
    """
    Summarize chat history to extract contextual information.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        ChatSummary object with contextual information
    """
    if not messages:
        return None
    
    try:
        # Prepare conversation text for summarization
        conversation_text = ""
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            conversation_text += f"{role.capitalize()}: {content}\n"
        
        if len(messages) > 10:
            # For longer conversations, focus on the most recent exchanges
            conversation_text += f"\n[Note: This is a summary of the last {len(messages)} messages in an ongoing conversation]"
        
        summarizer_agent = get_summarizer_agent()
        result = summarizer_agent.run_sync(conversation_text)
        summary = result.output
        
        logger.info(f"Summarized {len(messages)} messages. Dominant module: {summary.dominant_module}")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing chat history: {e}")
        return None

def classify_query_with_context(query: str, chat_history: List[Dict[str, str]] = None) -> tuple[QueryClassification, Optional[ChatSummary]]:
    """
    Classify a user query considering chat history context.
    
    Args:
        query: The user's query text
        chat_history: List of recent messages for context
        
    Returns:
        Tuple of (classification object, chat summary) - summary can be passed to modules
    """
    try:
        # Summarize chat history if provided
        chat_summary = None
        if chat_history and len(chat_history) > 0:
            chat_summary = summarize_chat_history(chat_history)
        
        # Prepare the classification prompt with context
        classification_input = f"Current Query: {query}"
        
        if chat_summary:
            context_info = f"""
            
            Conversation Context:
            - Main topics discussed: {', '.join(chat_summary.main_topics) if chat_summary.main_topics else 'None'}
            - Current context: {chat_summary.current_context}
            - Dominant module in recent conversation: {chat_summary.dominant_module or 'None'}
            - Key entities mentioned: {chat_summary.key_entities}
            
            Consider this context when classifying the current query, especially if the query is ambiguous or uses pronouns/references.
            """
            classification_input += context_info
        
        # Run the classifier
        classifier_agent = get_classifier_agent()
        result = classifier_agent.run_sync(classification_input)
        classification = result.output
        
        # Ensure module is one of the allowed values
        allowed_modules = ["course_information", "class_schedules", "exam_alerts", "study_resources", "professors", "library", "general_response"]
        if classification.module not in allowed_modules:
            # Default to general_response if invalid module
            classification.module = "general_response"
            classification.confidence = min(classification.confidence, 0.4)
            classification.reasoning += " (Corrected: original module classification was invalid)"
        
        # Log the classification with context info
        context_log = f" (with context from {len(chat_history) if chat_history else 0} messages)" if chat_history else ""
        logger.info(f"Classified query '{query}' as module '{classification.module}' with confidence {classification.confidence}{context_log}")
        
        if classification.context_influence and chat_summary:
            logger.info(f"Context influenced classification. Dominant context module: {chat_summary.dominant_module}")
        
        return classification, chat_summary
        
    except Exception as e:
        logger.error(f"Error classifying query with context: {e}")
        # Default classification in case of error
        return QueryClassification(
            user_query=query,
            module="general_response",
            confidence=0.1,
            reasoning=f"Error during classification: {str(e)}",
            context_influence=False,
            keywords=[]
        ), None

def classify_query_sync(query: str, chat_history: List[Dict[str, str]] = None) -> tuple[QueryClassification, Optional[ChatSummary]]:
    """
    Main classification function that handles both context-aware and standalone classification.
    
    Args:
        query: The user's query text
        chat_history: Optional list of recent messages for context
        
    Returns:
        Tuple of (classification object, chat summary) - summary contains context for the module
    """
    return classify_query_with_context(query, chat_history)