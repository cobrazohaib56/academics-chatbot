from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
import os
import streamlit as st
# Set up OpenAI client for intent detection

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def detect_exam_notification_intent(query: str, chat_summary=None) -> Tuple[bool, float, str]:
    """
    Detect if the query is about setting up exam notifications.
    
    Args:
        query: The user's question
        chat_summary: Summary of previous conversation
        
    Returns:
        Tuple containing:
        - Boolean indicating if the intent is to set exam notifications
        - Confidence score (0-1)
        - Reasoning for the classification
        - Exam/subject name
    """
    system_prompt = f"""
    You are an intent classifier for a university assistant. Determine if the query is about setting up exam notifications or reminders or asking for general questions.
    
    1. Consider the attached summary of the previous conversation to determine the intention:
        - {chat_summary}

    
    Return ONLY a JSON object with the following structure:
    {{
        "is_notification_intent": true/false,
        "confidence": <float between 0 and 1>,
        "reasoning": "<brief explanation>",
        "exam_subject": "<name of the exam/subject>"
    }}
    
    Examples of EXAM NOTIFICATION REQUESTS (return true):
    - "I want to set a notification for my Math exam" → {{"is_notification_intent": true, "confidence": 0.95, "reasoning": "Explicitly mentions setting notification for exam"}}
    - "Can I get reminders for the Physics final?" → {{"is_notification_intent": true, "confidence": 0.9, "reasoning": "Asks about getting reminders for exam"}}
    - "How do I set up alerts for my Chemistry test?" → {{"is_notification_intent": true, "confidence": 0.9, "reasoning": "Asking about setting up exam alerts"}}
    - "Remind me about the Biology exam" → {{"is_notification_intent": true, "confidence": 0.85, "reasoning": "Request to be reminded about exam"}}
    - "Set notification for Computer Science midterm" → {{"is_notification_intent": true, "confidence": 0.95, "reasoning": "Direct request to set exam notification"}}
    
    Examples of GENERAL QUERIES (return false):
    - "What is the date of the Math exam?" → {{"is_notification_intent": false, "confidence": 0.8, "reasoning": "Asking about exam date, not setting notifications"}}
    - "Tell me about the Physics exam syllabus" → {{"is_notification_intent": false, "confidence": 0.95, "reasoning": "Asking about exam syllabus, not notifications"}}
    - "What topics are covered in Chemistry exam?" → {{"is_notification_intent": false, "confidence": 0.95, "reasoning": "Asking about exam topics, not notifications"}}
    - "Where is the Biology exam hall located?" → {{"is_notification_intent": false, "confidence": 0.9, "reasoning": "Asking about exam location, not notifications"}}
    - "What is the duration of Computer Science exam?" → {{"is_notification_intent": false, "confidence": 0.95, "reasoning": "Asking about exam duration, not notifications"}}
    - "Who is the instructor for the History exam?" → {{"is_notification_intent": false, "confidence": 0.9, "reasoning": "Asking about instructor, not notifications"}}
    
    Only classify as a notification intent if the user is specifically asking to set up, create, or arrange notifications/reminders/alerts for exams. General information queries about exams should be classified as false.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Convert the string response to a Python dictionary
        import json
        parsed_response = json.loads(result)
        print("="*50)
        print( parsed_response["is_notification_intent"],
            parsed_response["confidence"],
            parsed_response["reasoning"],
            parsed_response["exam_subject"])
        print("="*50)

        return (
            parsed_response["is_notification_intent"],
            parsed_response["confidence"],
            parsed_response["reasoning"],
            parsed_response["exam_subject"]
        )
    except Exception as e:
        print(f"Error classifying exam notification intent: {e}")
        # Default to False if there's an error
        return False, 0.0, f"Error during classification: {str(e)}", None

def setup_exam_notification(exam_subject: Optional[str] = None, is_arabic: bool = False) -> Dict[str, Any]:
    """
    Set up exam notification for the specified subject.
    This is a placeholder that will be replaced with actual notification system later.
    
    Args:
        exam_subject: Optional name of the exam/subject
        is_arabic: Whether the response should be in Arabic
        
    Returns:
        Dictionary with notification setup information
    """
    # Extract exam subject name if provided
    subject_display = f" for {exam_subject}" if exam_subject else ""
    
    # Placeholder demo notification setup - this would be replaced with actual notification system
    notification_id = f"exam-notification-{exam_subject.lower().replace(' ', '-')}" if exam_subject else "exam-notification-general"
    
    if is_arabic:
        subject_display = f" لـ {exam_subject}" if exam_subject else ""
        return {
            "notification_id": notification_id,
            "exam_subject": exam_subject,
            "status": "نشط",
            "message": f"تم إعداد الإشعارات بنجاح{subject_display}. "
                      f"ستتلقى تذكيرات حول جدول امتحاناتك، والتواريخ المهمة، والمواعيد النهائية. "
                      f"هذا نظام إشعارات تجريبي سيتم استبداله بنظام إشعارات حقيقي في المستقبل."
        }
    
    return {
        "notification_id": notification_id,
        "exam_subject": exam_subject,
        "status": "active",
        "message": f"Notification has been successfully set up{subject_display}. "
                   f"You will receive reminders about your exam schedule, important dates, and deadlines. "
                   f"This is a demo notification system that will be replaced with a real notification service in the future."
    }

def process_exam_query(query: str, collection_name: str, chat_summary=None, is_arabic: bool = False) -> Dict[str, Any]:
    """
    Process a query related to exams, determining if it's about setting up
    notifications or a general question.
    
    Args:
        query: The user's question
        collection_name: The name of the Qdrant collection for exams
        chat_summary: Summary of previous conversation
        is_arabic: Whether the response should be in Arabic
        
    Returns:
        A dictionary with the response and metadata
    """
    # Detect if this is an exam notification intent
    is_notification, confidence, reasoning, exam_subject = detect_exam_notification_intent(query, chat_summary)
    
    # If it's a notification request with reasonable confidence
    if is_notification and confidence > 0.7:
        # Set up exam notification
        notification_info = setup_exam_notification(exam_subject, is_arabic)
        
        if is_arabic:
            response = f"ممتاز! لقد قمت بإعداد إشعارات الامتحان{' لـ ' + exam_subject if exam_subject else ''}. " + notification_info["message"]
            response += f"\n\nمعرف الإشعار: {notification_info['notification_id']}"
            response += f"\nالحالة: {notification_info['status'].upper()}"
        else:
            response = f"Perfect! I've set up exam notifications{' for ' + exam_subject if exam_subject else ''}. " + notification_info["message"]
            response += f"\n\nNotification ID: {notification_info['notification_id']}"
            response += f"\nStatus: {notification_info['status'].upper()}"
        
        return {
            "response": response,
            "notification_info": notification_info,
            "is_notification_request": True,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    # For generic questions, return None to indicate the main RAG pipeline should be used
    return None