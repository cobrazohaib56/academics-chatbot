import re
from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
import os
from Library.DB_endpoint import db_endpoint
from modules.classified_chatbot import rag_pipeline_simple
import streamlit as st
import sqlite3
# Set up OpenRouter client for intent detection
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def detect_query_intent(query: str,chat_summary=None) -> Tuple[str, float, str]:
    """
    Detect the intent of the library query (booking, information, etc.).
    
    Args:
        query: The user's question
        
    Returns:
        Tuple containing:
        - Intent type ('booking', 'information', or 'unknown')
        - Confidence score (0-1)
        - Reasoning for the classification
    """
    system_prompt = """
    You are an intent classifier for a university library assistant. Determine if the query is about:
    1. Booking/reserving a book
    2. General information about books (availability of a book, author, etc.)
    3. Mentioned is the Chat Summary to determine the intent of the query
        -{chat_summary}


    Return ONLY a JSON object with the following structure:
    {
        "intent": "booking" or "information",
        "confidence": <float between 0 and 1>,
        "reasoning": "<brief explanation>"
    }
    
    Examples of BOOKING QUERIES (return "booking"):
    - "I want to reserve the book Clean Code"
    - "Can I book Harry Potter?"
    - "I'd like to check out this book"
    - "Reserve this book for me"
    - "Can you reserve Rule the World book?"
    
    Examples of INFORMATION QUERIES (return "information"):
    - "Do you have any books on machine learning?"
    - "Is 'Clean Code' available in the library?"
    - "Which books by Robert Martin do you have?"
    - "Are there any books on computer science still available?"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # OpenRouter model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"Intent classification result: {result}")  # Debug print
        
        # Convert the string response to a Python dictionary
        import json
        parsed_response = json.loads(result)
        
        return (
            parsed_response["intent"],
            parsed_response["confidence"],
            parsed_response["reasoning"]
        )
    except Exception as e:
        print(f"Error classifying query intent: {e}")
        # Default to information if there's an error
        return "information", 0.6, f"Error during classification, defaulting to information query: {str(e)}"

def extract_book_name(query: str,chat_summary=None) -> str:
    """
    Extract the book name from a booking query.
    
    Args:
        query: The user's booking request
        
    Returns:
        The extracted book name or None if not found
    """
    system_prompt = """
    Intelligently Extract the book name from the user's query or chat summary as mentioned. Return ONLY the book name without any additional text or formatting.
    If no book name is found, return "unknown".
    Check the Previous chat summary is there any talk about the book or no.
    Mentioned is the attached chat summary
    -{chat_summary}
    
    Examples:
    Query: "I want to reserve the book Clean Code"
    Response: "Clean Code"
    
    Query: "Can I book Harry Potter?"
    Response: "Harry Potter"
    
    Query: "Reserve this book"
    Response: "unknown"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        
        book_name = response.choices[0].message.content.strip()
        return book_name if book_name.lower() != "unknown" else None
        
    except Exception as e:
        print(f"Error extracting book name: {e}")
        return None

def check_book_availability(book_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a book is available for booking.
    
    Args:
        book_name: Name of the book to check
        
    Returns:
        Tuple of (is_available, book_info)
    """
    try:
        # Connect to the database
        conn = sqlite3.connect("library.db")
        cursor = conn.cursor()
        
        # Query to check book availability
        query = """
        SELECT * FROM Library 
        WHERE LOWER(Book_Name) LIKE LOWER(?) AND (Booked IS NULL OR Booked = '')
        """
        
        cursor.execute(query, (f"%{book_name}%",))
        result = cursor.fetchone()
        
        if result:
            # Get column names
            columns = [description[0] for description in cursor.description]
            # Create dictionary of book info
            book_info = dict(zip(columns, result))
            return True, book_info
        
        return False, None
        
    except Exception as e:
        print(f"Error checking book availability: {e}")
        return False, None
    finally:
        if 'conn' in locals():
            conn.close()

def book_available_book(book_name: str) -> Tuple[bool, str]:
    """
    Book an available book.
    
    Args:
        book_name: Name of the book to book
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Connect to the database
        conn = sqlite3.connect("library.db")
        cursor = conn.cursor()
        
        # First check if the book exists and is available
        check_query = """
        SELECT Book_Name FROM Library 
        WHERE LOWER(Book_Name) LIKE LOWER(?) AND (Booked IS NULL OR Booked = '')
        """
        cursor.execute(check_query, (f"%{book_name}%",))
        result = cursor.fetchone()
        
        if not result:
            return False, f"Could not find '{book_name}' or it is already booked."
        
        # Update query to mark book as booked
        update_query = """
        UPDATE Library 
        SET Booked = 'Yes'
        WHERE LOWER(Book_Name) LIKE LOWER(?) AND (Booked IS NULL OR Booked = '')
        """
        
        cursor.execute(update_query, (f"%{book_name}%",))
        conn.commit()
        print(f"Rows updated: {cursor.rowcount}")  # Debug print
        
        if cursor.rowcount > 0:
            # Get the updated book info
            cursor.execute("SELECT * FROM Library WHERE LOWER(Book_Name) LIKE LOWER(?)", (f"%{book_name}%",))
            book_info = cursor.fetchone()
            columns = [description[0] for description in cursor.description]
            book_dict = dict(zip(columns, book_info))
            
            return True, f"Successfully booked '{book_name}'. The book is now marked as booked."
        else:
            return False, f"Could not book '{book_name}'. It may no longer be available."
        
    except Exception as e:
        print(f"Error booking book: {e}")
        return False, f"Error booking book: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

def format_library_response(query: str, results: Dict[str, Any], is_arabic: bool = False) -> str:
    """
    Use an LLM to format the library query results into a well-structured, readable response.
    
    Args:
        query: The original user query
        results: The database query results
        is_arabic: Whether the response should be in Arabic
        
    Returns:
        A well-formatted, conversational response string
    """
    try:
        # If this is a booking response, handle it differently
        if results.get("is_booking", False):
            return results.get("response", "Sorry, there was an error processing your booking request.")
        
        # Extract data from results
        data = results.get("results", [])
        sql_query = results.get("sql", "")
        
        # Prepare the system prompt
        system_prompt = """
        You are a library assistant that formats database query results into a conversational response.
        Your task is to create a clean, well-formatted response following these rules:
        
        1. Format all prices with $ symbol and 2 decimal places (e.g., $6.11)
        2. Present prices in a clear format: "Booking fee: $X.XX, Selling price: $Y.YY"
        3. Keep the response in a single, well-structured paragraph
        4. Use proper spacing and punctuation
        5. Do not use any special characters, markdown, or formatting
        6. Make the response natural and conversational
        7. If multiple books are found, list them clearly with their details
        
        Example of good formatting in English:
        "Yes, we have the book '1984' by George Orwell. It is a dystopian novel that explores themes of totalitarianism and surveillance. The booking fee is $6.11, and the selling price is $31.22."
        
        Example of good formatting in Arabic:
        "نعم، لدينا كتاب '1984' لجورج أورويل. إنه رواية ديستوبية تستكشف موضوعات الشمولية والمراقبة. رسوم الحجز هي 6.11 دولار، وسعر البيع هو 31.22 دولار."
        """
        
        # Add language instruction
        if is_arabic:
            system_prompt += "\n\nPlease provide the response in Arabic, maintaining the same formatting rules but using Arabic text and right-to-left formatting where appropriate."
        else:
            system_prompt += "\n\nPlease provide the response in English."
        
        # Prepare the context with query results
        context = f"User query: {query}\n\nSQL query used: {sql_query}\n\nQuery results:\n"
        
        # Add result data
        if "error" in results:
            context += f"Error: {results['error']}"
        elif not data:
            context += "No books found matching the query."
        else:
            context += f"Found {len(data)} books:\n\n"
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    context += f"Book {i+1}:\n"
                    for key, value in item.items():
                        if value is not None:
                            context += f"- {key}: {value}\n"
                    context += "\n"
                else:
                    context += f"Book {i+1}: {str(item)}\n\n"
        
        # Generate formatted response using OpenRouter
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error formatting library response: {e}")
        # Return a simple formatted response as fallback
        if "error" in results:
            if is_arabic:
                return f"عذراً، حدث خطأ أثناء البحث عن الكتب: {results['error']}"
            return f"Sorry, I encountered an error while searching for books: {results['error']}"
        
        data = results.get("results", [])
        if not data:
            if is_arabic:
                return "لم أتمكن من العثور على أي كتب تطابق استفسارك. ربما يمكنك تجربة كلمات مفتاحية مختلفة أو السؤال عن كتاب آخر؟"
            return "I couldn't find any books matching your query. Perhaps try different keywords or ask about another book?"
        
        # Let the LLM format the fallback response as well
        fallback_context = f"Format these book results in a conversational way:\n\n"
        for i, item in enumerate(data):
            if isinstance(item, dict):
                fallback_context += f"Book {i+1}:\n"
                for key, value in item.items():
                    if value is not None:
                        fallback_context += f"- {key}: {value}\n"
            else:
                fallback_context += f"Book {i+1}: {str(item)}\n"
        
        fallback_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fallback_context}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return fallback_response.choices[0].message.content.strip()

def process_library_query(query: str, is_arabic_query: bool, chat_summary=None) -> Dict[str, Any]:
    """
    Process a query related to the library, determining if it should use the database
    endpoint or be handled by a general response using the RAG pipeline.
    
    Args:
        query: The user's question
        is_arabic_query: Whether the query is in Arabic
        chat_summary: Optional chat history summary
        
    Returns:
        A dictionary with the response and metadata
    """
    try:
        # First detect the intent of the query
        intent, confidence, reasoning = detect_query_intent(query)
        print(f"Detected intent: {intent} with confidence {confidence}")  # Debug print
        
        # Handle booking intent
        if intent == "booking" and confidence > 0.6:
            # Extract book name from query
            book_name = extract_book_name(query, chat_summary)
            print(f"Extracted book name: {book_name}")  # Debug print
            
            if not book_name:
                return {
                    "response": "لم أتمكن من تحديد الكتاب الذي تريد حجزه. يرجى تحديد اسم الكتاب." if is_arabic_query else "I couldn't identify which book you want to book. Please specify the book name.",
                    "is_booking": True,
                    "success": False
                }
            
            # Check if book is available
            is_available, book_info = check_book_availability(book_name)
            print(f"Book availability: {is_available}")  # Debug print
            
            if not is_available:
                return {
                    "response": f"عذراً، الكتاب '{book_name}' غير متاح للحجز في الوقت الحالي." if is_arabic_query else f"I'm sorry, but '{book_name}' is not available for booking at the moment.",
                    "is_booking": True,
                    "success": False
                }
            
            # Book the book
            success, message = book_available_book(book_name)
            print(f"Booking result: {success} - {message}")  # Debug print
            
            if success:
                response = f"تم حجز الكتاب '{book_name}' بنجاح. الكتاب الآن محجوز." if is_arabic_query else message
            else:
                response = f"عذراً، لم نتمكن من حجز الكتاب '{book_name}'. قد لا يكون متاحاً الآن." if is_arabic_query else message
            
            return {
                "response": response,
                "is_booking": True,
                "success": success,
                "book_info": book_info
            }
        
        # For information queries, use the existing DB endpoint
        results = db_endpoint(query)
        
        # Format the results using LLM
        formatted_response = format_library_response(query, results, is_arabic_query)
        
        return {
            "response": formatted_response,
            "is_booking": False,
            "confidence": confidence,
            "reasoning": reasoning,
            "sql": results.get("sql"),
            "results": results
        }
    except Exception as e:
        print(f"Error in process_library_query: {e}")
        return {
            "response": f"Sorry, there was an error processing your request: {str(e)}",
            "is_booking": False,
            "success": False
        }
