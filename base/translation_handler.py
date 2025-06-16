from openai import OpenAI
import os
from typing import Tuple
import json

# Set up OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def is_arabic(text: str) -> bool:
    """
    Check if the given text is in Arabic.
    
    Args:
        text: The text to check
        
    Returns:
        bool: True if the text is in Arabic, False otherwise
    """
    # Check if any character is in the Arabic Unicode range
    arabic_range = range(0x0600, 0x06FF)
    return any(ord(char) in arabic_range for char in text)

def translate_to_english_variations(arabic_text: str) -> Tuple[str, bool, list]:
    """
    Translate Arabic text to English and generate query variations.
    If input is English, only generate variations.
    
    Args:
        arabic_text: The text to translate (if Arabic) or generate variations for (if English)
        
    Returns:
        Tuple of (translated_text, success, variations)
    """
    try:
        system_prompt = """
        You are a translator and query variation generator. Your task is to:
        1. If the input is in Arabic:
           - First translate it to English
           - Then generate 3 variations of the translated query
        2. If the input is in English:
           - Generate 3 variations of the query
        
        Return ONLY a JSON object with this structure:
        {
            "translated_query": "the translated query (if input was Arabic) or original query (if input was English)",
            "variations": [
                "variation 1",
                "variation 2",
                "variation 3"
            ]
        }
        
        Rules for variations:
        - Keep the same meaning but use different wording
        - Make variations more specific or more general
        - Use synonyms where appropriate
        - Maintain the original intent
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arabic_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result["translated_query"], True, result["variations"]
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return arabic_text, False, []