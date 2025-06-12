from openai import OpenAI
import os
from typing import Tuple

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

def translate_to_english(arabic_text: str) -> Tuple[str, bool]:
    """
    Translate Arabic text to English using OpenAI.
    
    Args:
        arabic_text: The Arabic text to translate
        
    Returns:
        Tuple of (translated_text, success)
    """
    try:
        system_prompt = """
        You are a translator. Your task is to translate Arabic text to English.
        Return ONLY the translated text without any additional explanation.
        
        NOTE: If the written text in Arabic does not makes sense, like if the order of the words seems off, you need to
        readjust its order. 
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arabic_text}
            ],
            temperature=0.1
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text, True
        
    except Exception as e:
        print(f"Error translating text: {e}")
        return arabic_text, False