# University Assistant Chatbot

A comprehensive university assistant chatbot system with RAG (Retrieval Augmented Generation) capabilities to answer questions about courses, schedules, faculty, admission requirements, and more.

## Features

- **Hybrid Search**: Combines vector similarity with keyword and heading matching to provide more accurate responses
- **Document Structure Awareness**: Preserves document hierarchies and headings for contextual understanding
- **Multi-language Support**: Full support for English and Arabic interfaces with automatic language detection
- **Various Data Source Handling**: Process PDFs, DOCXs, TXTs, CSVs, and Excel files
- **Pre-processed Data Support**: Can ingest both raw and pre-processed structured data
- **Rich Metadata**: Extracts and utilizes headings, keywords, and entities for improved retrieval
- **Multiple AI Models**: Support for various models via OpenRouter and OpenAI, with a convenient dropdown selector
- **Conversation History**: Maintains chat history for contextual understanding in conversations
- **Module-Specific Processing**: Specialized handling for different types of queries (library, professors, exams)

## System Architecture

### Core Components

1. **Main Application (`app.py`)**
   - Streamlit-based user interface
   - Language detection and translation
   - Query classification and routing
   - Model selection and management
   - Conversation history management

2. **Query Classification (`base/query_classifier.py`)**
   - Determines the appropriate module for each query
   - Considers conversation context
   - Provides confidence scores and reasoning

3. **RAG Pipeline (`modules/classified_chatbot.py`)**
   - Implements hybrid search
   - Handles context management
   - Generates responses using selected AI model

### Specialized Modules

1. **Library Module (`Library/library_orchestrator.py`)**
   - Book availability checking
   - Booking functionality
   - Price and fee formatting
   - Bilingual response generation

2. **Professor Module (`modules/professors/professor_orchestrator.py`)**
   - Meeting scheduling intent detection
   - Professor information retrieval
   - Meeting link generation
   - Bilingual response handling

3. **Exam Module (`modules/examdata/exam_data_orchestrator.py`)**
   - Exam notification setup
   - Exam information retrieval
   - Notification management
   - Bilingual response generation

### Language Support

1. **Translation Handler (`base/translation_handler.py`)**
   - Arabic language detection
   - Query translation to English
   - Response translation to Arabic

2. **Bilingual Response Generation**
   - All modules support both English and Arabic
   - Maintains formatting consistency
   - Handles right-to-left text appropriately

## Data Flow

1. **Query Processing**
   ```
   User Query → Language Detection → Translation (if needed) → Query Classification → Module Routing
   ```

2. **Response Generation**
   ```
   Module Processing → RAG Pipeline → Response Generation → Language Formatting → User Response
   ```

3. **Specialized Module Flow**
   ```
   Query → Intent Detection → Specialized Processing → Response Generation → Language Formatting
   ```

## Setup and Usage

### Environment Setup

1. Create a `.env` file based on the `env_template.txt`
   - OpenAI API key
   - OpenRouter API key
   - Other configuration settings

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

```
streamlit run app.py
```

## Module-Specific Features

### Library Module
- Book availability checking
- Booking system integration
- Price and fee management
- Book information retrieval

### Professor Module
- Meeting scheduling
- Professor information lookup
- Office hours management
- Contact information retrieval

### Exam Module
- Exam notification setup
- Schedule management
- Deadline tracking
- Exam information retrieval

## Future Enhancements

- Enhanced language support
- Improved entity extraction
- Personalized search based on user profiles
- Integration with university systems
- Advanced conversation history management
- More AI models integration
- Real-time data synchronization
- Enhanced error handling and recovery
- User preference management
- Analytics and reporting features 