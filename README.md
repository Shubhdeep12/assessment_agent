# Assessment Agent

The **Assessment Agent** is an intelligent AI system that autonomously processes educational content and generates contextually relevant assessment questions. Using advanced natural language processing and machine learning capabilities, it understands content, maintains knowledge in a vector database, and adaptively generates questions based on context and requirements. The agent offers two interactive interfaces: a terminal-based interface and a web-based interface powered by Streamlit.

## Key Capabilities

### 1. Intelligent Content Processing
- Autonomous text extraction from various file formats (PDF, DOCX, TXT, HTML, CSV)
- Smart text chunking with context preservation
- Vector embeddings generation for semantic understanding

### 2. Knowledge Management
- Vector database storage for efficient content retrieval
- Intelligent caching of processed content
- Metadata tracking for content updates

### 3. Adaptive Question Generation
- Context-aware question creation
- Multiple difficulty levels and question types
- Semantic search for relevant content
- Quality assurance through understanding-focused questions

### 4. Interactive Interfaces
- Real-time response to user inputs
- Session state management
- Progress tracking and feedback

## Prerequisites

- **Python 3.8** or higher
- A reliable internet connection (required for API calls and model downloads)

## Setup

### 1. Environment Configuration

Create a `.env` file in the project root with the following keys:

- **LLM_API_URL**: The URL for the LLM API.
- **LLM_MODEL_NAME**: The model name used by the LLM API.

Example **.env** file:

```properties
LLM_API_URL=your_model_chat_api_url
LLM_MODEL_NAME=chat_llm_model_name
```

### 2. Dependency Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Data Directory Setup

The project retrieves input files from the data directory. By default, a sample file (`react_basics.docx`) is available.

To use custom data: Remove `react_basics.docx` from the data folder and add your own files.
Supported formats: PDF, DOCX, TXT, HTML, and CSV.

## Running the Agent

You can run the Assessment Agent in two ways:

### 1. Terminal Interface

For a command-line experience, execute:

```bash
python assessment.py
```

Features:
- Interactive command-line interface
- Process files from the data directory
- Generate questions with specified difficulty and type
- Option to generate multiple sets of questions

### 2. Web Interface

For a user-friendly web interface, execute:

```bash
streamlit run chat.py
```

Features:
- Modern web-based UI
- Drag-and-drop file upload
- Visual progress tracking
- Option to use existing processed data
- Download generated questions as text files
- Maintain history of generated question sets
- Download individual sets or all questions at once

## Logging

The application logs events both to the console and to the `assessment_agent.log` file located in the project root.
This log file provides detailed debugging information and runtime insights.

## Project Structure

```plaintext
assessment_agent/
├── assessment.py         # Terminal interface implementation
├── chat.py              # Web interface implementation
├── config.py            # Configuration settings and environment variable loader
├── requirements.txt     # Project dependencies
├── .env                # Environment configuration file
├── data/               # Directory containing input files
│   └── react_basics.docx # Default sample file (remove to use your own data)
└── README.md           # This file
```

## Additional Information

- **Model Initialization**: Uses Hugging Face's Transformers library with a default model (`bert-base-uncased`) for embeddings.
- **Database Storage**: Embedded vectors and related metadata are managed via LanceDB.
- **Rate Limiter**: In-built mechanism to manage and throttle requests to the LLM API, ensuring adherence to API limits.
- **Web Interface**: Built with Streamlit for a responsive and intuitive user experience.

## Interface Comparison

| Feature                    | Terminal Interface | Web Interface |
|---------------------------|-------------------|---------------|
| File Processing           | From data directory | Drag-and-drop upload |
| Question Generation       | Interactive prompts | Form-based input |
| Question History         | Single session     | Persistent across runs |
| Download Options         | Display only       | Text file export |
| Progress Tracking        | Text-based         | Visual progress bar |
| Multiple Question Sets   | Yes                | Yes, with history |

For further details on configuration and troubleshooting, consult the inline comments and documentation within the source code.