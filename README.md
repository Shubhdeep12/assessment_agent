---
title: "Assessment Agent"
---

# Assessment Agent

The **Assessment Agent** is a production-grade tool designed to process various file formats, generate text embeddings, and create assessment questions using a Language Learning Model (LLM) API.

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

To start the assessment process, execute:

```bash
python assessment.py
```

Follow the on-screen instructions to generate assessment questions from the processed content.

## Logging

The application logs events both to the console and to the `assessment_agent.log` file located in the project root.
This log file provides detailed debugging information and runtime insights.

## Project Structure

```plaintext
assessment_agent/
├── assessment.py         # Main application file
├── config.py             # Configuration settings and environment variable loader
├── requirements.txt      # Project dependencies
├── .env                  # Environment configuration file
├── data/                 # Directory containing input files
│   └── react_basics.docx # Default sample file (remove to use your own data)
└── README.md             # This file
```

## Additional Information

- **Model Initialization**: Uses Hugging Face's Transformers library with a default model (`bert-base-uncased`) for embeddings.
- **Database Storage**: Embedded vectors and related metadata are managed via LanceDB.
- **Rate Limiter**: In-built mechanism to manage and throttle requests to the LLM API, ensuring adherence to API limits.

For further details on configuration and troubleshooting, consult the inline comments and documentation within the source code.