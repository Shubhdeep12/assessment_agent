import streamlit as st
from pathlib import Path
import time
from typing import List
import os
import torch
from streamlit.runtime.uploaded_file_manager import UploadedFile

from assessment import AssessmentAgent, Config, process_file

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False

def create_agent():
    """Create and return AssessmentAgent instance."""
    if st.session_state.agent is None:
        config = Config.from_env()
        st.session_state.agent = AssessmentAgent(config)
    return st.session_state.agent

def handle_file_upload(uploaded_files: List[UploadedFile]) -> None:
    """Process uploaded files and store in vector database."""
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return

    agent = create_agent()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create temporary directory for uploads
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    try:
        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file temporarily
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Process stages with progress updates
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress(idx / total_files)
            
            # Process the file
            process_file(str(temp_path), agent)
            progress_bar.progress((idx + 0.5) / total_files)

            # Clean up temporary file
            os.remove(temp_path)
            progress_bar.progress((idx + 1) / total_files)

        st.session_state.files_processed = True
        status_text.text("All files processed successfully!")
        progress_bar.progress(1.0)
        time.sleep(1)  # Show completion for a moment
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                file.unlink()
            temp_dir.rmdir()

def main():
    st.set_page_config(
        page_title="AI Assessment Generator",
        page_icon="📚",
        layout="wide"
    )

    initialize_session_state()

    st.title("AI Assessment Generator")
    st.write("Welcome to the AI Assessment Generator! Use /start to begin.")

    # Chat input
    user_input = st.text_input("Enter your command:", key="user_input")

    if user_input.strip().lower() == "/start":
        st.session_state.chat_started = True

    if st.session_state.chat_started:
        if not st.session_state.files_processed:
            st.write("Would you like to:")
            col1, col2 = st.columns(2)
            with col1:
                uploaded_files = st.file_uploader(
                    "Choose files to upload",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'txt', 'html', 'csv']
                )
                if uploaded_files and st.button("Process Files"):
                    handle_file_upload(uploaded_files)

            with col2:
                if st.button("Use Existing Data"):
                    agent = create_agent()
                    if agent and len(agent.table.search().to_list()) > 1:  # Check if there's data beyond initialization
                        st.session_state.files_processed = True
                        st.rerun()
                    else:
                        st.warning("No existing data found. Please upload files first.")

        else:
            # Show assessment generation options
            st.subheader("Generate Assessment")
            
            agent = create_agent()
            
            col1, col2 = st.columns(2)
            with col1:
                difficulty = st.selectbox(
                    "Select difficulty level:",
                    options=agent.config.valid_difficulty_levels
                )
            
            with col2:
                question_type = st.selectbox(
                    "Select question type:",
                    options=agent.config.valid_question_types
                )

            generate_button = st.button(
                "Generate Questions",
                disabled=st.session_state.generating_questions
            )
            
            if generate_button:
                st.session_state.generating_questions = True
                with st.spinner("Generating questions..."):
                    try:
                        # Create query embedding
                        query = f"Generate {difficulty} {question_type} questions"
                        inputs = agent.tokenizer(
                            query,
                            return_tensors="pt",
                            truncation=True,
                            max_length=agent.config.max_sequence_length
                        )
                        
                        with torch.no_grad():
                            outputs = agent.model(**inputs)
                            query_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]

                        # Retrieve and generate
                        similar_chunks = agent.search_similar_chunks(query_embedding)
                        context = " ".join(similar_chunks)
                        questions = agent.generate_questions(context, difficulty, question_type)
                        
                        # Add new questions to the list
                        st.session_state.generated_questions.append({
                            'questions': questions,
                            'difficulty': difficulty,
                            'type': question_type,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.session_state.generating_questions = False
                        
                        # Display all generated questions
                        st.write("### Generated Questions")
                        for i, q_set in enumerate(st.session_state.generated_questions, 1):
                            st.write(f"\n#### Set {i} ({q_set['difficulty']} {q_set['type']} - {q_set['timestamp']})")
                            st.write(q_set['questions'])
                            
                            # Download button for each set
                            q_text = f"Questions Set {i}\nDifficulty: {q_set['difficulty']}\nType: {q_set['type']}\nGenerated: {q_set['timestamp']}\n\n{q_set['questions']}"
                            st.download_button(
                                f"Download Set {i}",
                                q_text,
                                file_name=f"questions_set_{i}.txt",
                                mime="text/plain",
                                key=f"download_{i}"
                            )
                        
                        # Add buttons for generating more or resetting
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Generate More"):
                                st.rerun()  # This will keep the same settings but generate new questions
                        with col2:
                            # Download all button
                            if st.session_state.generated_questions:
                                all_text = "\n\n".join([
                                    f"Questions Set {i+1}\nDifficulty: {q['difficulty']}\nType: {q['type']}\nGenerated: {q['timestamp']}\n\n{q['questions']}"
                                    for i, q in enumerate(st.session_state.generated_questions)
                                ])
                                st.download_button(
                                    "Download All Sets",
                                    all_text,
                                    file_name="all_questions.txt",
                                    mime="text/plain",
                                    key="download_all"
                                )
                        with col3:
                            if st.button("Reset"):
                                st.session_state.chat_started = False
                                st.session_state.files_processed = False
                                st.session_state.generated_questions = []
                                st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
                        st.session_state.generating_questions = False

if __name__ == "__main__":
    main()