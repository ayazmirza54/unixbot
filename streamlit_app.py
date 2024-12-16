import os
import pandas as pd
import streamlit as st
import google.generativeai as genai


class GeminiSQLChatInterface:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp"):
        # Configure API
        genai.configure(api_key=api_key)

        # Generation configuration
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Create the model with system instruction
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            system_instruction=(
                "You are an expert shell script developer and you have thorough knowledge of unix. The user will upload file as additional context for making a shell script."
                "Your job is to help the user with unix concepts, if users asks for any unix commands help provide the user a detailed description about the command."
                "if the user asks for a shell script help with the same"
            ),
        )

        # Initialize chat session
        self.chat_session = None
        self.files_data = []

    def process_files(self, file_paths):
        """
        Read and store the full content of uploaded files for context.

        :param file_paths: List of file paths to process
        :return: None
        """
        self.files_data = []

        for path in file_paths:
            try:
                if path.endswith('.csv'):
                    # Read CSV file
                    with open(path, "r") as file:
                        content = file.read()

                    # Store the full content as a string
                    file_data = {
                        "file_name": os.path.basename(path),
                        "content": content,
                    }
                    self.files_data.append(file_data)
                else:
                    st.sidebar.error(f"Unsupported file format for: {path}")
            except Exception as e:
                st.sidebar.error(f"Error processing {path}: {e}")

    def start_chat(self):
        """
        Start a new chat session with the full file content as context.
        """
        # Prepare initial context with full file data
        initial_context = [
            {
                "role": "user",
                "parts": [
                    "The following files have been uploaded. Please use their content for writing shell script :\n\n",
                    *[
                        f"File: {file_data['file_name']}\nContent:\n{file_data['content']}\n\n"
                        for file_data in self.files_data
                    ],
                ],
            }
        ]

        # Start chat session
        self.chat_session = self.model.start_chat(history=initial_context)
        st.sidebar.info("Chat session started with full file data as context.")

    def send_message(self, user_input):
        """
        Send a message to Gemini and get a response.

        :param user_input: User's query/message
        :return: Response from Gemini
        """
        # Ensure the chat session is initialized
        if not self.chat_session:
            self.start_chat()

        # Send the user's message and get response
        response = self.chat_session.send_message(user_input)
        return response.text


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Unixbot : Gemini powered unix Assistant",
        page_icon=":material/terminal:",
        layout="wide",
    )

    # Title
    st.title("🤖 Unixbot : Gemini powered Unix Assistant")
    st.markdown("Develop shell script with AI assistance using full file data!")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Fetch API Key from environment variables
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key:
        st.sidebar.error("API key not found in environment variables! Set GEMINI_API_KEY.")
        return

    # Initialize session state for chat history and Gemini interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'gemini_interface' not in st.session_state:
        st.session_state.gemini_interface = None

    # Initialize or retrieve Gemini Chat Interface
    if st.session_state.gemini_interface is None:
        st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)

    # File Upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files (CSV only)",
        type=["csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Process uploaded files if provided
    if uploaded_files:
        # Save uploaded files temporarily
        temp_files = []
        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_path)

        # Process files and prepare content
        st.session_state.gemini_interface.process_files(temp_files)

    # Chat input
    user_input = st.chat_input("Enter your unix help request...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating shell script / Unix commands..."):
                response = st.session_state.gemini_interface.send_message(user_input)
            st.write(response)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    st.sidebar.header("Chat History")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.sidebar.text(f"👤 {message['content']}")
        else:
            st.sidebar.text(f"🤖 {message['content']}")

    # Reset button
    if st.sidebar.button("Start New Conversation"):
        st.session_state.chat_history = []
        st.session_state.gemini_interface = None
        st.rerun()


if __name__ == "__main__":
    main()
