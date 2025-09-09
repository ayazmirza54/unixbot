import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from google.generativeai import types # Import types for explicit type hints

class GeminiSQLChatInterface:
    def __init__(self, api_key, model_name="gemini-2.5-flash"): # Updated model name as per user's example
        # Configure API
        genai.configure(api_key=api_key)

        # Initialize the client directly
        self.client = genai.Client(api_key=api_key)

        # Model name
        self.model_name = model_name

        # Generation configuration (converted to types.GenerateContentConfig later)
        self.raw_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            # "response_mime_type": "text/plain", # Not directly supported in types.GenerateContentConfig
        }

        # Convert to types.GenerateContentConfig
        self.generation_config = types.GenerateContentConfig(
            temperature=self.raw_generation_config["temperature"],
            top_p=self.raw_generation_config["top_p"],
            top_k=self.raw_generation_config["top_k"],
            max_output_tokens=self.raw_generation_config["max_output_tokens"],
            # Add thinking_config here
            thinking_config=types.ThinkingConfig(
                thinking_budget=0, # This turns off the thinking mode as requested
            ),
        )

        # Original system instruction text
        self.system_instruction_text = (
            "You are an expert shell script developer and you have thorough knowledge of unix. The user will upload file as additional context for making a shell script."
            "Your job is to help the user with unix concepts, if users asks for any unix commands help provide the user a detailed description about the command."
            "if the user asks for a shell script help with the same"
        )

        # Initialize conversation history (list of types.Content for API calls)
        self.conversation_history = []
        self.files_data = [] # Stores dictionary with 'file_name' and 'content' for context

    def process_files(self, file_paths):
        """
        Read and store the full content of uploaded files for context.
        After processing, it re-initializes the conversation context.

        :param file_paths: List of file paths to process
        :return: None
        """
        self.files_data = [] # Clear previous files_data

        if not file_paths: # If no files, just re-initialize context without them
            self._initialize_conversation_with_context()
            return

        for path in file_paths:
            try:
                if path.endswith('.csv'):
                    with open(path, "r") as file:
                        content = file.read()
                    file_data = {
                        "file_name": os.path.basename(path),
                        "content": content,
                    }
                    self.files_data.append(file_data)
                else:
                    st.sidebar.error(f"Unsupported file format for: {path}")
            except Exception as e:
                st.sidebar.error(f"Error processing {path}: {e}")
        
        # After processing new files, re-initialize conversation history to include them
        self._initialize_conversation_with_context()

    def _initialize_conversation_with_context(self):
        """
        Prepares the initial conversation history with system instruction and file data.
        This method is private (`_`) and should be called by `__init__` or `process_files`.
        It clears any existing history and sets up the base context.
        """
        initial_parts = [types.Part.from_text(self.system_instruction_text + "\n\n")]

        if self.files_data:
            initial_parts.append(types.Part.from_text("The following files have been uploaded. Please use their content for writing shell script :\n\n"))
            for file_data in self.files_data:
                # Using triple backticks for better formatting of file content within the prompt
                initial_parts.append(types.Part.from_text(f"File: {file_data['file_name']}\nContent:\n```\n{file_data['content']}\n```\n\n"))
        
        # Always start with an initial user content block, combining system instruction and file context
        self.conversation_history = [
            types.Content(
                role="user",
                parts=initial_parts
            )
        ]
        st.sidebar.info("Chat context initialized with system instruction and file data.")

    def send_message(self, user_input):
        """
        Send a message to Gemini and get a response.

        :param user_input: User's query/message
        :return: Response from Gemini
        """
        # Ensure initial context is set up if no conversation has started yet
        if not self.conversation_history:
            self._initialize_conversation_with_context()

        # Add the current user input as a new user message to the conversation history
        self.conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(user_input)]
            )
        )

        try:
            # Call the model using generate_content_stream as per your example
            response_chunks = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=self.conversation_history, # Pass the entire history for context
                config=self.generation_config, # Use the generation config including thinking_config
            )

            full_response_text = ""
            for chunk in response_chunks:
                if chunk.parts: # Check if the chunk has parts
                    for part in chunk.parts:
                        if part.text: # Check if the part has text content
                            full_response_text += part.text
            
            # Append assistant's full response to history for subsequent API calls
            self.conversation_history.append(
                types.Content(
                    role="model", # Use 'model' for assistant/LLM responses in this API
                    parts=[types.Part.from_text(full_response_text)]
                )
            )
            return full_response_text

        except Exception as e:
            st.error(f"Error generating response: {e}")
            # If an error occurs, remove the last user message to allow retry without polluting history
            if self.conversation_history and self.conversation_history[-1].role == "user":
                self.conversation_history.pop()
            return "Sorry, I encountered an error while generating a response."


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Unixbot : Gemini powered Unix Assistant",
        page_icon="pngwing.com.png", # Make sure this file exists in your app directory
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
        # Initialize interface and its internal history for the first time
        st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)
        # Call to set up initial system instruction context
        st.session_state.gemini_interface._initialize_conversation_with_context()

    # File Upload
    uploaded_files_streamlit = st.sidebar.file_uploader(
        "Upload Files (CSV only)",
        type=["csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Use a flag to track if file selection has changed to avoid unnecessary reprocessing
    if 'last_uploaded_file_ids' not in st.session_state:
        st.session_state.last_uploaded_file_ids = []

    current_file_ids = [f.file_id for f in uploaded_files_streamlit] if uploaded_files_streamlit else []

    # Process uploaded files if they are new or have changed
    if current_file_ids != st.session_state.last_uploaded_file_ids:
        st.session_state.last_uploaded_file_ids = current_file_ids
        
        if uploaded_files_streamlit:
            temp_files = []
            os.makedirs("temp", exist_ok=True) # Ensure 'temp' directory exists
            for uploaded_file in uploaded_files_streamlit:
                temp_path = os.path.join("temp", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_path)
            
            # Process files and prepare content in the interface
            st.session_state.gemini_interface.process_files(temp_files)
            st.session_state.chat_history = [] # Clear display history when new files are uploaded

            # Clean up temporary files after processing (good practice)
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except OSError as e:
                    st.sidebar.warning(f"Error cleaning up temp file {temp_file}: {e}")
        else:
            # If no files are uploaded (e.g., user removed them), re-initialize context without files
            st.session_state.gemini_interface.process_files([]) # Pass empty list
            st.session_state.chat_history = [] # Clear display history

    # Chat input
    user_input = st.chat_input("Enter your unix help request...")

    if user_input:
        # Add user message to chat history for display
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating shell script / Unix commands..."):
                response = st.session_state.gemini_interface.send_message(user_input)
            st.write(response)

        # Add assistant response to chat history for display
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history from session state (for display only)
    st.sidebar.header("Chat History (Display)")
    # Render messages in reverse order to show newest at top if sidebar is long
    for message in reversed(st.session_state.chat_history):
        if message['role'] == 'user':
            st.sidebar.text(f"👤 {message['content'][:50]}...") # Truncate for sidebar display
        else:
            st.sidebar.text(f"🤖 {message['content'][:50]}...") # Truncate for sidebar display

    # Reset button
    if st.sidebar.button("Start New Conversation"):
        st.session_state.chat_history = []
        # Re-initialize the Gemini interface to clear its internal history and files_data
        st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)
        st.session_state.gemini_interface._initialize_conversation_with_context() # Set initial context
        st.session_state.last_uploaded_file_ids = [] # Reset file tracking
        st.session_state.file_uploader = [] # Clear file uploader state if possible (Streamlit handles some internal state)
        st.rerun()


if __name__ == "__main__":
    main()
