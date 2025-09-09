import os
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types  # new SDK

class GeminiSQLChatInterface:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        # Initialize client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Typed generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,  # disables “thinking” mode
            ),
        )

        # System instruction
        self.system_instruction_text = (
            "You are an expert shell script developer and you have thorough knowledge of Unix. "
            "The user may upload files as context for making a shell script. "
            "Help with Unix concepts, explain commands in detail, and generate scripts when asked."
        )

        self.conversation_history = []
        self.files_data = []

    def process_files(self, file_paths):
        """Read and store uploaded CSV file contents."""
        self.files_data = []

        if not file_paths:
            self._initialize_conversation_with_context()
            return

        for path in file_paths:
            try:
                if path.endswith(".csv"):
                    with open(path, "r") as file:
                        content = file.read()
                    self.files_data.append(
                        {"file_name": os.path.basename(path), "content": content}
                    )
                else:
                    st.sidebar.error(f"Unsupported file format: {path}")
            except Exception as e:
                st.sidebar.error(f"Error processing {path}: {e}")

        self._initialize_conversation_with_context()

    def _initialize_conversation_with_context(self):
        """Prepare initial system + file context."""
        initial_parts = [types.Part.from_text(self.system_instruction_text + "\n\n")]

        if self.files_data:
            initial_parts.append(types.Part.from_text("The following files have been uploaded:\n"))
            for file_data in self.files_data:
                initial_parts.append(
                    types.Part.from_text(
                        f"File: {file_data['file_name']}\nContent:\n```\n{file_data['content']}\n```\n"
                    )
                )

        self.conversation_history = [
            types.Content(role="user", parts=initial_parts)
        ]
        st.sidebar.info("Chat context initialized with system instruction and file data.")

    def send_message(self, user_input):
        """Send a message to Gemini and stream back the response."""
        if not self.conversation_history:
            self._initialize_conversation_with_context()

        # Add user input
        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(user_input)])
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=self.conversation_history,
                config=self.generation_config,
            )

            full_response_text = ""
            placeholder = st.empty()  # live update in Streamlit

            for chunk in response_stream:
                if chunk.text:
                    full_response_text += chunk.text
                    placeholder.markdown(full_response_text)

            # Save model response to history
            self.conversation_history.append(
                types.Content(role="model", parts=[types.Part.from_text(full_response_text)])
            )

            return full_response_text

        except Exception as e:
            st.error(f"Error generating response: {e}")
            if self.conversation_history and self.conversation_history[-1].role == "user":
                self.conversation_history.pop()
            return "Sorry, I encountered an error while generating a response."


def main():
    st.set_page_config(
        page_title="Unixbot : Gemini powered Unix Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Unixbot : Gemini powered Unix Assistant")
    st.markdown("Develop shell scripts with AI assistance using full file data!")

    # Sidebar
    st.sidebar.header("Configuration")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.sidebar.error("API key not found in environment variables! Set GEMINI_API_KEY.")
        return

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "gemini_interface" not in st.session_state:
        st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)
        st.session_state.gemini_interface._initialize_conversation_with_context()

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True, key="file_uploader"
    )

    if uploaded_files:
        temp_files = []
        os.makedirs("temp", exist_ok=True)
        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_path)

        st.session_state.gemini_interface.process_files(temp_files)
        st.session_state.chat_history = []

        # cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                st.sidebar.warning(f"Error cleaning up temp file {temp_file}: {e}")

    # Chat input
    user_input = st.chat_input("Enter your Unix help request...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response = st.session_state.gemini_interface.send_message(user_input)
            st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Sidebar chat history preview
    st.sidebar.header("Chat History (preview)")
    for message in reversed(st.session_state.chat_history):
        prefix = "👤" if message["role"] == "user" else "🤖"
        st.sidebar.text(f"{prefix} {message['content'][:50]}...")

    if st.sidebar.button("Start New Conversation"):
        st.session_state.chat_history = []
        st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)
        st.session_state.gemini_interface._initialize_conversation_with_context()
        st.rerun()


if __name__ == "__main__":
    main()
