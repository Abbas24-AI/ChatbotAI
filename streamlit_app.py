# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import openai
import streamlit as st
import tiktoken  # For token calculation

# Function to calculate token count
def calculate_token_count(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    return total_tokens

# Sidebar setup
with st.sidebar:
    st.title('🤖💬 OpenAI Chatbot')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Display token count
    if "messages" in st.session_state:
        token_count = calculate_token_count(st.session_state.messages)
        st.write(f"**Token Count:** {token_count}")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and generate assistant's response
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages], stream=True):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Update token count in the sidebar
    with st.sidebar:
        token_count = calculate_token_count(st.session_state.messages)
        st.write(f"**Token Count:** {token_count}")
