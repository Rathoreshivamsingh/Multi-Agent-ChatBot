import streamlit as st
from streamlit_chat import message
from agent import handle_input

import logging



st.title("Multi-Agent Question Answering App")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["url_embedded"] = False

user_input = st.text_input("Enter a URL to process or ask a question:")


if st.button('Submit'):
    with st.spinner("Processing your input..."):

        response = handle_input(user_input, st.session_state["chat_history"])

        logging.info(f"Input: {user_input}")
        logging.info(f"Response: {response}")

        formatted_response = response['answer'] if 'answer' in response else "No response available."

        st.session_state["user_prompt_history"].append(user_input)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", user_input))
        st.session_state["chat_history"].append(("ai", formatted_response))

if st.session_state.get("chat_answers_history"):
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"]
    ):
        message(user_query, is_user=True)
        message(generated_response)
