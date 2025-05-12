import streamlit as st
import rag_bot
import time

# Set up the Streamlit app page configuration
st.set_page_config(page_title="Maintenance Manuals Chat Bot", page_icon=":robot:")


if "api_key" not in st.session_state:
    st.session_state.api_key = None

st.session_state.api_key = st.secrets['GROQ_API_KEY']

st.title("🛠️:blue[Equipment Manuals Chatbot]")
bot = rag_bot.Rag_Bot(st.session_state.api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "create_and_upload" not in st.session_state:
    st.session_state.prevent_upload = True


with st.sidebar:
    st.header("🔽 Upload documents 🔽")
    files = st.file_uploader(label= "Upload files", accept_multiple_files=True, type="pdf", label_visibility='hidden')
    eqip_name = st.text_input('Insert equipment name')
    if files:
        st.session_state.prevent_upload = False

    process_btn = st.button(label= "Process Documents", use_container_width=True, disabled=st.session_state.prevent_upload)

    if process_btn:
        for file in files:
            try:
                with st.spinner("Wait for it...", show_time=True):
                    bot.tokenize_doc(file)
                    message3 = bot.create_get_collection(eqip_name)
                    st.write(message3)
            except AttributeError as e:
                st.write("Please upload file first!! and try again")



# Display messages from chat history each rerun
for message in st.session_state.messages:
    with st.chat_message(name= message['role'], avatar=f"{message['role']}.png"):
        st.markdown(message['content'])

chat_input = st.chat_input("Type your question here!")
if chat_input:
    # Display user chat message
    with st.chat_message(name= "user", avatar="user.png"):
        st.markdown(chat_input)
    st.session_state.messages.append({'role':"user", 'content':chat_input})


st.session_state.selected_collection = st.radio("Select Equipment Collection", bot.collections, horizontal=True)


if chat_input:
    # Get Bot response
    bot.query = chat_input
    bot.select_collection(st.session_state.selected_collection)
    try:
        bot.generate_initial_answer()
        bot.query_collection()
        bot_answer = bot.get_final_answer()

        # Append response to messages history to be dispalyed
        st.session_state.messages.append({'role':"assistant", 'content':bot_answer})

    except Exception as e:
        st.write("Sorry Service unavailable now:", e)


    st.rerun()


