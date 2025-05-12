import streamlit as st
import rag_bot
import time

@st.cache_resource 
def load_rag_bot(api_key):
    """Loads the Rag_Bot instance and caches it."""
    print("--- Initializing Rag_Bot (this should only run once or when cleared) ---")
    bot_instance = rag_bot.Rag_Bot(api_key)
    print("--- Rag_Bot Initialized ---")
    return bot_instance

# Set up the Streamlit app page configuration
st.set_page_config(page_title="Maintenance Manuals Chat Bot", page_icon=":robot:")


if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


st.session_state.api_key = st.secrets['GROQ_API_KEY']

st.title("🛠️:blue[Equipment Manuals Chatbot]")

try:
    bot = load_rag_bot(st.session_state.api_key)
except Exception as e:
    st.error(f"Error initializing RAG Bot: {e}")
    # Add more detailed logging or traceback if needed
    st.error("The application might be unstable. Please check logs or contact support.")
    st.stop() # Stop execution if bot fails to load

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

    if process_btn and eqip_name:
        for file in files:
            try:
                with st.spinner("Wait for it...", show_time=True):
                    # bot = rag_bot.Rag_Bot(st.session_state.api_key)
                    bot.tokenize_doc(file)
                    message3, st.session_state.vector_store = bot.create_faiss_index()
                    st.write(message3)
            except AttributeError as e:
                st.write(e)



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




if chat_input:
    # bot = rag_bot.Rag_Bot(st.session_state.api_key)
    # Get Bot response
    bot.query = chat_input
    try:
        bot.generate_initial_answer(eqip_name)
        bot.query_index(st.session_state.vector_store)
        bot_answer = bot.get_final_answer(eqip_name)

        # Display user chat message
        with st.chat_message(name= "assistant", avatar="assistant.png"):
            st.markdown(bot_answer)

        # Append response to messages history to be dispalyed
        st.session_state.messages.append({'role':"assistant", 'content':bot_answer})

    except Exception as e:
        st.write("Sorry Service unavailable now:", e)





