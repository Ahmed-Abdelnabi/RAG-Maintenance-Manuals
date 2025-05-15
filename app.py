import streamlit as st
import rag_bot
import time

@st.cache_resource 
def load_rag_bot(api_key):
    """Loads the Rag_Bot instance and caches it."""
    print("--- Initializing Rag_Bot ---")
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


st.markdown(
    "<h2 style='text-align: center; color: #388186;'>🚚Equipment Manuals Chatbot</h2>",
    unsafe_allow_html=True
)


try:
    bot = load_rag_bot(st.session_state.api_key)
except Exception as e:
    st.error(f"Error initializing RAG Bot: {e}")
    st.error("The application might be unstable. Please Try Again Later.")
    st.stop() # Stop execution if bot fails to load

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "create_and_upload" not in st.session_state:
    st.session_state.prevent_upload = True

st.divider()

st.markdown("<h5 style='text-align: center;'>💡 How to Use:</h5>", unsafe_allow_html=True)
st.markdown("""
- **Step 1:** Upload your equipment manual in PDF format using the upload box.  
- **Step 2:** Enter the equipment name in the text field below the upload area.  
- **Step 3:** Click on **Process Documents** to extract and analyze the content.  
- **Step 4:** Ask any technical question in the chat below about your equipment.""")
st.markdown("<h5 style='text-align: center;'>Example questions:</h5>", unsafe_allow_html=True)
st.markdown(""" 
    - What are the components of the "combustion" system for ther "Gas Turbine"? 
    - How do I troubleshoot a problem of "oil consumption" in the "Air compressor"?  
    """)

with st.sidebar:

    st.markdown("#### Upload Maintenance Manuals")
    files = st.file_uploader(label= "Upload files", accept_multiple_files=True, type="pdf", label_visibility='hidden')
    equip_name = st.text_input('Insert equipment name')
    if files:
        st.session_state.prevent_upload = False

    process_btn = st.button(label= "Process Documents", use_container_width=True, disabled=st.session_state.prevent_upload)

    if process_btn and equip_name:
        for file in files:
            try:
                with st.spinner("Processing...", show_time=True):
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


if chat_input and files and equip_name and st.session_state.vector_store:
    # Display user chat message
    with st.chat_message(name= "user", avatar="user.png"):
        st.markdown(chat_input)
    st.session_state.messages.append({'role':"user", 'content':chat_input})
    
    # Get Bot response
    with st.spinner("Bot is Thinking...", show_time=True):
        bot.query = chat_input
        try:
            bot.generate_initial_answer(equip_name)
            bot.query_index(st.session_state.vector_store)
            bot_answer = bot.get_final_answer(equip_name)

            # Display Assistant chat message
            with st.chat_message(name= "assistant", avatar="assistant.png"):
                st.markdown(bot_answer)

            # Append response to messages history to be dispalyed
            st.session_state.messages.append({'role':"assistant", 'content':bot_answer})

        except Exception as e:
            st.write("Sorry Service unavailable now:", e)

elif chat_input and (not files or not equip_name or not st.session_state.vector_store):
    st.write("Please upload your documents, name your equipment and process the documents first!!")





