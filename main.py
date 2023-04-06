"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import pickle

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0.5), chain_type="stuff", retriever=store.as_retriever(search_kwargs={"k": 10}))

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Stacks Q&A Bot", page_icon=":robot_face:")
st.header("Stacks Q&A Bot")
st.caption(":books: Trained with Stacks knowledge :zap: Powered by OpenAI, LangChain")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def convert_to_links(s):
    substrings = s.split(",")
    result = []

    github_repos = {
        "docs": "https://github.com/stacks-network/docs/blob/master/",
        "sips": "https://github.com/stacksgov/sips/blob/main/",
        "stacks": "https://github.com/stacks-network/stacks/blob/master/",
        "stacks-blockchain": "https://github.com/stacks-network/stacks-blockchain/blob/main/",
        "book": "https://github.com/clarity-lang/book/blob/main/",
        "shatoshi-paper": "https://github.com/bitcoinbook/shatoshi-paper/blob/master/",
        "bitcoinbook": "https://github.com/bitcoinbook/bitcoinbook/blob/develop/",
        "core-eng": "https://github.com/Trust-Machines/core-eng/blob/main/",
        "bitcoin": "https://github.com/bitcoin/bitcoin/blob/master/",
        "bips": "https://github.com/bitcoin/bips/blob/master/",
        "ord": "https://github.com/casey/ord/blob/master/",
        "awesome-ordinals": "https://github.com/crypt0biwan/awesome-ordinals/blob/main/",
        "awesome-ordinals2": "https://github.com/neu-fi/awesome-ordinals/blob/main/"
    }

    for substring in substrings:
        if '/' in substring and 'N/A' not in substring:
            prefix, content = substring.strip()[7:].split('/', 1)
            link = github_repos.get(prefix.strip(), substring)
            result.append(f'\n - {link}{content}')
        else:
            result.append(f'\n - {substring}')

    return " ".join(result)


def get_text():
    input_text = st.text_input("Type/Edit your question here: ", "What is stacks?", key="input", help="Ask questions about Stacks, Clarity, sBTC, Bitcoin, Ordinals etc.")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"input_documents": store, "question": user_input})
    sources = convert_to_links(result['sources'])
    output = f"#### Answer: ####\n {result['answer']}\nSources: {sources}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        # message(st.session_state["generated"][i], key=str(i))
        # message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        # st.markdown(st.session_state["generated"][i], unsafe_allow_html=False)
        col1, col2, col3 = st.columns([1,6,1])

        with col1:
            st.image("https://avatars.githubusercontent.com/u/8165984?s=60&v=4")

        with col2:
            st.markdown(st.session_state["generated"][i], unsafe_allow_html=False)

        with col3:
            st.write("")
        
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
