"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import pickle
import random
import langchain
from langchain.cache import InMemoryCache
import os
import discord
from discord.ext import commands
import re
import asyncio

TOKEN = os.getenv('DISCORD_TOKEN')
# Initialize Discord client
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True

client = commands.Bot(command_prefix="!", intents=intents)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Stacks Q&A Bot", page_icon=":robot_face:")

langchain.llm_cache = InMemoryCache()
langchain.debug = True

# Load the LangChain.
@st.cache_resource
def load_store():
    index = faiss.read_index("docs.index")

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
    index.nprobe = 10
    store.index = index
    return store

store = load_store()

chain1 = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0.2, model_name='gpt-4-0314', verbose=True), chain_type="stuff", retriever=store.as_retriever(search_kwargs={"k": 10}))

chain1.verbose = True

topics = ['Stacks','Bitcoin','Ordinals','sBTC','Clarity','Consensus','Epoch','Hard Fork','Taproot','PoX','Wallet','elliptic curve','zk','chain state', 'blockchain', 'satoshi','dlc']

styles = [
        "Master Yoda",
        "ELI5",
        "Hip-Hop Song",
        "Shakespearean",
        "News Anchor",
        "Poetic",
        "Scientific",
        # "Formal",
        # "Informal",
        "Humorous",
        "Sarcastic",
        "Mystery Novel Narrator",
        # "Motivational Speaker",
        "Sports Commentator",
        "Old West Cowboy",
        "Film Noir Detective",
        "Superhero",
        "Fairy Tale",
        "Medieval Knight",
        "Alien",
        "Friendly",
        # "Neutral",
        # "Professional",
        # "Casual",
        # "Energetic",
        # "Calm",
        # "Encouraging",
        # "Sincere",
        # "Empathetic",
        # "Enthusiastic",
        "Conversational",
        "Storytelling",
        # "Argumentative",
        "Analytical",
        # "Persuasive",
        "Educational",
        # "Reflective",
        "Descriptive",
        "Nostalgic",
        "Satirical",
        # "Whimsical",
        # "Inquisitive",
        "Instructive",
        "Philosophical",
        "Fantasy",
        "Sci-Fi",
        "Romantic",
        # "Uplifting",
        # "Meditative",
        # "Introspective",
        "Futuristic",
        # "Pessimistic",
        # "Optimistic",
        # "Skeptical",
        # "Apologetic",
    ]

def remove_mentions(message_content, mentions):
    for mention in mentions:
        mention_string = mention.mention
        message_content = message_content.replace(mention_string, "")
    
    return message_content.strip()

def replace_link(match):
    link = match.group(0)
    return f"<{link}>"

def main():
    st.header("StacksGPT : Q&A Bot")
    st.caption(":books: Trained with Stacks Ecosystem knowledge :zap: Powered by OpenAI")
    st.caption(":toolbox: Built using LangChain, FAISS")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    user_input = None
    user_button_clicked = False
    icol1, icol2 = st.columns([6,2])
    with icol1:
        user_input = st.text_area('Curious about Stacks, Bitcoin, Clarity, sBTC, Ordinals... ?', 
                                placeholder='🗣️ Type your question here...', height=135)

    with icol2:
        st.header(' ')
        user_button_clicked = st.button('🔍 Search Answer', use_container_width=True)
        new_button_clicked = st.button('🐵 I\'m Feeling Curious', use_container_width=True)

    if new_button_clicked:
        auto_input = 'explain ' + random.choice(topics)+' using '+ random.choice(styles) + ' style'
        st.session_state.past.append(auto_input)
        with st.spinner('🧠 Thinking about auto-generated question: ' + auto_input):
            result = chain1({"question": auto_input+ '. Always put all sources at the end and always Respond in Markdown format.'})
            output = f"#### Answer: ####\n {result['answer']}\nSources:\n {result['sources']}"
            st.session_state.generated.append(output)

    elif user_button_clicked and user_input:
        st.session_state.past.append(user_input)
        with st.spinner('🧠 Thinking about your question: ' + user_input):
            result = chain1({"question": user_input + '. Always put all sources at the end and always Respond in Markdown format.'})
            output = f"#### Answer: ####\n {result['answer']}\nSources:\n {result['sources']}"
            st.session_state.generated.append(output)


    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            col1, col2, col3 = st.columns([1,8,1])

            with col1:
                st.image("https://avatars.githubusercontent.com/u/8165984?s=60&v=4")

            with col2:
                st.markdown(st.session_state["generated"][i], unsafe_allow_html=False)

            with col3:
                st.write("")
            
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


    @client.event
    async def on_ready():
        print(f'{client.user} has connected to Discord!')

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        print(message.content)
        if client.user in message.mentions:
            await message.reply(f"Hello {message.author.mention}! please wait while I think 🧠 about your question... ")
            question = remove_mentions(message.content, message.mentions)
            result = chain1({"question": question + '. Always put all sources at the end and always Respond in Markdown format.'})
            sources = re.sub(r'https?://\S+', replace_link, result['sources'])
            output = f"Hello {message.author.mention}! \n {result['answer']}\nSources:\n {sources}"
            await message.reply(output)

    @st.cache_resource
    async def run_bot():
        try:
            client.run(TOKEN)
        except KeyboardInterrupt:
            print("\nDiscordBot is shutting down due to CTRL+C.")
            client.close()

    run_bot()

if __name__ == "__main__":
    main()
    