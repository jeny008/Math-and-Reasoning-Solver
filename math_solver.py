
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_classic.agents import Tool, initialize_agent, AgentType

load_dotenv()

st.set_page_config(
    page_title="Math & Reasoning Solver",
    page_icon="🧮",
)

st.title("Text To Math Problem Solver using Groq")

groq_api = st.sidebar.text_input(
    "Please enter your Groq API Key",
    type="password",
)

if not groq_api:
    st.info("Please enter your Groq API Key.")
    st.stop()



model = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=groq_api,
    temperature=0,
)



# 1. Create Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description=(
        "Search Wikipedia for factual information "
        "about people, places, history, and general topics."
    ),
)

# 2. Create Math tool
math_chain = LLMMathChain.from_llm(
    llm=model,
    verbose=False,
)

def math_tool_func(question: str) -> str:
    """Solve mathematical questions using LLMMathChain."""
    return str(math_chain.run(question))

calculator = Tool(
    name="Calculator",
    func=math_tool_func,
    description=(
        "Use this tool for arithmetic calculations, "
        "mathematical expressions, and numerical problems. "
        "Input should contain the complete math question."
    ),
)

# 3. Create Reasoning tool
prompt = """
You are a helpful mathematics and logical reasoning tutor.

Solve the user's question carefully.
Explain the approach in clear, numbered steps.
For logic questions, explain the relevant clues.
For mathematics, show the formula and calculations.
Clearly state the final answer.
Do not invent missing information.

Question: {question}

Answer:
"""

prompt_template = PromptTemplate(
    template=prompt,
    input_variables=["question"],
)

reasoning_chain = LLMChain(
    prompt=prompt_template,
    llm=model,
)

def reasoning_tool_func(question: str) -> str:
    """Generate a step-by-step reasoning explanation."""
    return str(reasoning_chain.run(question=question))

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_tool_func,
    description=(
        "Use this tool for logical reasoning, aptitude, "
        "number series, puzzles, and questions requiring "
        "a step-by-step explanation."
    ),
)

# 4. Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

# 5. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I am a Math and Reasoning Chatbot. "
                "Ask me a math, logic, or general knowledge question."
            ),
        }
    ]

# 6. Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 7. User input
question = st.text_area(
    "Please ask your question:",
    placeholder="Example: Calculate 2456 * 1789",
)

if st.button("Find My Answer"):
    if question.strip():
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            callback_container = st.container()
            st_cb = StreamlitCallbackHandler(
                callback_container,
                expand_new_thoughts=False,
            )

            try:
                with st.spinner("Generating response..."):
                    response = assistant_agent.run(
                        question,
                        callbacks=[st_cb],
                    )

                st.write(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as error:
                st.error(
                    "Unable to generate an answer. "
                    "Check your API key, model access, "
                    "dependencies, and try again."
                )
                st.caption(str(error))
    else:
        st.warning("Please enter a question.")
