from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

LM_STUDIO = "http://localhost:1234/v1"

llm = ChatOpenAI(
    model="lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    openai_api_base=LM_STUDIO,
    openai_api_key="dummy-key",
    temperature=0.85,
    max_tokens=8000,
)

messages = [
    SystemMessage(
        "Translate the following from English into Italian. Translate the user sentence"
    ),
    HumanMessage("hi!"),
]

# stream
# for token in llm.stream(messages):
#     print(token.content, end="|", flush=True)


# regular
ai_msg = llm.invoke(messages)
print(ai_msg)
