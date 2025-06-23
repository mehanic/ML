from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI 
from transcriber_tool import transcribe_with_models

tools = [transcribe_with_models]
llm = OpenAI(
    temperature=0
)  

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Новый способ вызова (вместо .run)
response = agent.invoke("Расшифруй файл 2-24_Leccion_24.mp3 и выведи оба результата")
print(response)
