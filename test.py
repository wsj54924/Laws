from langchain_openai import ChatOpenAI
base_url="https://coding.dashscope.aliyuncs.com/v1"
api_key="sk-sp-1245bb936dbc4350b9629ee73baeed8e"
llm=ChatOpenAI(model="glm-5",
               api_key=api_key,
               base_url=base_url)
print(llm.invoke("你好"))