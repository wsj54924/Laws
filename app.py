from fastapi import FastAPI
from src import graph
import uvicorn
app=FastAPI()
@app.get("/")
async def root():
    return {"message":"主页面"}
@app.get("/query/{query}")
async def query(query:str):
    return {"answer":graph.app.invoke(query)}
if __name__=="__main__":
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True
    )








