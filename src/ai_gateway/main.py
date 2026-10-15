from fastapi import FastAPI
from ai_gateway.services.inference_service import InferenceService

inf = None 
app = FastAPI()


@app.get("/inference/start")
def inference():
    global inf
    if inf is None :
        inf = InferenceService("Qwen/Qwen2.5-0.5B-Instruct")
    return {"code":200, "inference":"Started"}


@app.get("/send/{chat_id}/{messages}")
async def send(chat_id:str, messages:str):
    global inf
    result = ""
    async for chunk in inf.send_message(messages, chat_id):
        print(chunk, end="", flush=True)
        result += chunk
    return {"code":200, "inference":result}

@app.get("/chat/new")
async def new():
    global inf
    result = await inf.new_chat()
    return {"code":200, "inference":result}