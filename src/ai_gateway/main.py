from fastapi import FastAPI
from ai_gateway.services.inference_service import InferenceService,Mode
from fastapi.responses import StreamingResponse

inf = None 
app = FastAPI()


@app.get("/inference/start")
def inference():
    global inf
    if inf is None :
        inf = InferenceService("Qwen/Qwen2.5-0.5B-Instruct")
    return {"code":200, "inference":"Started"}


@app.get("/inference/modes")
def get_modes():
    return {"code":200, "modes":[{"name":mode.name, "value":mode.value} for mode in Mode]}


@app.get("/send/{chat_id}/{messages}")
async def send(chat_id:str, messages:str):
    global inf
    return StreamingResponse(inf.new_message(int(chat_id), messages), media_type="text/plain")

@app.get("/chat/new/{mode}")
async def new(mode:int):
    global inf
    result = await inf.new_chat(Mode(mode))
    return {"code":200, "inference":result}