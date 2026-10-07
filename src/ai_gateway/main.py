from fastapi import FastAPI
from ai_gateway.services.inference_service import InferenceService

inf = None 
app = FastAPI()


@app.get("/inference/start")
def inference():
    global inf
    if inf is None :
        inf = InferenceService("Qwen/Qwen2.5-0.5B-Instruct")
    return {"code":200, "inference":inf.print_inputs()}


@app.get("/send/{messages}")
def send(messages:str):
    global inf
    result = inf.send_message(messages)
    return {"code":200, "inference":result}