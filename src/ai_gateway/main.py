from fastapi import FastAPI
from src.services.inference_service import InferenceService

inf = InferenceService("Qwen/Qwen2.5-0.5B-Instruct")
app = FastAPI()


@app.get("/inference/{messages}")
def inference(messages:str):
    return {"code":200, "inference":inf.print_inputs()}