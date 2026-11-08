from ai_gateway.services.mode import Mode
from typing import AsyncIterator
from threading import Thread, Event
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer
from ai_gateway.services.cancellation_criteria import CancellationCriteria
from  ai_gateway.services.api_service import ApiService
from  ai_gateway.utils.inference_utils import InferenceUtils
class Chat:

    def __init__(self, chat_id:str, mode:Mode, tokenizer: AutoTokenizer, model:AutoModelForCausalLM):
        self._chat_id = chat_id
        self._mode = mode
        self._tokenizer = tokenizer
        self._model = model
        self._is_generating = False
        self.cancellation_event = Event() 
        self._api_service = ApiService("https://stage.sports.gov.ae/")
        self._cancellation_criteria = [
            CancellationCriteria(self.cancellation_event)
        ]
        self._system_prompt = {
            Mode.CLASSIFICATION : """
                        You are a text classification model.

                        Classify the user's message into exactly one of these labels:

                        - request: the user is asking for an action or service.
                        - issue: the user is reporting a problem, failure, or incorrect behavior.
                        - recommendation: the user is proposing an improvement or suggestion.

                        Rules:
                        - Return exactly one label.
                        - Use lowercase English.
                        - Do not explain your answer.
                        - If the message contains multiple intents, choose the primary intent.

                        Valid outputs:
                        request
                        issue
                        recommendation
                        """,
            Mode.CHAT : """
                        You are a chat model.

                        Communicate with the user friendly and answer their questions:
                        Help the user identifies and solves the problem
                        Rules:
                        - Use lowercase English.
                        - explain your answer.
                        
                        """,
            Mode.SERVICE_ASSISTANT : """
                        You are a Service assistant.

                        Communicate with the user friendly and help them applying for the Ministry of sports Services:
                        use the provided tools to call the required APIs and get the required data
                        Rules:
                        - Use lowercase English.
                        - explain your attitude

                        """}.get(mode)
        
        self._tools = [{
        "type": "function",
        "function": {
            "name": "get_available_services",
            "description": "Retrieve the available Ministry of Sports e-Services.",
        }
    }]
        self._tools_registery = {
            "get_available_services": self._api_service.get_available_services
        }
        # move messages to chat level
        self._messages = [{"role":"system","content":self._system_prompt}]

    async def send_message(self, message) -> AsyncIterator[str]:
        if self._is_generating:
            return
        self._messages.append(
            {"role": "user","content":message
        })
        self._streamer = AsyncTextIteratorStreamer(self._tokenizer)

        response = ""
        async for chunk in self._streamer:
            response += chunk
            yield chunk
                
            self._messages.append({
                "role":"assistant", "content":response
            })
            calls = InferenceUtils.parse_tool_call(response)
            if calls is not None:
                for call  in calls:
                    response = InferenceUtils.execute_tool(call)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        # return generated

    def stop_generation(self):
        self.cancellation_event.set()
        return True

    def is_generating(self) -> bool:
        return self._is_generating
    
    async def _run_inference(self) -> AsyncTextIteratorStreamer:
        try:
            
            self._inputs = self._tokenizer.apply_chat_template(
                self._messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                tools= self._tools,
                return_dict=True,
                skip_prompt=True,
        skip_special_tokens=True
            ).to(self._model.device)
            
            self._is_generating = True    
            thread = Thread(target=self._generate, args=(self._streamer,))
            thread.start()
            
        
        finally:
            await asyncio.to_thread(thread.join)
            self.cancellation_event.clear()
            self._is_generating = False;       

    @torch.inference_mode()
    def _generate(self, streamer):
        self._model.generate(
            **self._inputs,
            max_new_tokens=100,
            streamer=streamer,
            stopping_criteria = self._cancellation_criteria
        ) 
            

        
