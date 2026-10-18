from ai_gateway.services.mode import Mode
from typing import AsyncIterator
from threading import Thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer

class Chat:

    def __init__(self, chat_id:str, mode:Mode, tokenizer: AutoTokenizer, model:AutoModelForCausalLM):
        self._chat_id = chat_id
        self._mode = mode
        self._tokenizer = tokenizer
        self._model = model
        # move messages to chat level
        self._system_prompt = """
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
                        """ if mode is Mode.CLASSIFICATION else  """
                        You are a chat model.

                        Communicate with the user friendly and answer their questions:
                        Help the user identifies and solves the problem
                        Rules:
                        - Use lowercase English.
                        - explain your answer.

                        Valid outputs:
                        request
                        issue
                        recommendation
                        """
        self._messages = [{"role":"System","content":self._system_prompt}]

        
    async def send_message(self, message) -> AsyncIterator[str]:
        self._messages.append(
            {"role": "user","content":message
        })
        self._streamer = AsyncTextIteratorStreamer(self._tokenizer, )
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self._model.device)
        input_length = self._inputs["input_ids"].shape[-1]

        thread = Thread(target=self._generate, args=(self._streamer,))
        thread.start()
        response = ""
        async for chunk in self._streamer:
            print(
            f" Generated chunk: {chunk!r}",
            flush=True)

            response += chunk
            yield chunk
            
        self._messages.append({
            "role":"assistant", "content":response
        })
        # return generated
        

    @torch.inference_mode()
    def _generate(self, streamer):
        self._model.generate(
            **self._inputs,
            max_new_tokens=100,
            streamer=streamer
        ) 
            

        
