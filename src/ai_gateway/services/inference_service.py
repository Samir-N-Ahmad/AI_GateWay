from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer
import torch
from threading import Thread
class InferenceService:

    def __init__(self, model:str):
        self._model_name = model
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self._template = chatTemplate
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._messages = dict()
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
                        """

    async def new_chat(self):
        self._streamer = AsyncTextIteratorStreamer(self._tokenizer)
        chat_id = 0 if self._messages is None else len(self._messages)
        self._messages[str(chat_id)] = [
            {   
                "role": "system",
                "content": self._system_prompt
            }
        ]
        print(list(self._messages)[0])
        return chat_id
    


    # def print_inputs(self):
        # print(self._inputs["input_ids"])
        # print( self._messages)
        # print(self._inputs["input_ids"].shape)
        # print(self._inputs["attention_mask"].shape)
        # print(self._outputs.hidden_states[-1].shape)
        # print(torch.argmax(self._outputs.logits[:,-1,:],dim=-1))
        # print(self._tokenizer.decode(torch.argmax(self._outputs.logits[:,30,:],dim=-1)))
        # print(self._tokenizer.decode(torch.argmax(self._outputs.logits[:,31,:],dim=-1)))
        # print(self._tokenizer.decode(torch.argmax(self._outputs.logits[:,32,:],dim=-1)))
        # print(self._tokenizer.decode(torch.argmax(self._outputs.logits[:,33,:],dim=-1)))

        # print(self._tokenizer.batch_decode(self._generated,skip_special_tokens=True)[0])

    async def send_message(self, message, chat_id) -> str:
        if self._messages is None:
            yield "No Chats"
        self._messages[chat_id].append(
            {"role": "user","content":message
        })
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages[chat_id],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self._model.device)
        input_length = self._inputs["input_ids"].shape[-1]

        thread = Thread(target=self._generate)
        thread.start()
        response = ""
        async for chunk in self._streamer:
            response += chunk
            yield chunk
            
        self._messages[chat_id].append({
            "role":"assistant", "content":response
        })
        # return generated
        

    @torch.inference_mode()
    def _generate(self):
        self._model.generate(
            **self._inputs,
            max_new_tokens=100,
            streamer=self._streamer
        ) 
            
