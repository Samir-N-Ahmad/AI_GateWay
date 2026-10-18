from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer
import torch
from ai_gateway.services.chat import Chat, Mode
from typing import AsyncIterator

class InferenceService:

    def __init__(self, model:str):
        self._model_name = model
        
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self._template = chatTemplate
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._chats = dict()


    async def new_chat(self, mode:Mode):

        chat_id = 0 if self._chats is None else len(self._chats)
        chat = Chat(chat_id, mode, self._tokenizer, self._model)
        self._chats[chat_id] = chat
        
        return chat_id
    
    async def new_message(self, chat_id:str, message:str) -> AsyncIterator[str]:

        if chat_id in self._chats:
            async for chunk in self._chats[int(chat_id)].send_message(message):
                print(
            f" passed chunk: {chunk!r}",
            flush=True)
                yield chunk

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
