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
            if not self._chats[int(chat_id)].is_generating():
                async for chunk in self._chats[int(chat_id)].send_message(message):
                    print(
                f" passed chunk: {chunk!r}",
                flush=True)
                    yield chunk

    async def stopGeneration(self, chat_id:str) -> AsyncIterator[str]:
        if chat_id in self._chats:
            if  self._chats[int(chat_id)].is_generating():
                self._chats[int(chat_id)].s