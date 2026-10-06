from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class InferenceService:

    def __init__(self, model:str):
        self._model_name = model
        # self._template = chatTemplate
        self._tokenizer =  AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name, device_map="auto")
        self._prepare_model()

    def _prepare_model(self):
        self._messages = [
            {
                "role":  "user",
                "content":"whas is Sports Federal Authority in the united Arab Emirates"
            }
        ]
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        with torch.inference_mode():
            self._outputs = self._model.generate(**self._inputs,  max_new_tokens=100)
            self._generated = self._outputs[:,self._inputs["input_ids"].shape[1]:]

    def print_inputs(self):
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

        print(self._tokenizer.batch_decode(self._generated,skip_special_tokens=True)[0])