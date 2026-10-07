from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class InferenceService:

    def __init__(self, model:str):
        self._model_name = model
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self._template = chatTemplate
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._prepare_model()

    def _prepare_model(self):
        self._messages = [
             {
                "role": "system",
                "content": """
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
            },
            {
                "role": "user",
                "content": "التطبيق لا يعمل"
            }
        ]
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        self._inputs = {
            key: value.to(self._device)
            for key, value in self._inputs.items()
        }
        with torch.inference_mode():
            self._outputs = self._model.generate(**self._inputs,  max_new_tokens=20)
            self._generated = self._outputs[:,self._inputs["input_ids"].shape[1]:]
        output = self._tokenizer.batch_decode(self._generated,skip_special_tokens=True)[0]
        
        self._messages.append({"role": "assistant","content":output})

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

    def send_message(self, message) -> str:
        self._messages.append(
            {"role": "user","content":message
        })
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self._model.device)
        input_length = self._inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = self._model.generate(
                **self._inputs,
                max_new_tokens=20
            )[:,input_length:]

       
        text = self._tokenizer.batch_decode(generated,skip_special_tokens=True)[0]
        print(text)
        self._messages.append({
            "role":"assistant", "content":text
        })
        # return generated
        

