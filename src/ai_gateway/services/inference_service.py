from transformers import AutoTokenizer, AutoModelForCausalLM

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
                "content":"The Capital of France is?"
            }
        ]
        self._inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        



    def print_inputs(self):
        print(self._inputs["input_ids"])
        print(self._inputs["input_ids"].shape)