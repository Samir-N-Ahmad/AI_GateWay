
import json
import re

class InferenceUtils:
    

    async def execute_tool(self, tool_call: dict, tools_registery: dict):
        name = tool_call["name"]
        arguments = tool_call["arguments"]
        if name not in tools_registery :
            raise ValueError("Tool is not registered")
        if name == "get_available_services":
            func = tools_registery[name]
        return await func(**arguments)
    

    def parse_tool_calls(text: str):
        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL
        )

        if not matches:
            return None
        
        if "<tool_call>" not in text:
            raise ValueError("Incomplete tool call")

        calls = []
        for callJson in matches:
            call = json.loads(callJson)
            if not isinstance(call, dict):
                calls.append(call)
        if len(calls) == 0:
            raise ValueError("Invalid Tool calls") 
        
        return calls
    
    def extract_assisitant_text_from_call_response(text:str):
        assistant_text = re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        text,
        flags=re.DOTALL
    ).strip()
        return assistant_text
