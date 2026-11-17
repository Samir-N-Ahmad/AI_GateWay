from ai_gateway.services.mode import Mode
from typing import AsyncIterator
from threading import Thread, Event
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, AsyncTextIteratorStreamer
from ai_gateway.services.cancellation_criteria import CancellationCriteria
from ai_gateway.services.api_service import ApiService
from ai_gateway.utils.inference_utils import InferenceUtils
import json
class Chat:


    MAX_TOOL_CALLS = 2
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
        # move messages to chat level
        self._tools = [
            {
                "type":"function",
                "function":{
                    "name":"get_available_services",
                    "description":(
                        "Fetch the current services offered by the Ministry of Sports. "
                        "Use when the user asks what services are available, what they "
                        "can apply for, what the ministry offers, or requests a list "
                        "of services. Example questions: 'How can you help me?', "
                        "'What can I apply for?', 'Show me your services'. "
                        "Returns service names and available details."
                    ),
                    "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                },
            }
        } 
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
                        You are the Ministry of Sports service agent. Your job is to help
                        users discover services, understand requirements, and complete
                        supported service tasks.

                        Interpret ordinary language as service requests. Users do not need
                        to know service names, tool names, or API terminology.

                        Workflow:
                        1. Identify the user's intended outcome.
                        2. If a tool can help achieve that outcome, use it proactively.
                        3. If the user describes a need without naming a service, retrieve
                        available services and identify relevant options.
                        4. Ask a focused clarification only when missing information prevents
                        the next useful action. Retrieve available information first.
                        5. Answer using tool results. Never invent services, requirements,
                        eligibility rules, or successful actions.
                        6. If no available tool supports the request, explain that limitation
                        and provide a relevant next step.
                        When the user describes a service need, asks what they can apply
                        for, or asks what the ministry offers, call get_available_services
                        before answering.

                        Do this even when the user does not explicitly ask for a service
                        list or mention the tool name.

                        For example:
                        "I want to register a club"
                        → Call get_available_services to find the relevant service.

                        Do not ask permission to retrieve the service catalog.
                        Do not invent service names or requirements.
                        Examples:
                        - "What can I do here?"
                        → Retrieve available services and summarize the options.
                        - "I want to register a sports club."
                        → Retrieve services and look for a relevant registration service.
                        - "I need support for my sports organization."
                        → Retrieve services, identify relevant options, then clarify if needed.
                        - "Hello"
                        → Briefly introduce your service role and ask what the user needs.

                        Use clear, concise language. Move the user's task forward.
                        call the tool get_available_services  on you first run
                        """}.get(mode)
        

        self._tools_registery = {
            "get_available_services": ApiService.get_available_services
        }
        # move messages to chat level
        self._messages = [{"role":"system","content":self._system_prompt}]
        print(self._messages)





    async def send_message(self, message) -> AsyncIterator[str]:
        if self._is_generating:
            raise RuntimeError("Chat is already generating")
        self._is_generating = True
        
        self._messages.append(
            {"role": "user","content":message
        })

        async for chunk in self._run_agent(1):
            yield chunk

        self._is_generating = False
        



    async def _run_agent(self, recursiveCall):
        if recursiveCall > 3:
            raise RuntimeError("Max infernce count reached")
        response = ""
        async for chunk in self._run_inference():
            response += chunk
            yield chunk
                
        if response:
            
            calls = InferenceUtils.parse_tool_calls(response)
            if calls:
                self._messages.append(
                    {
                    "role":"assistant",
                    "content": InferenceUtils.extract_assisitant_text_from_call_response(response),
                    "tool_calls":[
                        {
                            "type":"function",
                            "function":call
                        } for call in calls
                    ]
                }
                )
                await self._handle_tool_calls(calls)
                async for chunk in self._run_agent(
                recursiveCall + 1
            ):
                    yield chunk
            else:
                self._messages.append(
                {"role": "assistant","content":response})


            
            


        # return generated

    def stop_generation(self):
        self.cancellation_event.set()
        return True

    def is_generating(self) -> bool:
        return self._is_generating
    
    async def _run_inference(self) -> AsyncIterator:
        response = ""
        try:
            self._is_generating = True    
            self._inputs = self._tokenizer.apply_chat_template(
                self._messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                tools= self._tools,
                return_dict=True,).to(self._model.device)
            streamer = AsyncTextIteratorStreamer(
                self._tokenizer,  
                skip_prompt=True,
                timeout=60,
                skip_special_tokens=True)
            self._is_generating = True
            thread = Thread(target=self._generate, args=(streamer,))
            thread.start()

            async for chunk in streamer:
                yield chunk
        
        finally:
            await asyncio.to_thread(thread.join)


    async def _handle_tool_calls(self, calls):
        if  calls:
            # for call  in calls:
            # handle only one call for now
            call_results = []
            for call in calls:
                toolcallResult = await InferenceUtils.execute_tool(call, self._tools_registery)
                call_results.append(toolcallResult)
                self._messages.append(
                    ## Add tool call request
                    {
                        "role":"tool",
                        "name": call["name"],
                        "content": json.dumps(toolcallResult)
                    }
                )
                
            

    @torch.inference_mode()
    def _generate(self, streamer):
        self._model.generate(
            **self._inputs,
            max_new_tokens=100,
            streamer=streamer,
            stopping_criteria = self._cancellation_criteria
        ) 
            

        
