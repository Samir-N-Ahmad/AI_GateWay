import httpx
from typing import Any, TypeVar, Generic
from enum import Enum
import json
import logging

T = TypeVar('T')
class Status(Enum):
    Success = 1
    Failure = 0
    Pending = 2


class Data(Generic[T]):
    def __init__(self, status: Status, message:str, data:T):
        self.status = status 
        self.message = message
        self.data : T = data

    @classmethod
    def success(cls, data:T):
        return cls(status=Status.Success, message="Data Fetched Successfully", data=data)
   
    @classmethod
    def failure(cls, error:str):
        return cls(status=Status.Failure, message=error, data=None)
    
    @classmethod
    def loading(cls):
        return cls(status=Status.Pending, message="Fetching data", data=None)

    def is_success(self):
        return self.status == Status.Success
    
    def is_failure(self):
        return self.status == Status.Failure
    
class NetworkUtils:

    @staticmethod
    async def  GET(url:str) -> Data[T]:
        try:
            async with httpx.AsyncClient(headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json"},trust_env=False,
            timeout=5.0,
            follow_redirects=True,) as client:
                response = await client.get(url)
            if response.is_success:
                return Data.success(response.json())
            else:
                return Data.failure(response.json())
        except httpx.ReadTimeout as e:
            print("Timoutmout")
            logging.exception("GET failed: %s", type(e).__name__)

            return Data.failure("Request time out")
        except (TypeError, NameError) as e:
            print(f"UnNkown Error {e}")
            return Data.failure(f"UnNkown Error {e}")