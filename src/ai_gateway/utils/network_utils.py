import requests
from typing import Any, TypeVar, Generic
from enum import Enum
import json


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
    async def  GET(url:str) -> Data[T]:
        response: requests.Response  = await requests.get(url)
        try:
            if response.ok:
                return Data.success(response.json())
            else:
                return 
        except TimeoutError:
            return Data.failure("Request time out")
        except (TypeError, NameError) as e:
            return Data.failure(f"UnNkown Error {e}")