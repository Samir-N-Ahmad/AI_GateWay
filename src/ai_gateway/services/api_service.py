from ai_gateway.utils.network_utils import NetworkUtils, Data
import json
class ApiService:

    def __init__(self, base_url:str):
        self._base_url = base_url
        
    @staticmethod
    async def get_available_services():
        res:Data = await NetworkUtils.GET("https://stage.sports.gov.ae/serviceapi/api/Service/getServiceList")

        if res.is_failure:
            raise RuntimeError(res.message)
        
        if res.is_success:
            
            return [{
                "title":  data["title"],
                "isElectronic" :  data["isElectronic"],
                "isActive": data["isActive"]
            }
                 for data in res["Data"]
            ]