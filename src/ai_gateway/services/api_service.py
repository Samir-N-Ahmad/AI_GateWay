from ai_gateway.utils.network_utils import NetworkUtils, Data

class ApiService:

    def __init__(self, base_url:str):
        self._base_url = base_url
        

    async def get_available_services(self):
        res:Data = await NetworkUtils.GET(self._base_url + 'serviceapi/api/Service/getServiceList')

        if res.is_failure:
            raise RuntimeError()
        
        if res.is_success:
            return res.data["Data"]