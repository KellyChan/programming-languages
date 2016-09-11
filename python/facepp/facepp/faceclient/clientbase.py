import settings.settings as settings
from faceapi.facepp import API

class FaceClientBase(object):

    def __init__(self, api_key=settings.API_KEY,
                       api_secret=settings.API_SECRET):
        self.api = API(api_key, api_secret)
        super(FaceClientBase, self).__init__()


    def wait_async(self, session_id):
        return api.wait_async(session_id=session_id)
