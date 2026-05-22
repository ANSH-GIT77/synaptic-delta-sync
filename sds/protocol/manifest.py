import datetime

class PatchManifest:
    def __init__(self, version, target_model):
        self.version = version
        self.target_model = target_model
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self):
        return self.__dict__