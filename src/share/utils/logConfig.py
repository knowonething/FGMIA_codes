import json


def logConfig(confClass):
    ret = {k: v for k, v in vars(confClass).items() if not k.startswith('__')}
    return json.dumps(ret, indent=4)
