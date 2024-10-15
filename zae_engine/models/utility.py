import torch.nn as nn


def initializer(m):
    """
    Initialize the parameters in model.
    For Linear layer, weights are initialized following normal distribution.
    For BatchNorm layer, weights are initialized following uniform distribution.
    For LayerNorm layer, weights are not initialized.
    For others, weights are initialized following kaiming method based on normal distribution.
    (https://arxiv.org/abs/1502.01852v1)

    :param m:
        The member of model.
    """
    if "weight" in dir(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
        elif type(m) == nn.BatchNorm1d:
            nn.init.ones_(m.weight)
        elif type(m) == nn.LayerNorm:
            pass
        else:
            pass
            # nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


# def load_weights(model_type: str) -> OrderedDict:
#     """
#     Get model weights from cloud bucket
#
#     - model_type = choice one of model type in [CUSTOM_MODEL_NAME]
#
#     """
#
#     def find_file(filename, path="."):
#         bag = os.listdir(path)
#         if filename in bag:
#             return os.path.join(path, filename)
#         else:
#             for b in bag:
#                 if os.path.isdir(os.path.join(path, b)):
#                     result = find_file(filename, os.path.join(path, b))
#                     if result:
#                         return result
#
#     pull_path = find_file(filename=".env", path=os.path.dirname(__file__))
#     if not pull_path:
#         raise Exception("Must be .env file at the same location with utility.py")
#
#     if model_type.upper() not in ["CUSTOM", "MODEL", "NAME", "AS", "BEAT"]:
#         raise Exception(f'There is no type {model_type} choose one of ["CUSTOM", "MODEL", "NAME", "AS", "BEAT"]')
#
#     # env_values = dotenv_values(pull_path)
#
#     # endpoint = env_values['END_POINT']
#     # bucket_name = env_values['BUCKET_NAME']
#     # model_path = env_values[model_type.upper()]
#     # model_name = model_path.split('/')[-1]
#     #
#     # download(endpoint=endpoint, bucket_name=bucket_name, model_path=model_path)
#     #
#     # state_dict = torch.load(model_name, map_location='cpu')
#     # os.remove(model_name)
#     #
#     # return state_dict
#     return {}
#
#
# def minio_download(
#     endpoint: str,
#     bucket_name: str,
#     model_path: str,
#     access_key: str = "a-key",
#     secret_key: str = "s-key",
# ):
#     """
#     Download the weight from the given URL.
#     There is no return, but write a file to the device.
#     :param endpoint: str
#     :param bucket_name: str
#     :param model_path: int, optional
#     :param access_key:str
#     :param secret_key :str
#     """
#     client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=False)
#
#     model_name = model_path.split("/")[-1]
#
#     client.fget_object(bucket_name=bucket_name, object_name=model_path, file_path=model_name)
#
#
# class WeightLoader:
#     state_dict = {
#         "segmentation": None,
#         "peak": None,
#         "classification": None,
#     }
#
#     def __init__(self, model_type: str):
#         w_dict = load_weights(model_type)
#         self.set_w(model_type, w_dict)
#
#     @staticmethod
#     def set_w(model_type: str, w_dict: dict):
#         WeightLoader.state_dict[model_type] = w_dict
#
#     @classmethod
#     def get(cls, model_type: str):
#         if cls.state_dict[model_type] is None:
#             cls(model_type)
#         return cls.state_dict[model_type]
