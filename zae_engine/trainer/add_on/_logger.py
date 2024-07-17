from functools import partial

import neptune as nep
from neptune.common.exceptions import NeptuneInvalidApiTokenException as InvalidTkn


class NeptuneLogger:
    def __init__(self, project_name: str, api_tkn: str = "", **kwargs):
        """
        Initial neptune tokens using given token.

        :param project_name: Logging path to let neptune track each experiment.
        :param api_tkn: Authorization token
        :param kwargs: Auxiliary arguments to track models.
        :return:
        """
        self.api_root = f"zae-park/{project_name}"
        self.api_tkn = api_tkn
        self.kwargs = kwargs

        self.run, self.model = None, None
        self.init()

    def default(self):
        self.run, self.model = None, None

    def init(self):
        # name = self.kwargs['name'] if 'name' in self.kwargs.keys() else 'Prediction model'
        # key = self.kwargs['key'] if 'key' in self.kwargs.keys() else 'MOD'

        if self.api_tkn:
            try:
                self.run = nep.init_run(project=self.api_root, api_token=self.api_tkn)
                # self.model = nep.init_model(name=name, key=key, project=self.api_root, api_token=self.api_tkn)
            except InvalidTkn as e:
                print(f'{"-" * 100}')
                print(
                    "Receive invalid api token. Fail to generate Neptune instance, "
                    "please check again @ https://app.neptune.ai/o/zae-park/-/projects"
                )
                print(f'{"-" * 100}')
                raise e
            else:
                self.add_state_checker(self.run)

    @staticmethod
    def add_state_checker(*objects):
        for obj in objects:
            obj.is_live = partial(lambda self: self._state.value != "stopped", obj)

    def log(self, key, value):
        if self.run.is_live():
            self.run[key].log(value)

    def eliminate(self):
        if isinstance(self.run, nep.metadata_containers.MetadataContainer):
            self.run.stop()
        if isinstance(self.model, nep.metadata_containers.MetadataContainer):
            self.model.stop()
