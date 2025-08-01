from errloom.interop import vast_manager
from errloom.tui.button_list import ButtonList
from errloom.deploy.remote_view import RemoteView
from errloom.interop.vast_instance import VastInstanceView


from prompt_toolkit.application import get_app


class InstanceList(ButtonList):
    """
    Displays VastInstance
    """

    def __init__(self):
        super().__init__(data=["Empty!"], hidden_headers=['_remote'])
        self.instances = []

    @property
    def has_data(self):
        return len(self.data) > 0 and isinstance(self.data[0], RemoteView)

    # async def add_instance(self, remote: DiscoRemote):
    #     self.data.append(await VastInstanceView.from_instance(remote))
    #     self.update()
    #     self.enable_confirm = True

    async def fetch_instances(self):
        self.enable_confirm = False
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        infos = await vast_manager.instance.fetch_instances()
        self.data = [VastInstanceView.from_instance(info) for info in infos]
        self.update()

        self.enable_confirm = True
        get_app().invalidate()

    def update(self):
        if not self.data:
            self.data = ["No instances found. Press enter to create !"]
            self.enable_confirm = False