from src.deploy import vast_manager
from src.deploy.tui.button_list import ButtonList
from src.deploy.remote import LoomRemote
from src.deploy.remote_view import RemoteView
from prompt_toolkit.application import get_app


class RemoteList(ButtonList):
    """
    Displays DiscoRemoteViews
    """

    def __init__(self):
        super().__init__(data=["Empty!"], hidden_headers=['_remote'])
        self.instances = []
        # self.headers = ['index', 'cuda', 'model', 'price']

    @property
    def has_data(self):
        return len(self.data) > 0 and isinstance(self.data[0], RemoteView)

    async def add_instance(self, remote: LoomRemote):
        self.data.append(await RemoteView.from_remote(remote))
        self.update()
        self.enable_confirm = True

    async def fetch_instances(self):
        self.enable_confirm = False
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        infos = await vast_manager.instance.fetch_instances()
        self.data = [await (await remote_manager.instance.get_remote(info)).to_view() for info in infos]
        self.update()

        self.enable_confirm = True
        get_app().invalidate()

    def update(self):
        if not self.data:
            self.data = ["No instances found. Press enter to create !"]
            self.enable_confirm = False


