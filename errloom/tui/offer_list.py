from errloom.deploy import vast_manager
from errloom.tui.button_list import ButtonList
from prompt_toolkit.application import get_app


class OfferList(ButtonList):
    def __init__(self, handler):
        super().__init__(data=[(None, "Empty!")], handler=handler)
        self.offers = []

    async def fetch_offers(self):
        self.data = [(None, "Loading...")]
        get_app().invalidate()

        self.offers = await vast_manager.instance.fetch_offers()
        self.data = self.offers
        self._process_data()

        self.sort_column(self._cached_headers.index('price'))  # hehe

        get_app().invalidate()