from prompt_toolkit.layout import Window


class Spacer(Window):
    def __init__(self, height=1):
        super().__init__(height=height)
