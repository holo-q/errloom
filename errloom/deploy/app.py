from prompt_toolkit import Application

class App(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = None
        self.just_started = True
        self.state_enters = {}
        self.state_exits = {}

    def change_state(self, new_state):
        if self.state == new_state:
            return

        # Safely call the exit handler for the current state, if it exists
        if self.state in self.state_exits:
            self.state_exits[self.state]()

        self.state = new_state

        # Safely call the enter handler for the new state, if it exists
        if new_state in self.state_enters:
            self.state_enters[new_state]()
