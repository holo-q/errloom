import enum


class ExecutionState(enum.Enum):
    OFF = 0
    RUNNING = 1

    @classmethod
    def from_bool(cls, b):
        if not b:
            return ExecutionState.OFF
        else:
            return ExecutionState.RUNNING