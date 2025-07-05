from typing import Literal

from verifiers.envs.qa_env import QuestionAnswerLoom


class SingleTurnLoom(QuestionAnswerLoom):
    """
    Environment for single-turn tasks (chat or completion).
    """

    def __init__(self, message_type: Literal["chat", "completion"] = "chat", **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type
