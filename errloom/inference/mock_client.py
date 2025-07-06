from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


class MockCompletions(Completions):
    def __init__(self, client: OpenAI, placeholder: str):
        super().__init__(client)
        self.placeholder = placeholder

    def create(self, *args, **kwargs) -> ChatCompletion:
        message = ChatCompletionMessage(
            content=self.placeholder,
            role="assistant",
        )
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=message,
        )
        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=5,
            total_tokens=10,
        )
        return ChatCompletion(
            id="mock-completion",
            choices=[choice],
            created=12345,
            model="mock-model",
            object="chat.completion",
            usage=usage,
        )


class MockChat(Chat):
    def __init__(self, client: OpenAI, placeholder: str):
        super().__init__(client)
        self._completions_mock = MockCompletions(client, placeholder)

    @property
    def completions(self) -> Completions:
        return self._completions_mock


class MockClient(OpenAI):
    def __init__(self, placeholder: str = "[MOCK]", **kwargs):
        super().__init__(api_key="mock-key", base_url="http://mock.url/v1")
        self.placeholder = placeholder
        self._chat_mock = MockChat(self, placeholder)

    @property
    def chat(self) -> Chat:
        return self._chat_mock