from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions
from openai.resources.completions import Completions as LegacyCompletions
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.completion import Completion


class MockCompletions(Completions):
    def __init__(self, client: OpenAI, placeholder: str):
        super().__init__(client)
        self.placeholder = placeholder

    def create(self, *args, **kwargs) -> ChatCompletion:
        return ChatCompletion(
            id="mock-completion",
            choices=[Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=self.placeholder,
                    role="assistant",
                ),
            )],
            created=12345,
            model="mock-model",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=5,
                prompt_tokens=5,
                total_tokens=10,
            ),
        )


class MockLegacyCompletions(LegacyCompletions):
    def __init__(self, client: OpenAI, placeholder: str):
        super().__init__(client)
        self.placeholder = placeholder

    def create(self, *args, **kwargs) -> Completion:
        # Need to create a simple object that has the required attributes
        # Since the Completion type might be complex, let's create a minimal mock
        class MockChoice:
            def __init__(self, text: str):
                self.finish_reason = "stop"
                self.index = 0
                self.text = text
        
        class MockCompletion:
            def __init__(self, placeholder: str):
                self.id = "mock-completion"
                self.choices = [MockChoice(placeholder)]
                self.created = 12345
                self.model = "mock-model"
                self.object = "text_completion"
                self.usage = CompletionUsage(
                    completion_tokens=5,
                    prompt_tokens=5,
                    total_tokens=10,
                )
        
        return MockCompletion(self.placeholder)  # type: ignore


class MockChat(Chat):
    def __init__(self, client: OpenAI, placeholder: str):
        super().__init__(client)
        self._completions_mock = MockCompletions(client, placeholder)

    @property
    def completions(self) -> Completions:
        return self._completions_mock


class MockClient(OpenAI):
    def __init__(self, placeholder: str = "MOCK", **kwargs):
        super().__init__(api_key="mock-key", base_url="http://mock.url/v1")
        self.placeholder = placeholder
        self._chat_mock = MockChat(self, placeholder)
        self._completions_mock = MockLegacyCompletions(self, placeholder)

    @property
    def chat(self) -> Chat:
        return self._chat_mock

    @property
    def completions(self) -> LegacyCompletions:
        return self._completions_mock
