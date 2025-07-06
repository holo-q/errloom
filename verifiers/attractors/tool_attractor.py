import json
from typing import Callable, List

from verifiers.holoware.openai_chat import MessageList
from verifiers.parsers import Parser, XMLParser
from verifiers.attractors import Attractor

tool_use_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

def rule_correct_answer(self, completion, answer, **kwargs) -> float:
    """
    Gravitate towards the correct answer.
    """
    response = str(tool_use_parser.parse_answer(completion))
    return 1.0 if answer == response else 0.0

def rule_executes_tool(completion: MessageList, **kwargs) -> float:
    """
    Gravitate towards tool usage, with regards to correctness and success.
    Uses XMLParser to identify proper tool calls.
    """
    tool_attempts = 0
    successful_executions = 0

    # Find assistant messages with tools and their responses
    for i, msg in enumerate(completion):
        if msg['role'] == 'assistant':
            # Use parser to check for tool tag
            parsed = tool_use_parser.parse(msg['content'])
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                # Found a properly formatted tool message
                if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                    tool_attempts += 1
                    # Check response with env_parser
                    parsed_response = result_parser.parse(completion[i + 1]['content'])
                    if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                        successful_executions += 1

    # Calculate reward
    if tool_attempts == 0:
        return 0.0

    return (successful_executions / tool_attempts)

# noinspection PyDefaultArgument
class ToolUserAttractor(Attractor):
    def __init__(self, tools: List[Callable] = []):
        super().__init__()
        self.tools = {
            tool.__name__ if hasattr(tool, '__name__') else str(tool): tool
            for tool in tools
        }
        self.attraction_rules = [
            rule_correct_answer,
            rule_executes_tool,
            self.parser.get_format_attraction_rule(),
        ]
        self.rule_weights = [
            1.0,
            0.2,
            0.2,
        ]
        for tool_name in self.tools.keys():
            self.attraction_rules.append(self.get_named_tool_attraction_rule(tool_name))
            self.rule_weights.append(0.0)
            self.attraction_rules.append(self.get_named_tool_count_attraction_rule(tool_name))
            self.rule_weights.append(0.0)
            self.attraction_rules.append(self.get_named_tool_attempt_attraction_rule(tool_name))
            self.rule_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import sys
        import signal
        from contextlib import redirect_stdout

        try:
            test_cases = json.loads(answer)['test_cases']
        except:
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith('```python'):
            code_str = code_str[9:]
        elif code_str.startswith('```'):
            code_str = code_str[3:]
        if code_str.endswith('```'):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return '\n'.join(line.strip() for line in output.splitlines())

        total_cases = 0
        passed = 0

        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test['input'])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test['output'])

                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1

            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__

        return passed / total_cases if total_cases else 0.0



    def get_named_tool_attraction_rule(self, tool_name: str) -> Callable:
        """
        Returns an attraction rule that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """

        def tool_attraction_rule(completion: MessageList, **kwargs) -> float:
            """
            Attraction rule that checks execution success for the {tool_name} tool.

            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json

            tool_attempts = 0
            successful_executions = 0

            # Find assistant messages with the specific tool and their responses
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if isinstance(command, dict) and command.get("name") == tool_name:
                                # Found a properly formatted tool message for the specific tool
                                if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                                    tool_attempts += 1
                                    # Check response with env_parser
                                    parsed_response = self.env_parser.parse(completion[i + 1]['content'])
                                    if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass

            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)

        # Create a function with the dynamic name based on tool_name
        tool_attraction_rule.__name__ = f"{tool_name}_attraction_rule"
        return tool_attraction_rule

    def get_named_tool_count_attraction_rule(self, tool_name: str) -> Callable:
        """
        Returns an attraction rule that counts the number of times the {tool_name} tool is used.
        """

        def tool_count_attraction_rule(completion: MessageList, **kwargs) -> float:
            """
            Attraction rule that counts the number of times the {tool_name} tool is used.
            """
            import json

            successful_executions = 0.0
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if isinstance(command, dict) and command.get("name") == tool_name:
                                # Found a properly formatted tool message for the specific tool
                                if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                                    parsed_response = result_parser.parse(completion[i + 1]['content'])
                                    if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass
            return successful_executions

        tool_count_attraction_rule.__name__ = f"{tool_name}_count_attraction_rule"
        return tool_count_attraction_rule

    def get_named_tool_attempt_attraction_rule(self, tool_name: str) -> Callable:
        """
        Returns an attraction rule that counts the number of times the {tool_name} tool is used.
        """

        def tool_attempt_attraction_rule(completion: MessageList, **kwargs) -> float:
            """
            Attraction rule that counts the number of times the {tool_name} tool is used.
            """
            import json

            attempted_executions = 0.0
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if isinstance(command, dict) and command.get("name") == tool_name:
                                attempted_executions += 1
                        except json.JSONDecodeError:
                            pass
            return attempted_executions

        tool_attempt_attraction_rule.__name__ = f"{tool_name}_attempt_attraction_rule"
        return tool_attempt_attraction_rule
