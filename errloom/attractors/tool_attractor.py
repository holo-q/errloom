import json
from typing import Callable, List

from errloom.utils.openai_chat import MessageList
from errloom.parsers import Parser, XMLParser
from errloom.attractor import Attractor
from errloom.utils.eval_utils import evaluate_code

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
    def __init__(self,
                 parser: Parser = tool_use_parser,
                 env_parser: Parser = result_parser,
                 tools: List[Callable] = []):
        super().__init__()
        self.parser = parser
        self.env_parser = env_parser
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
        return evaluate_code(code_str, answer, **kwargs)

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
