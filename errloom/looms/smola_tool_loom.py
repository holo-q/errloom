import json
from typing import Any, Callable, Dict, List

from datasets import Dataset

from errloom.holophore import Holophore
from errloom.holoware import ClassSpan
from errloom.utils.openai_chat import MessageList
from errloom import FnRule
from errloom.parsing.smola_parser import SmolaParser
from errloom.hol.system_prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from errloom.attractors.smola_tool_attractor import SmolaToolAttractor
from errloom.looms.multiturn_loom import MultiTurnLoom

class SmolaToolLoom(MultiTurnLoom):
    def __init__(self,
                 data_train: Dataset | None = None,
                 data_bench: Dataset | None = None,
                 tools: List[Any] = [],
                 system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
                 few_shot: MessageList = [],
                 mask_env_response: bool = True,
                 max_turns: int = 10, **kwargs):
        # Format the system prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        super().__init__(
            roll_dataset=data_train,
            eval_dataset=data_bench,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_turns=max_turns,
            **kwargs
        )
        self.dataset_name = data_train
        self.max_turns = max_turns
        self.tools = {tool.name: tool for tool in tools}
        self.attractor = SmolaToolAttractor(tools=tools)
        self.llm_parser = SmolaParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = SmolaParser(fields=["result"])

    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Formats tool schemas into a user-friendly description string."""
        descriptions = []
        for tool in tools:
            desc = [f"{tool.name}: {tool.description}"]

            desc.append("\nArguments:")
            for arg_name, arg_info in tool.inputs.items():
                desc.append(f"  - {arg_name}: {arg_info['description']}")

            desc.append(f"\nReturns: {tool.output_type}")

            descriptions.append("\n".join(desc))

        return "\n\n".join(descriptions)

    def get_attraction_rules(self, **kwargs: Any) -> List[FnRule]:
        return self.attractor.get_rule_funcs()

    def get_rule_weights(self, **kwargs: Any) -> List[float]:
        return self.attractor.get_rule_weights()

    def _get_step_count(self, messages: MessageList) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0

        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)

        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count

    def is_completed(self, messages: MessageList, state: Dict[str, Any], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_turns:
                return True

            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a SmolaAgents Tool object based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object, e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name', e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

            tool = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                return f"Error: Arguments for {tool_name} must be a JSON object matching the tool's input schema, not a string."

            # Call the tool object with arguments
            result = tool(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
        except Exception as e:
            return f"Error: {str(e)}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

    def loom_response(self, messages: MessageList, state: Dict[str, Any], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(result=result)}, {}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}, {}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, {}


# Holoware implementation

class HoloRule:
    """Base class for a Holoware rule within an Attractor."""
    def __init__(self, holoware, **kwargs):
        self.holoware = holoware

    def __holo__(self, context, **kwargs):
        """
        When a HoloRule is executed, it registers itself to the context.
        """
        if not hasattr(context, 'rules'):
            context.rules = []
        context.rules.append(self)

    def eval(self, messages: MessageList, **kwargs) -> float:
        """
        Applies the rule to the last message and returns a score.
        A higher score indicates better alignment with the rule.
        """
        raise NotImplementedError

class CorrectAnswerRule(HoloRule):
    """
    Checks if the model has provided a final answer.
    """
    def __init__(self, holoware, **kwargs):
        super().__init__(holoware, **kwargs)
        self.llm_parser = SmolaParser(fields=[("answer",)])

    def eval(self, messages: MessageList, **kwargs) -> float:
        """
        Returns 1.0 if a final answer is present, 0.0 otherwise.
        """
        if not messages or messages[-1]["role"] != "assistant":
            return 0.0

        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return 1.0
        except Exception:
            return 0.0
        return 0.0

class ToolRule(HoloRule):
    """
    Checks if the model has correctly called a tool.
    """
    def __init__(self, holoware, **kwargs):
        super().__init__(holoware, **kwargs)
        self.llm_parser = SmolaParser(fields=[("tool",)])

    def eval(self, messages: MessageList, **kwargs) -> float:
        """
        Returns 1.0 if a tool call is present, 0.0 otherwise.
        """
        if not messages or messages[-1]["role"] != "assistant":
            return 0.0

        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                return 1.0
        except Exception:
            return 0.0
        return 0.0

# class Attractor:
#     """
#     A Holotype that collects and applies a set of HoloRules to a generation.
#     It supports receiving rules as arguments or collecting them from the context
#     where they have been registered by HoloRule holotypes in the body.
#     """
#     def __init__(self, holoware, **kwargs):
#         self.holoware = holoware
#
#     def __holo__(self, holophore: Holophore, span: ClassSpan, *args, **kwargs) -> Dict[str, float]:
#         """
#         Applies rules to a generation. Rules can be provided as arguments
#         or as Holotypes in the body (which register to the context).
#         """
#         rules = []
#
#         # Syntax 1: Rules as class name arguments.
#         # e.g. <|Attractor CorrectAnswerRule ToolRule|>
#         for rule_name in args:
#             RuleClass = globals().get(rule_name)
#             if RuleClass and issubclass(RuleClass, HoloRule):
#                 rules.append(RuleClass(holoware=self.holoware))
#
#         # Syntax 2: Rules registered in the context from a body.
#         # e.g.
#         # <|Attractor|>
#         #     <|CorrectAnswerRule|>
#         #     <|ToolRule|>
#         if span.body:
#             holophore = span.body()
#
#         if context and hasattr(context, 'rules'):
#             # Collect and consume the rules from the context
#             rules.extend(context.rules)
#             context.rules = []
#
#         # Apply the collected rules and get their scores
#         scores = {}
#         for rule in rules:
#             rule_name = rule.__class__.__name__
#             scores[rule_name] = rule.eval(messages, **kwargs)
#         return scores

class ToolTurns:
    """
    A Holotype to manage multi-turn tool-use conversations.
    """
    def __init__(self, holoware, max_turns: int = 10, **kwargs):
        self.holoware = holoware
        self.max_turns = int(max_turns)
        self.tools = kwargs.get('tools', {})
        self.llm_parser = SmolaParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = SmolaParser(fields=["result"])

    def __holo__(self, messages: MessageList, sample_fn: Callable, **kwargs) -> MessageList:
        """
        Runs the multi-turn conversation.
        """
        completion_messages = []
        turn = 0
        while turn < self.max_turns:
            if self._is_completed(messages):
                break

            response_content = sample_fn(context=messages)

            assistant_message = {"role": "assistant", "content": response_content}
            messages.append(assistant_message)
            completion_messages.append(assistant_message)
            turn += 1

            if self._is_completed(messages):
                break

            env_msg = self._loom_response(messages)
            messages.append(env_msg)
            completion_messages.append(env_msg)

        return completion_messages

    def _is_completed(self, messages: MessageList) -> bool:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def _call_tool(self, tool_json: str) -> str:
        """Calls a tool based on its JSON representation."""
        try:
            command = json.loads(tool_json)
            tool_name = command.get("name")
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'."

            tool = self.tools[tool_name]
            tool_args = command.get("args", {})
            result = tool(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format."
        except Exception as e:
            return f"Error: {str(e)}."

    def _loom_response(self, messages: MessageList) -> Dict[str, str]:
        """Generates a response from the tool environment."""
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self._call_tool(parsed.tool)
                if result:
                    return {"role": "user", "content": self.env_parser.format(result=result)}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format."}