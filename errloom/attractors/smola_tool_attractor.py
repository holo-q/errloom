import json
from typing import List, Any

from errloom.parsing.smola_parser import SmolaParser
from errloom.attractors.tool_attractor import ToolUserAttractor
from errloom.utils.eval_utils import evaluate_code

class SmolaToolAttractor(ToolUserAttractor):
    def __init__(self,
                 parser: SmolaParser = SmolaParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: SmolaParser = SmolaParser(fields=["result"]),
                 tools: List[Any] = []):
        super().__init__(parser, env_parser, tools)
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.name: tool for tool in tools}
        self.attraction_rules = [
            self.rule_correct_answer,
            self.parser.get_format_attraction_rule(),
        ]
        self.rule_weights = [
            1.0,
            0.2,
        ]
        for tool_name in self.tools.keys():
            self.add_rule(self.get_named_tool_attraction_rule(tool_name), weight=0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        return evaluate_code(code_str, answer, **kwargs)