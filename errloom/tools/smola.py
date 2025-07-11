from errloom.holoware import Holophore, ClassSpan

class ToolPrompt:
    def __holo__(self, holophore: Holophore, node: ClassSpan, *args, **kwargs) -> str:
        """
        Formats tool descriptions for the prompt.
        This is triggered by a <|ToolPrompt tools|> tag.
        """
        tools_var_name = node.kargs[0]
        tools = holophore.rollout.env.get(tools_var_name, [])

        descriptions = []
        for tool in tools:
            desc = [f"{tool.name}: {tool.description}"]

            if hasattr(tool, 'inputs') and tool.inputs:
                desc.append("\nArguments:")
                for arg_name, arg_info in tool.inputs.items():
                    desc.append(f"  - {arg_name}: {arg_info['description']}")

            if hasattr(tool, 'output_type') and tool.output_type:
                desc.append(f"\nReturns: {tool.output_type}")

            descriptions.append("\n".join(desc))

        return "\n\n".join(descriptions)

class ToolRunner:
    def __holo__(self, holophore: Holophore, node: ClassSpan, *args, **kwargs) -> str:
        """
        Manages the multi-turn reasoning loop.
        This is triggered by a <|ToolRunner max_turns=N|> tag.
        """
        max_turns_str = node.kwargs.get('max_turns', '10')
        max_turns = int(max_turns_str)

        turn = holophore.env.get('turn', 0)

        if turn >= max_turns:
            holophore.completed = True
            return "Max turns reached. Concluding."

        holophore.env['turn'] = turn + 1
        return "" # Does not output text, just manages state

class ToolExecutor:
    def __holo__(self, holophore: Holophore, node: ClassSpan, *args, **kwargs) -> str:
        """
        Executes a tool based on the model's output.
        This is triggered by a <|ToolExecutor turn|> tag.
        """
        if not holophore.contexts or not holophore.contexts[-1].messages:
            return "Error: No messages to process."

        last_message = holophore.contexts[-1].messages[-1]
        if last_message['role'] != 'assistant':
            return "Error: Last message is not from assistant."

        content = last_message['content']

        # We need a parser for the LLM output.
        # This logic is based on smola_tool_loom.py, which uses SmolaParser.
        # Here we'll do a simplified extraction assuming a format.
        # A more robust solution would involve reusing or adapting SmolaParser.

        try:
            # Assuming format: <think>...</think><tool>...</tool>
            import re
            tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
            if not tool_match:
                # If no tool call, check for answer
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if answer_match:
                    holophore.rollout.completed = True
                return "" # No tool to execute

            tool_json_str = tool_match.group(1).strip()

            import json
            command = json.loads(tool_json_str)

            tool_name = command.get("name")
            tool_args = command.get("args", {})
            tools = holophore.rollout.env.get('tools', [])
            tool_map = {tool.name: tool for tool in tools}

            if tool_name not in tool_map:
                return f"Error: Unknown tool '{tool_name}'"

            tool_to_call = tool_map[tool_name]
            result = tool_to_call(**tool_args)

            # This result should be added as a new user message.
            # The __holo__ interface just returns a string to be appended.
            # So we format it and the Holoware runner will add it.
            # To create a new message, we'd need to modify the runner.
            # For now, we format it as if it's a response.
            return f"\n<result>{result}</result>"

        except Exception as e:
            return f"Error executing tool: {e}"
