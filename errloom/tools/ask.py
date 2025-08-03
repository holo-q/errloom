import os

from errloom.holoware.holophore import Holophore
from errloom.holoware.holoware import ClassSpan

BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY = os.getenv("DEEPINFRA_API_KEY")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def python(code: str) -> str:
    """Evaluates a block of Python code and returns output of print() statements. Allowed libraries: astropy, biopython, networkx, numpy, scipy, sympy.

    Args:
        code (str): A block of Python code

    Returns:
        The output of the code (truncated to 1000 chars) or an error message

    Examples:
        {"code": "import numpy as np; print(np.array([1, 2, 3]) + np.array([4, 5, 6]))"} -> "[5 7 9]"
        {"code": "import scipy; print(scipy.linalg.inv(np.array([[1, 2], [3, 4]])))"} -> "[[-2.   1. ] [ 1.5 -0.5]]"
        {"code": "import sympy; x, y = sympy.symbols('x y'); print(sympy.integrate(x**2, x))"} -> "x**3/3"
    """

    import subprocess
    try:
        # Run the code block in subprocess with 10-second timeout
        result = subprocess.run(
            ['python', '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            text=True
        )
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        output = result.stdout.strip() if result.stdout else ""
        if len(output) > 1000:
            output = output[:1000] + "... (truncated to 1000 chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"

def calculator(expression: str) -> str:
    """Evaluates a single line of Python math expression. No imports or variables allowed.

    Args:
        expression (str): A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

    Returns:
        The result of the calculation or an error message

    Examples:
        "2 + 2" -> "4"
        "3 * (17 + 4)" -> "63"
        "100 / 5" -> "20.0"
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_url_markdown(url: str) -> str:
    """Get contents of URL as nicely formatted markdown."""
    import requests
    from markdownify import markdownify as md
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return md(response.text)
    except Exception as e:
        return f"Error: {str(e)}"

def ask(question: str, url: str) -> str:
    """Ask a question about a web page returned from search results.

    Args:
        question: The question to be answered (by an LLM who will be given the web page contents)
        url: The URL of the web page to query

    Returns:
        A LLM-generated answer to the question based on the web page contents.

    Examples:
        {"question": "What is the capital of France?", "url": "https://en.wikipedia.org/wiki/France"} -> "The capital of France is Paris."
        {"question": "How many people live in the United States?", "url": "https://en.wikipedia.org/wiki/United_States"} -> "The population of the United States is approximately 340 million people."
    """

    contents = get_url_markdown(url)[:50000]

    if contents.startswith("Error:"):
        return "Error: Failed to fetch URL contents."

    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = f"""Answer the following question based on the provided web page contents:

    Question: {question}

    Page: {url}

    Page contents:
    {contents}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return response.choices[0].message.content or "Error: No response from model."
    except Exception as e:
        return f"Error: {str(e)}"

def search_ddg(query: str, num_results: int = 5) -> str:
    """Searches DuckDuckGo and returns concise summaries of top results.

    Args:
        query (str): The search query string
        num_results (int): Number of results to return (default: 5, max: 10)

    Returns:
        Formatted string with bullet points of top results, each with title and brief summary

    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """

    try:
        from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(num_results, 10)))
            if not results:
                return "No results found"

            summaries = []
            for r in results:
                title = r['title']
                snippet = r['body'][:200].rsplit('.', 1)[0] + '.'
                summaries.append(f"• {title}\n  {snippet}")

            return "\n\n".join(summaries)
    except Exception as e:
        return f"Error: {str(e)}"

def search(query: str) -> str:
    """Searches the web and returns summaries of top results.

    Args:
        query: The search query string

    Returns:
        Formatted string with bullet points of top 3 results, each with title, source, url, and brief summary

    Examples:
        {"query": "who invented the lightbulb"} -> ["Thomas Edison (1847-1931) - Inventor of the lightbulb", ...]
        {"query": "what is the capital of France"} -> ["Paris is the capital of France", ...]
        {"query": "when was the Declaration of Independence signed"} -> ["The Declaration of Independence was signed on July 4, 1776", ...]
    """

    try:
        from brave import Brave  # type: ignore
        brave = Brave()
        results = brave.search(q=query, count=10, raw=True)  # type: ignore
        web_results = results.get('web', {}).get('results', [])  # type: ignore

        if not web_results:
            return "No results found"

        summaries = []
        for r in web_results:
            if 'profile' not in r:
                continue
            header = f"{r['profile']['name']} ({r['profile']['long_name']})"
            title = r['title']
            snippet = r['description'][:300] + " ..."
            url = r['url']
            summaries.append(f"•  {header}\n   {title}\n   {snippet}\n   {url}")

        return "\n\n".join(summaries[:3])
    except Exception as e:
        return f"Error: {str(e)}"

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
        if not holophore.contexts or not holophore.context.messages:
            return "Error: No messages to process."

        last_message = holophore.active_context.messages[-1]
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
