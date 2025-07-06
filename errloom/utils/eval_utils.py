import io
import json
import signal
import sys
from contextlib import redirect_stdout


def evaluate_code(code_str, answer, **kwargs) -> float:
    try:
        test_cases = json.loads(answer)['test_cases']
    except (json.JSONDecodeError, KeyError):
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
        finally:
            sys.stdin = sys.__stdin__

    return passed / total_cases if total_cases else 0.0 