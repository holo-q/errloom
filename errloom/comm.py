import json

from pydantic import BaseModel

from errloom.holoware import holostatic

@holostatic
class CommModel(BaseModel):
    """
    A well-defined data object / schema for communication with a LLM.
    Just a Pydantic BaseModel with some additional nice things for intercommunication.
    We can dump the schema and pass it to the LLM to declare the critique fields to
    generate.
    """

    def __holo__(self) -> str:
        """Special method for holoware to inject content. Returns a compact schema."""
        return self.get_compact_schema(include_descriptions=True)

    @classmethod
    def get_json_schema(cls, indent=2) -> str:
        """
        Returns the JSON schema for the model as a formatted string.
        """
        return json.dumps(cls.model_json_schema(), indent=indent)

    @classmethod
    def get_compact_schema(cls, include_descriptions: bool = True) -> str:
        """
        Generates a compact, human-readable schema from the Pydantic model fields automatically.
        Can optionally include field descriptions as comments.
        """
        from typing import get_args, get_origin, Literal as PyLiteral, List as PyList
        import json

        lines = []

        # We use cls.model_fields which is available on Pydantic models
        for i, (fname, finfo) in enumerate(cls.model_fields.items()):
            field_type = finfo.annotation
            origin = get_origin(field_type)
            args = get_args(field_type)

            value = None
            if origin is PyLiteral:
                value = "|".join(map(str, args))
            elif origin is list or origin is PyList:
                value = [finfo.description or "list of items"]
            else:
                if field_type is not None and hasattr(field_type, '__name__'):
                    value = field_type.__name__
                elif field_type is not None:
                    value = str(field_type)
                else:
                    value = 'Any'  # Default for untyped fields

            # Use json.dumps to correctly format the value part of the key-value pair
            line = f'  "{fname}": {json.dumps(value)}'
            if i < len(cls.model_fields) - 1:
                line += ","

            # Add description as comment if requested
            if include_descriptions and finfo.description:
                line += f"  # {finfo.description}"

            lines.append(line)

        return "{\n" + "\n".join(lines) + "\n}"
