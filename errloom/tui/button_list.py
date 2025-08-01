import asyncio
import enum
from dataclasses import is_dataclass, fields
from typing import (
    Generic,
    TypeVar,
    Sequence,
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Coroutine,
)
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import (
    FormattedTextControl,
    Window,
    ScrollbarMargin,
    ConditionalMargin,
    Dimension,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import get_app
import logging

from prompt_toolkit.styles import Style

from errloom.tui.draw_canvas import DrawCanvas
from errloom.deploy.deploy_utils import forget

STYLE_SEPARATOR = "class:separator"
STYLE_HEADER = "class:header"
STYLE_ROW_HIGHLIGHT = "class:row-highlight"
STYLE_FOOTER = "class:footer"
STYLE_COL_ODD = "class:column-odd"
STYLE_COL_EVEN = "class:column-even"
STYLE_COL_SORTED = "class:column-sorted"

_T = TypeVar("_T")
Item = Union[tuple[_T, str], Dict[str, Any], Any]

draw = DrawCanvas()


class ButtonListError(Exception):
    """Custom exception for ButtonList errors."""

    pass


class ButtonList(Generic[_T]):
    open_character = "["
    close_character = "]"
    container_style = "class:button-list"
    default_style = "class:button-list-item"
    selected_style = "class:button-list-item-selected"
    show_scrollbar = True
    enable_sorting = True
    enable_confirm = True

    def __init__(
        self,
        data: Sequence[Item],
        handler: Optional[Callable] = None,
        headers: Optional[List[str]] = None,
        hidden_headers: Optional[List[str]] = None,
        first_item_as_headers: bool = False,
        separator_width: int = 1,
    ):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)

        if not data:
            raise ButtonListError(
                "ButtonList must be initialized with non-empty values."
            )

        self.data = data
        self.handler = handler
        self.separator_width = separator_width
        self.headers = headers
        self.hidden_headers = hidden_headers
        self._first_item_as_headers = first_item_as_headers
        self._selected_index = 0
        self._sort_column = None
        self._sort_ascending = True

        self.style = Style.from_dict(
            {
                "button-list": "bg:#000080",
                "header": "bold #ffffff bg:#000080",
                "row": "#ffffff",
                "selected-row": "bg:#4169E1 #ffffff",
                "footer": "italic #ffffff",
                "column-even": "bg:#1a1a1a",
                "column-odd": "bg:#262626",
            }
        )

        # Cached state
        self._cached_headers = None
        self._cached_keys = None
        self._cached_column_widths = None

        self.control = FormattedTextControl(
            self._get_draw,
            key_bindings=self._create_key_bindings(),
            focusable=True,
        )

        self.window = Window(
            content=self.control,
            style=self.container_style,
            right_margins=[
                ConditionalMargin(
                    margin=ScrollbarMargin(display_arrows=True),
                    filter=Condition(lambda: self.show_scrollbar),
                )
            ],
            dont_extend_height=False,
            dont_extend_width=False,
            width=Dimension(preferred=100),
        )

    def _process_data(self) -> Tuple[List[str], List[str]]:
        """
        Process input data to extract attribute names (keys) and determine display labels (headers).

        This function handles various input data types and header configurations, ensuring
        type consistency and handling edge cases.

        Returns:
            Tuple[List[str], List[str]]: Extracted keys and resolved headers.

        Raises:
            ButtonListError: For invalid data, unsupported types, or conflicting configurations.
        """
        # Validate input data
        if not self.data:
            raise ButtonListError("No values to process.")

        first_item = self.data[0]

        # Extract keys (attribute names) based on first item's type
        match first_item:
            case _ if isinstance(first_item, (int, float, str, bool)):
                # For simple types: Use a single key 'value'
                # Allows direct use of simple values
                keys = ["value"]
            case tuple() as t if len(t) >= 2:
                # For tuples: Use predefined keys ['value', 'display']
                # Useful for simple key-value pairs
                keys = ["value", "display"]
            case dict() as d:
                # For dictionaries: Use the dictionary keys directly
                # Allows flexible attribute sets
                keys = list(d.keys())
            case object() if hasattr(first_item, "__dict__"):
                # For objects with __dict__: Use object attributes
                # Supports custom classes with dynamic attributes
                keys = list(first_item.__dict__.keys())
            case object() if is_dataclass(first_item):
                # For dataclasses: Use field names
                # Efficiently handles dataclass structures
                keys = [f.name for f in fields(first_item)]
            case _:
                # Unsupported type: Raise an error
                raise ButtonListError(f"Unsupported item type: {type(first_item)}")

        # Resolve headers (display labels) from keys or user-provided headers
        headers = self.headers or keys

        match headers:
            case dict():
                # Dictionary mapping: Use provided mapping or fallback to key
                # Allows custom labeling for some or all keys
                headers = [headers.get(key, key) for key in keys]
            case list() | tuple() as h if len(h) == len(keys):
                # List/tuple replacement: Use provided headers if length matches
                # Enables complete custom labeling
                headers = list(h)
            case list() | tuple():
                raise ButtonListError("Number of headers must match number of keys.")
            case _ if headers is not keys:
                raise ButtonListError(f"Unsupported headers type: {type(headers)}")

        # Handle 'hidden_headers' option
        if self.hidden_headers:
            # Remove hidden headers from display
            headers = [
                header for header in headers if header not in self.hidden_headers
            ]
            keys = [key for key in keys if key not in self.hidden_headers]

        # Remove headers that start with _ prefix
        headers = [header for header in headers if not header.startswith("_")]
        keys = [key for key in keys if not key.startswith("_")]

        # Handle 'first_item_as_headers' option
        if self._first_item_as_headers:
            if self.headers:
                # Conflicting options: Raise an error
                raise ButtonListError(
                    "Cannot use both 'headers' and 'first_item_as_headers'."
                )
            # Use first item as headers: Override previous headers
            headers = keys
            # Remove first item from data as it's now used for headers
            self.data = self.data[1:]

        # Cache results for future use
        self._cached_keys = keys
        self._cached_headers = headers
        return keys, headers

    def _calculate_column_widths(
        self, keys: List[str], headers: List[str]
    ) -> List[int]:
        """
        Calculate the width of each column based on the content and headers.

        This function determines the maximum width needed for each column by considering
        the length of headers and the content of each data item. It supports various
        data types and handles edge cases to ensure consistent display.

        Args:
            keys (List[str]): The attribute names or keys for accessing data.
            headers (List[str]): The display labels for each column.

        Returns:
            List[int]: The calculated width for each column.
        """
        # Initialize column widths based on headers
        # Add space for separator and sort indicator
        column_widths = [
            len(str(header)) + self.separator_width * 2 + 2 for header in headers
        ]

        for entry in self.data:
            # Extract values based on entry type
            match entry:
                case tuple() as t if len(t) >= 2:
                    # For tuples: Use second element as display value, pad others
                    values = [str(t[1])] + [""] * (len(headers) - 1)
                case str():
                    # For strings: Use the string directly
                    values = [str(entry)]
                case dict() as d:
                    # For dictionaries: Extract values using keys, handle missing keys
                    values = [str(d.get(key, "")) for key in keys]
                case object() if hasattr(entry, "__dict__"):
                    # For objects: Extract attributes using keys, handle missing attributes
                    values = [str(getattr(entry, key, "")) for key in keys]
                case _:
                    # Unsupported type: Use empty strings
                    values = [""] * len(headers)

            # Ensure values has the same length as headers
            values += [""] * (len(headers) - len(values))

            # Update column widths based on content length
            column_widths = [
                max(current, len(value))
                for current, value in zip(column_widths, values)
            ]

        # Cache calculated widths for future use
        self._cached_column_widths = column_widths
        return column_widths

    def _get_headers(self):
        if self._cached_headers is None:
            self._process_data()
        return self._cached_headers

    def _get_keys(self):
        if self._cached_keys is None:
            self._process_data()
        return self._cached_keys

    def _get_column_widths(self):
        if self._cached_column_widths is None:
            keys, headers = self._process_data()
            self._calculate_column_widths(keys, headers)
        return self._cached_column_widths

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        kb.add("up")(lambda event: self.move_cursor(-1))
        kb.add("down")(lambda event: self.move_cursor(1))
        kb.add("left")(lambda event: self.sort_column_dir(-1))
        kb.add("right")(lambda event: self.sort_column_dir(1))
        kb.add("pageup")(lambda event: self.move_cursor(-self._get_page_size()))
        kb.add("pagedown")(lambda event: self.move_cursor(self._get_page_size()))
        kb.add("enter")(lambda event: self.confirm_selection())
        kb.add(" ")(lambda event: self.confirm_selection())
        kb.add(Keys.Any)(self._find)

        # Sorting for columns 1-9
        for i in range(9):

            @kb.add(str(i + 1))
            def _(event, i=i):
                self.sort_column(i)

        # Sorting for columns 10-20
        shift_number_map = {
            "!": 10,
            "@": 11,
            "#": 12,
            "$": 13,
            "%": 14,
            "^": 15,
            "&": 16,
            "*": 17,
            "(": 18,
            ")": 19,
            "_": 20,  # This is Shift+- for the 20th column
        }

        for key, column in shift_number_map.items():

            @kb.add(key)
            def _(event, column=column):
                if column < len(self._get_headers()):
                    self.sort_column(column - 1)

        return kb

    def _get_page_size(self) -> int:
        app = get_app()
        if (
            app.layout
            and app.layout.current_window
            and app.layout.current_window.render_info
        ):
            return len(app.layout.current_window.render_info.displayed_lines)
        return 10  # Default value if unable to determine

    def _find(self, event: Any) -> None:
        """
        Find and select the next item starting with the given character.

        This function performs a case-insensitive search through the data,
        starting from the currently selected item and wrapping around to the
        beginning if necessary. It supports various data types and handles
        edge cases to ensure consistent behavior.

        Args:
            event (Any): The event containing the search character.

        Returns:
            None
        """
        if not self.data:
            return

        char = event.data.lower()
        start_index = self._selected_index
        headers = self._get_headers()

        for i in range(len(self.data)):
            # Calculate index with wrap-around
            index = (start_index + i + 1) % len(self.data)
            entry = self.data[index]

            # Extract searchable text based on entry type
            text = self._extract_searchable_text(entry, headers[0])

            if text.lower().startswith(char):
                self._selected_index = index
                return

    def _extract_searchable_text(self, entry: Any, key: str) -> str:
        """
        Extract searchable text from an entry based on its type.

        Args:
            entry (Any): The data entry to extract text from.
            key (str): The key or attribute name to use for dict/object entries.

        Returns:
            str: The extracted text, or an empty string if extraction fails.
        """
        match entry:
            case tuple() if len(entry) >= 2:
                return str(entry[1])
            case dict():
                return str(entry.get(key, ""))
            case object() if hasattr(entry, "__dict__"):
                return str(getattr(entry, key, ""))
            case _:
                return ""

    def move_cursor(self, offset: int) -> None:
        if not self.data:
            return
        self._selected_index = max(
            0, min(len(self.data) - 1, self._selected_index + offset)
        )

    def confirm_selection(self) -> None:
        # self.log.info(f"Confirming selection: {self._selected_index} ({self.enable_confirm}, {self.data})")

        if not self.enable_confirm:
            return
        if not self.data:
            return

        item = self.data[self._selected_index]

        def execute_handler(handler: Union[Callable, Coroutine], *args):
            try:
                if asyncio.iscoroutinefunction(handler):
                    forget(handler(*args))
                else:
                    handler(*args)
            except Exception as e:
                self.log.error(f"Error in handler: {str(e)}")
                raise ButtonListError(f"Error in handler: {str(e)}")

        if self.handler:
            execute_handler(self.handler, item)
        elif isinstance(item, tuple) and len(item) > 0 and callable(item[0]):
            execute_handler(item[0])

    def sort_column(self, column_index: int) -> None:
        if not self.enable_sorting:
            return

        headers = self._get_headers()
        if column_index < 0 or column_index >= len(headers):
            # self.log.error(f"Invalid column index: {column_index}")
            return

        if self._sort_column == column_index:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = column_index
            self._sort_ascending = True

        key = headers[column_index]
        try:
            self.data = sorted(
                self.data,
                key=lambda x: (
                    x.get(key, "")
                    if isinstance(x, dict)
                    else getattr(x, key, "")
                    if hasattr(x, "__dict__")
                    else x[1]
                    if isinstance(x, tuple)
                    else ""
                ),
                reverse=not self._sort_ascending,
            )
        except Exception as e:
            self.log.error(f"Error sorting values: {str(e)}")
            raise ButtonListError(f"Error sorting values: {str(e)}")

    def sort_column_dir(self, direction: int) -> None:
        if self._sort_column is None:
            return

        next_column = self._sort_column + direction
        next_column = next_column % len(self._get_headers())
        self.sort_column(next_column)

    def _get_draw(self) -> StyleAndTextTuples:
        """Generate the formatted content for drawing the button list."""
        keys, headers = self._process_data()
        column_widths = self._calculate_column_widths(keys, headers)

        draw.reset(total_width=sum(column_widths) + len(column_widths) + 1)
        draw.horizontal_line()
        if headers:
            self._draw_header_rows(headers, column_widths)
            draw.horizontal_line()
        self._draw_content_rows(keys, headers, column_widths)
        draw.horizontal_line()
        self._draw_footer()

        return draw.get_fragments()

    def _draw_header_rows(self, headers: List[str], column_widths: List[int]):
        """Draw the header row with sort indicators."""
        cells = []
        for i, (header, width) in enumerate(zip(headers, column_widths)):
            is_sorted_col = i == self._sort_column
            sort_indicator = self._get_sort_indicator(is_sorted_col)
            header_text = f"{sort_indicator}{header}"
            cells.append((header_text, width))
        draw.table_row(cells, STYLE_HEADER)

    def _draw_content_rows(
        self, keys: List[str], headers: List[str], column_widths: List[int]
    ):
        """Draw the content rows."""
        for row_index, item in enumerate(self.data):
            is_selected_row = row_index == self._selected_index

            if is_selected_row:
                draw.text("[SetCursorPosition]", "")

            try:
                column_text_entries = self._get_column_text_entries(item, keys, headers)
                cells = list(zip(column_text_entries, column_widths))
                style = self.get_cell_style(row_index)
                if is_selected_row:
                    style = STYLE_ROW_HIGHLIGHT
                draw.table_row(cells, style)
            except Exception as e:
                self.log.error(f"Error formatting item {row_index}: {str(e)}")
                draw.text("class:error", f"Error: {str(e)}").newline()

    def _draw_footer(self):
        """Draw the footer with item count."""
        footer_text = f" Total items: {len(self.data)} "
        draw.centered_text(footer_text, STYLE_FOOTER)

    def _get_sort_indicator(self, is_sorted_col: bool) -> str:
        """Get the appropriate sort indicator for a column."""
        UP_ARROW_FA, DOWN_ARROW_FA = "\uf062", "\uf063"
        if not is_sorted_col:
            return "  "
        return f"{UP_ARROW_FA} " if self._sort_ascending else f"{DOWN_ARROW_FA} "

    def _get_column_text_entries(
        self, item: Any, keys: List[str], headers: List[str]
    ) -> List[str]:
        """
        Extract text entries for each column based on item type.

        Handles enums by returning their name or value appropriately.
        """

        def format_value(value: Any) -> str:
            if isinstance(value, enum.Enum):
                if isinstance(value.value, str):
                    return value.value
                elif isinstance(value.value, (int, float)):
                    return value.name
                else:
                    return str(value.name)  # Fallback for other types
            return str(value)

        match item:
            case str():
                return [item]
            case tuple() if len(item) >= 2:
                return [format_value(item[1])] + [""] * (len(headers) - 1)
            case dict():
                return [format_value(item.get(key, "")) for key in keys]
            case object() if hasattr(item, "__dict__"):
                return [format_value(getattr(item, key, "")) for key in keys]
            case _:
                raise ButtonListError(f"Unsupported item type: {type(item)}")

    def get_cell_style(self, col_index):
        column_style = ""

        is_sort_column = col_index == self._sort_column
        if col_index % 2 == 0:
            column_style = STYLE_COL_EVEN
        else:
            column_style = STYLE_COL_ODD

        if is_sort_column:
            column_style += " " + STYLE_COL_SORTED

        return column_style

    def __pt_container__(self) -> Window:
        return self.window
