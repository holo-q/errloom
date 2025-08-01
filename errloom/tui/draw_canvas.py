from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from prompt_toolkit.formatted_text import StyleAndTextTuples


@dataclass
class DrawCanvas:
    """
    A singleton canvas for efficient drawing of styled text fragments.

    This class provides a set of utilities for drawing text-based UI elements,
    managing styles, and handling layout concerns.
    """
    fragments: StyleAndTextTuples = field(default_factory=list)
    total_width: int = 0
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DrawCanvas, cls).__new__(cls)
        return cls._instance

    def reset(self, total_width: int = 0):
        """Reset the canvas for a new drawing operation."""
        self.fragments = []
        self.total_width = total_width

    def text(self, style: str, text: str) -> 'DrawCanvas':
        """Add a styled text fragment."""
        self.fragments.append((style, text))
        return self

    def newline(self) -> 'DrawCanvas':
        """Add a newline."""
        return self.text('', '\n')

    def horizontal_line(self, char: str = '─', style: str = 'class:separator') -> 'DrawCanvas':
        """Add a horizontal line with specified character and style."""
        line = char * (self.total_width - 2)
        return self.text(style, f'{char}{line}{char}').newline()

    def pad(self, width: int, char: str = ' ', style: str = '') -> 'DrawCanvas':
        """Add padding with specified width, character, and style."""
        return self.text(style, char * width)

    def centered_text(self, text: str, style: str, width: Optional[int] = None) -> 'DrawCanvas':
        """Add centered text with given style and optional width."""
        width = width or self.total_width
        padding = max(0, width - len(text))
        left_pad = padding // 2
        right_pad = padding - left_pad
        return self.pad(left_pad).text(style, text).pad(right_pad)

    def bordered_text(self, text: str, style: str, width: Optional[int] = None,
                      top_char: str = '─', side_char: str = '│', corner_char: str = '+') -> 'DrawCanvas':
        """Add text surrounded by a border."""
        width = width or self.total_width
        content_width = width - 4  # Account for borders and padding

        self.text(style, f'{corner_char}{top_char * (width - 2)}{corner_char}').newline()
        self.text(style, f'{side_char} ').text(style, f'{text:<{content_width}}').text(style, f' {side_char}').newline()
        self.text(style, f'{corner_char}{top_char * (width - 2)}{corner_char}').newline()

        return self

    def table_row(self, cells: List[Tuple[str, int]], style: str, separator: str = ' ') -> 'DrawCanvas':
        """Add a table row with given cell contents, widths, and style."""
        for text, width in cells:
            self.text(style, f'{text:<{width}}')
            if separator:
                self.text(style, separator)
        return self.newline()

    def progress_bar(self, progress: float, width: int, style: str,
                     fill_char: str = '█', empty_char: str = '░') -> 'DrawCanvas':
        """Add a progress bar with given progress (0-1), width, and style."""
        fill_width = int(progress * width)
        empty_width = width - fill_width
        return self.text(style, fill_char * fill_width + empty_char * empty_width)

    def get_fragments(self) -> StyleAndTextTuples:
        """Get the accumulated fragments."""
        return self.fragments
