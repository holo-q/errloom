from typing import Generic, Sequence

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Window, ConditionalMargin, ScrollbarMargin
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from errloom.deploy import _T


class _BaseList(Generic[_T]):
    open_character = "("
    close_character = ")"
    container_style = "class:radio-list"
    default_style = "class:radio"
    selected_style = "class:radio-selected"
    checked_style = "class:radio-checked"
    multiple_selection = False
    show_scrollbar = True

    def __init__(self, values: Sequence[tuple[_T, str]], default_values: Sequence[_T] | None = None):
        self.values = values
        self.keys = [value for value, _ in values]
        self.current_values = [v for v in (default_values or []) if v in self.keys]
        self.current_value = next((v for v in (default_values or []) if v in self.keys), values[0][0])
        self._selected_index = self.keys.index(self.current_values[0] if self.current_values else values[0][0])

        self.control = FormattedTextControl(
            self._get_text_fragments,
            key_bindings=self._create_key_bindings(),
            focusable=True
        )

        self.window = Window(
            content=self.control,
            style=self.container_style,
            right_margins=[ConditionalMargin(
                margin=ScrollbarMargin(display_arrows=True),
                filter=Condition(lambda: self.show_scrollbar),
            )],
            dont_extend_height=True,
        )

    def _create_key_bindings(self):
        kb = KeyBindings()

        kb.add("up")(lambda event: self._move_cursor(-1))
        kb.add("down")(lambda event: self._move_cursor(1))
        kb.add("pageup")(lambda event: self._move_cursor(-self._get_page_size()))
        kb.add("pagedown")(lambda event: self._move_cursor(self._get_page_size()))
        kb.add("enter")(lambda event: self._handle_enter())
        kb.add(" ")(lambda event: self._handle_enter())
        kb.add(Keys.Any)(self._find)
        return kb

    def _move_cursor(self, offset: int):
        self._selected_index = max(0, min(len(self.values) - 1, self._selected_index + offset))

    def _get_page_size(self):
        return len(get_app().layout.current_window.render_info.displayed_lines)

    def _handle_enter(self):
        selected_value = self.values[self._selected_index][0]
        if self.multiple_selection:
            if selected_value in self.current_values:
                self.current_values.remove(selected_value)
            else:
                self.current_values.append(selected_value)
        else:
            self.current_value = selected_value

    def _find(self, event):
        char = event.data.lower()
        values = self.values[self._selected_index + 1:] + self.values[:self._selected_index + 1]
        for value, text in values:
            if to_formatted_text(text)[0][1].lower().startswith(char):
                self._selected_index = self.values.index((value, text))
                return

    def _get_text_fragments(self) -> StyleAndTextTuples:
        def mouse_handler(mouse_event: MouseEvent) -> None:
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._selected_index = mouse_event.position.y
                self._handle_enter()

        fragments = []
        for i, (value, text) in enumerate(self.values):
            checked = value in self.current_values if self.multiple_selection else value == self.current_value
            selected = i == self._selected_index
            style = f"{self.checked_style if checked else ''} {self.selected_style if selected else ''}"

            row = [
                (style, self.open_character),
                ("[SetCursorPosition]", "") if selected else ("", ""),
                (style, "*" if checked else " "),
                (style, self.close_character),
                (self.default_style, " "),
                *to_formatted_text(text, style=self.default_style),
                ("", "\n")
            ]
            fragments.extend((style, text, mouse_handler) for style, text in row)

        return fragments[:-1]  # Remove last newline

    def __pt_container__(self) -> Window:
        return self.window
