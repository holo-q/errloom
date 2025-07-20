import asyncio
import inspect
from typing import Callable, Tuple, Union, Type

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPress, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Container, BufferControl, Window
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.widgets import Button, Label

from src.deploy.deploy_utils import forget, invalidate


# from src.deploy.ui_vars import *
def _get_containers_with_buffer(container):
    if isinstance(container, Container):
        yield container
        for child in container.get_children():
            yield from _get_containers_with_buffer(child)

class AppButton(Button):
    shortcut_enabled_types = set()

    @classmethod
    def register_shortcut_enabled_type(cls, widget_type: Type):
        cls.shortcut_enabled_types.add(widget_type)

    @classmethod
    def is_shortcut_enabled_widget_focused(cls):
        app = get_app()
        focused_element = app.layout.current_window
        return any(isinstance(focused_element, t) for t in cls.shortcut_enabled_types)

    def __init__(self, text, handler: Callable, key: Union[str, Tuple[str], None] = None):
        super().__init__(text,
                         handler=handler,
                         left_symbol='【',
                         right_symbol='】')

        self.is_pressed = False

        def global_focus_aware_handler(event):
            if self.is_shortcut_enabled_widget_focused():
                self._handler(event)

        # Register this type if it hasn't been already

        kb = KeyBindings()
        match key:
            case str():
                self.keybinding_char_indices = self.find_keybinding_char_indices((key), text)
                kb.add(key, is_global=True)(global_focus_aware_handler)
            case tuple():
                self.keybinding_char_indices = self.find_keybinding_char_indices(key, text)
                for k in key:
                    kb.add(k, is_global=True)(global_focus_aware_handler)
            case None:
                self.keybinding_char_indices = []

        @kb.add('enter')
        def _(event):
            self._handler(event)

        self.control.key_bindings = kb

        # Autosize button
        self.window.width = len(text) + 4

        def get_style() -> str:
            if self.is_pressed:
                return "class:button.pressed"
            elif get_app().layout.has_focus(self):
                return "class:button.focused"
            else:
                return "class:button"

        self.window.style = get_style

    def find_keybinding_char_indices(self, keys, text):
        text = text.lower()
        keys = [k.lower() for k in keys]
        return [text.find(k) for k in keys]

    def _handler(self, event):
        # if is_control_visible(app, self.window):
        #     self.handler()

        ret = None

        if self.handler is not None:
            is_shift_enter = isinstance(event, KeyPressEvent) and \
                             isinstance(event.key_sequence[0], KeyPress) and \
                             event.key_sequence[0].key == Keys.Enter and \
                             event.key_sequence[0].data == '\x1b[13;2u'

            params = inspect.signature(self.handler).parameters
            if len(params) > 0:
                # noinspection PyArgumentList
                ret = self.handler(is_shift_enter)
            else:
                ret = self.handler()
            invalidate()
            forget(self._animate_press())

        if inspect.iscoroutine(ret):
            forget(ret)

    def _get_text_fragments(self) -> StyleAndTextTuples:
        def mouse_handler(mouse_event: MouseEvent) -> None:
            if self.handler is not None and mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._handler(mouse_event)

        fragments = [
            ("class:button.arrow", self.left_symbol, mouse_handler),
            ("[SetCursorPosition]", ""),
        ]

        # Add text fragments with bolding if necessary
        for i, char in enumerate(self.text):
            style = "class:button.text"
            if i in self.keybinding_char_indices:
                # Change
                style = "class:button.keymap-highlight-char"

            fragments.append((style, char, mouse_handler))

        fragments.append(("class:button.text", " ", mouse_handler))
        fragments.append(("class:button.arrow", self.right_symbol, mouse_handler))

        return fragments

    def __pt_container__(self):
        return self.window

    async def _animate_press(self):
        self.is_pressed = True
        invalidate()

        await asyncio.sleep(0.1)

        self.is_pressed = False
        invalidate()

AppButton.register_shortcut_enabled_type(Button)
AppButton.register_shortcut_enabled_type(AppButton)
AppButton.register_shortcut_enabled_type(Label)
AppButton.register_shortcut_enabled_type(Window)
