from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.styles import Style

from errloom.deploy.app import App

kb: KeyBindings = KeyBindings()
style: Style = Style.from_dict({
    'frame.border': '#888888',
    'frame.label': 'bg:#ffffff #000000',
    'button': 'bg:#cccccc #000000',
    'button.keymap-highlight-char': 'bg:#999999 bold underline',
    'button.pressed': 'bg:#2f833a #ffffff',
    # 'button.focused': 'bg:#007acc #ffffff',
    'radiolist': 'bg:#f0f0f0 #000000',
    'radiolist-selected': 'bg:#007acc #ffffff',
    'textarea': 'bg:#f0f0f0 #000000',
    'textarea.cursor': '#ff0000',
    'scrollbar.background': 'bg:#000000',
    'scrollbar.button': 'bg:#cccccc',
    # 'button-list': 'bg:#f0f0f0',
    # 'button-list-item': 'bg:#000000 #000000',
    'button-list-item-selected': 'bg:#cccccc #000000',
    'row-highlight': 'bg:#cccccc #000000',
    'column-even': 'bg:#050505',
    'column-odd': 'bg:#111111',
    'column-sorted': 'bg:#333333 #ffffff',
    'text-area.prompt': 'bg:#000000 #ffffff',
})
app = App(
    key_bindings=kb,
    mouse_support=True,
    full_screen=True,
    style=style,
    layout=Layout(Window())
)
