from src.deploy.deploy_utils import is_scrolled_to_bottom, scroll_to_end

from prompt_toolkit import ANSI
from prompt_toolkit.layout import FormattedTextControl
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea
import re


class LogDisplay:
    def __init__(self, 
                 scrollbar=True, 
                 wrap_lines=True, 
                 read_only=True,
                 focusable=True
                 ):
        self.text_area = TextArea(
            scrollbar=scrollbar,
            wrap_lines=wrap_lines,
            read_only=read_only,
            focusable=focusable
        )
        self._ansi_text = ""


    def update_text(self, new_text):
        if self._ansi_text != new_text:
            self._ansi_text = new_text
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            plain_text = ansi_escape.sub('', new_text)
            self.text_area.text = plain_text
            if not self.is_scrolled_to_bottom():
                self.scroll_to_end()

    def add_text(self, new_text):
        # Append the new text to the existing ANSI text
        self._ansi_text += new_text
        
        # Remove ANSI escape sequences for the plain text version
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        plain_text = ansi_escape.sub('', new_text)
        
        # Append the plain text to the TextArea
        self.text_area.text += plain_text
        
        # Scroll to the end if we were already at the bottom
        self.scroll_to_end()

    def set_formatted_text(self, formatted_text):
        self.text_area.window.content = FormattedTextControl(ANSI(formatted_text))

    def scroll_page(self, direction):
        w = self.window
        b = self.text_area.buffer

        if w and w.render_info:
            if direction == 'down':
                # Scroll down one page
                line_index = min(
                    b.document.line_count - 1,
                    w.render_info.last_visible_line() + w.render_info.window_height
                )
            else:  # direction == 'up'
                # Scroll up one page
                line_index = max(0, w.render_info.first_visible_line() - w.render_info.window_height)

            w.vertical_scroll = max(0, line_index - w.render_info.window_height + 1)
            b.cursor_position = b.document.translate_row_col_to_index(line_index, 0)
    
    def scroll(self, lines):
        w = self.window
        b = self.text_area.buffer

        if w and w.render_info:
            current_line = w.render_info.first_visible_line()
            new_line = max(0, min(current_line + lines, b.document.line_count - 1))
            
            w.vertical_scroll = max(0, new_line - w.render_info.window_height + 1)
            b.cursor_position = b.document.translate_row_col_to_index(new_line, 0)

    def is_scrolled_to_bottom(self):
        return is_scrolled_to_bottom(self.text_area)

    def scroll_to_end(self):
        scroll_to_end(self.text_area)

    @property
    def window(self):
        return self.text_area.window

    def __pt_container__(self):
        return self.text_area