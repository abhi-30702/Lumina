from __future__ import annotations
import html
from PyQt6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MessageLabel(QLabel):
    def __init__(self, role: str, text: str, parent=None) -> None:
        prefix = "> USER //" if role == "user" else "LUMINA //"
        color = "#00D4FF" if role == "user" else "#00FF88"
        super().__init__(
            f'<span style="color:{color};font-weight:bold;">{prefix}</span> {html.escape(text)}',
            parent,
        )
        self.setWordWrap(True)
        self.setFont(QFont("Courier New", 10))
        self.setStyleSheet("color: #CCCCCC; padding: 4px 8px;")
        self.setTextFormat(Qt.TextFormat.RichText)


class ChatPanel(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: transparent; border: none;")
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(4)
        self.setWidget(self._container)
        self.verticalScrollBar().rangeChanged.connect(
            lambda _min, _max: self.verticalScrollBar().setValue(_max)
        )

    def add_message(self, role: str, text: str) -> None:
        label = MessageLabel(role, text)
        self._layout.addWidget(label)
