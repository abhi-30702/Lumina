from __future__ import annotations
import math
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from ui.waveform import WaveformWidget
from ui.chat_panel import ChatPanel
from config import config

DARK_BG = "#020C14"
CYAN = "#00D4FF"
GREEN = "#00FF88"
AMBER = "#FFAA00"
RED = "#FF3333"
GRAY = "#555555"


class ArcReactor(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self._angle = 0.0
        self._pulse = 0.0
        self._speaking = False
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking

    def _tick(self) -> None:
        self._angle = (self._angle + 1.5) % 360
        if self._speaking:
            self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = 60, 60, 50
        # Outer ring
        pen = QPen(QColor(CYAN), 3)
        p.setPen(pen)
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        # Spinning dash
        pen2 = QPen(QColor(GREEN), 4)
        p.setPen(pen2)
        x = cx + r * math.cos(math.radians(self._angle))
        y = cy + r * math.sin(math.radians(self._angle))
        p.drawPoint(int(x), int(y))
        # Inner ring (reverse)
        pen3 = QPen(QColor(CYAN).darker(150), 2)
        p.setPen(pen3)
        p.drawEllipse(cx - 30, cy - 30, 60, 60)
        # Core glow
        glow_r = int(12 + (6 * math.sin(self._pulse) if self._speaking else 0))
        brush = QBrush(QColor(CYAN if not self._speaking else GREEN))
        p.setBrush(brush)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - glow_r, cy - glow_r, glow_r * 2, glow_r * 2)


class HUDWindow(QMainWindow):
    message_signal = pyqtSignal(str, str)  # role, text
    status_signal = pyqtSignal(str, str)   # label, color
    speaking_signal = pyqtSignal(bool)

    def __init__(self, on_text_input: callable | None = None) -> None:
        super().__init__()
        self._on_text_input = on_text_input or (lambda t: None)
        self._session_start = datetime.now()
        self._setup_window()
        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_session_timer)
        self._timer.start(1000)
        self.message_signal.connect(self._on_message)
        self.status_signal.connect(self._on_status)
        self.speaking_signal.connect(self._on_speaking)

    def _setup_window(self) -> None:
        self.setWindowTitle("LUMINA v1.0")
        self.setFixedSize(config.window_width, config.window_height)
        self.setWindowOpacity(config.opacity)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        if config.always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"background-color: {DARK_BG};")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)

        # Title bar
        title_bar = QHBoxLayout()
        title = QLabel("◉ LUMINA v1.0")
        title.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {CYAN};")
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {GREEN};")
        self._status_label = QLabel("ONLINE")
        self._status_label.setFont(QFont("Courier New", 9))
        self._status_label.setStyleSheet(f"color: {GREEN};")
        btn_min = QPushButton("—")
        btn_min.setFixedSize(24, 20)
        btn_min.setStyleSheet(f"color: {CYAN}; background: transparent; border: none;")
        btn_min.clicked.connect(self.showMinimized)
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(24, 20)
        btn_close.setStyleSheet(f"color: {RED}; background: transparent; border: none;")
        btn_close.clicked.connect(self.close)
        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(self._status_dot)
        title_bar.addWidget(self._status_label)
        title_bar.addWidget(btn_min)
        title_bar.addWidget(btn_close)
        root.addLayout(title_bar)

        # Arc reactor
        arc_row = QHBoxLayout()
        arc_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._arc = ArcReactor()
        arc_row.addWidget(self._arc)
        root.addLayout(arc_row)

        # Chat panel
        self._chat = ChatPanel()
        self._chat.setMinimumHeight(220)
        root.addWidget(self._chat)

        # Waveform
        self._waveform = WaveformWidget()
        root.addWidget(self._waveform)

        # Text input
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("> _")
        self._input.setFont(QFont("Courier New", 10))
        self._input.setStyleSheet(
            f"background: #0A1A28; color: {CYAN}; border: 1px solid {CYAN}; padding: 4px;"
        )
        self._input.returnPressed.connect(self._submit_text)
        self._mic_btn = QPushButton("MIC")
        self._mic_btn.setFixedWidth(50)
        self._mic_btn.setStyleSheet(
            f"background: transparent; color: {CYAN}; border: 1px solid {CYAN}; padding: 4px;"
        )
        input_row.addWidget(self._input)
        input_row.addWidget(self._mic_btn)
        root.addLayout(input_row)

        # Status bar
        status_row = QHBoxLayout()
        self._state_label = QLabel("STATUS: LISTENING...")
        self._state_label.setFont(QFont("Courier New", 8))
        self._state_label.setStyleSheet(f"color: {CYAN};")
        self._session_label = QLabel("SESSION: 00:00:00")
        self._session_label.setFont(QFont("Courier New", 8))
        self._session_label.setStyleSheet(f"color: {GRAY};")
        status_row.addWidget(self._state_label)
        status_row.addStretch()
        status_row.addWidget(self._session_label)
        root.addLayout(status_row)

    def _submit_text(self) -> None:
        text = self._input.text().strip()
        if text:
            self._input.clear()
            self._on_text_input(text)

    def _update_session_timer(self) -> None:
        elapsed = datetime.now() - self._session_start
        total = int(elapsed.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        self._session_label.setText(f"SESSION: {h:02d}:{m:02d}:{s:02d}")

    def _on_message(self, role: str, text: str) -> None:
        self._chat.add_message(role, text)

    def _on_status(self, label: str, color: str) -> None:
        self._state_label.setText(f"STATUS: {label}")
        self._state_label.setStyleSheet(f"color: {color};")

    def _on_speaking(self, speaking: bool) -> None:
        self._arc.set_speaking(speaking)
        self._waveform.set_active(speaking)

    # Thread-safe public API
    def show_message(self, role: str, text: str) -> None:
        self.message_signal.emit(role, text)

    def set_status(self, label: str, color: str = CYAN) -> None:
        self.status_signal.emit(label, color)

    def set_speaking(self, speaking: bool) -> None:
        self.speaking_signal.emit(speaking)

    # Drag to move frameless window
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_pos"):
            self.move(self.pos() + event.globalPosition().toPoint() - self._drag_pos)
            self._drag_pos = event.globalPosition().toPoint()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    hud = HUDWindow()
    hud.show()
    hud.show_message("lumina", "Lumina online. Ready when you are.")
    hud.set_status("LISTENING", CYAN)
    sys.exit(app.exec())
