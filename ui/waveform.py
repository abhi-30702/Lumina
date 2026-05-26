from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPainter, QColor, QPen


class WaveformWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._amplitudes: list[float] = [0.0] * 40
        self._active = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setMinimumHeight(40)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._amplitudes = [0.0] * len(self._amplitudes)
            self._timer.stop()
            self.update()
        else:
            self._timer.start(50)

    def push_amplitude(self, value: float) -> None:
        self._amplitudes.pop(0)
        self._amplitudes.append(min(1.0, max(0.0, value)))

    def _tick(self) -> None:
        if not self._active:
            return
        noise = float(np.random.uniform(0.1, 0.6))
        self.push_amplitude(noise)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        bar_w = max(2, w // len(self._amplitudes) - 2)
        color = QColor("#00FF88") if self._active else QColor("#004422")
        pen = QPen(color)
        pen.setWidth(bar_w)
        painter.setPen(pen)
        for i, amp in enumerate(self._amplitudes):
            x = int(i * (w / len(self._amplitudes))) + bar_w // 2
            bar_h = int(amp * h)
            y1 = (h - bar_h) // 2
            y2 = y1 + bar_h
            painter.drawLine(x, y1, x, y2)
