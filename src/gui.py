"""PyQt live monitor: pick a behavior, watch RTSP with overlay and alerts."""

from __future__ import annotations

import time

import cv2
from PyQt6.QtCore import QPoint, QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.alerts import alert_title, format_alert
from src.pipeline import (
    TASK_LABELS,
    Pipeline,
    _open_capture,
    _probe_stream,
    _redact_source,
    apply_task_to_cfg,
    load_config,
)
from src.recorder import Recorder
from src.visualizer import BANNER_TEXT


TASK_ORDER = ("bag", "dual_entry", "open")

ALERT_ROW_COLORS = {
    "put_box_in_bag": QColor(70, 18, 18),
    "open_box": QColor(70, 42, 12),
    "dual_entry": QColor(72, 16, 28),
}


class StreamWorker(QThread):
    frame_ready = pyqtSignal(object)
    event_ready = pyqtSignal(dict, str, str)
    status = pyqtSignal(str)
    fps_ready = pyqtSignal(float)
    stats_ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    recording_changed = pyqtSignal(bool)
    pipeline_ready = pyqtSignal()

    def __init__(self, cfg: dict, source: str, task: str, pipeline: Pipeline | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.pipeline = pipeline
        self.source = source
        self.task = task
        self._stop = False
        self._pending_task: str | None = None
        self._toggle_record = False
        self._pending_roi: tuple[float, float, float, float] | None = None
        self._roi_update = False

    def stop(self) -> None:
        self._stop = True

    def request_task(self, task: str) -> None:
        self._pending_task = task

    def request_record_toggle(self) -> None:
        self._toggle_record = True

    def request_roi(self, roi: tuple[float, float, float, float] | None) -> None:
        self._pending_roi = roi
        self._roi_update = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — surface any stream/model failure in the UI
            self.failed.emit(str(exc))

    def _run(self) -> None:
        self.status.emit("Loading detector…")
        if self.pipeline is None:
            apply_task_to_cfg(self.cfg, self.task)
            self.cfg.setdefault("visualizer", {})["show"] = False
            self.cfg.setdefault("visualizer", {})["write"] = False
            self.pipeline = Pipeline(self.cfg)
        self.pipeline_ready.emit()
        self.pipeline.apply_task(self.task)
        self.pipeline.alerter.desktop = bool(self.cfg.get("alerts", {}).get("desktop", True))
        self.pipeline.alerter.sound = bool(self.cfg.get("alerts", {}).get("sound", True))

        src_cfg = self.pipeline.cfg.get("source", {})
        self.status.emit(f"Connecting {_redact_source(self.source)}…")
        cap = _open_capture(self.source, src_cfg)
        if cap is None or not cap.isOpened():
            self.failed.emit(f"Could not open {_redact_source(self.source)}")
            return
        try:
            fps, width, height, cap = _probe_stream(cap, self.source, src_cfg)
        except FileNotFoundError as exc:
            self.failed.emit(str(exc))
            return

        self.pipeline.recorder = Recorder(
            self.pipeline._recorder_cfg,
            self.pipeline.output_dir,
            fps,
            (width, height),
        )
        reconnect = bool(src_cfg.get("reconnect", True))
        reconnect_delay = float(src_cfg.get("reconnect_delay", 1.0))
        stride = int(src_cfg.get("stride", 1))
        self.status.emit(f"Live  {width}×{height} @ {fps:.0f} fps")

        frame_idx = 0
        t0 = time.time()
        shown = 0
        try:
            while not self._stop:
                if self._pending_task is not None:
                    self.task = self._pending_task
                    self._pending_task = None
                    self.pipeline.apply_task(self.task)
                    self.status.emit(f"Behavior: {TASK_LABELS.get(self.task, self.task)}")
                if self._roi_update:
                    self._roi_update = False
                    self.pipeline.set_roi(self._pending_roi)
                    self.status.emit("ROI updated" if self._pending_roi else "ROI cleared")
                if self._toggle_record:
                    self._toggle_record = False
                    if self.pipeline.recorder is not None:
                        self.pipeline.recorder.toggle_manual()
                        self.recording_changed.emit(bool(self.pipeline.recorder.is_manual_recording))

                ok, frame = cap.read()
                if not ok:
                    if reconnect and not self._stop:
                        self.status.emit("Stream dropped, reconnecting…")
                        cap.release()
                        time.sleep(reconnect_delay)
                        cap = _open_capture(self.source, src_cfg)
                        if cap is None or not cap.isOpened():
                            continue
                        continue
                    break
                if stride > 1 and frame_idx % stride != 0:
                    frame_idx += 1
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                vis, events = self.pipeline.process_frame(frame, frame_idx, fps)
                self.frame_ready.emit(vis)
                if self.pipeline.dual is not None:
                    self.stats_ready.emit(self.pipeline.dual.status_line())
                for event in events:
                    payload = event.to_dict()
                    if self.pipeline.recorder is not None and self.pipeline.recorder._clip_path is not None:
                        payload["clip"] = str(self.pipeline.recorder._clip_path)
                    self.event_ready.emit(
                        payload,
                        alert_title(event),
                        format_alert(event),
                    )
                frame_idx += 1
                shown += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self.fps_ready.emit(shown / elapsed)
                    t0 = time.time()
                    shown = 0
        finally:
            if self.pipeline.recorder is not None:
                self.pipeline.recorder.close()
                self.pipeline.recorder = None
            cap.release()
            self.status.emit("Disconnected")


class VideoView(QLabel):
    roi_drawn = pyqtSignal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setText("Connect to the camera to start live annotation")
        self.setObjectName("videoView")
        self._pixmap: QPixmap | None = None
        self._drawing = False
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None

    def set_drawing(self, enabled: bool) -> None:
        self._drawing = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        if not enabled:
            self._drag_origin = None
            self._drag_current = None
            self.update()

    def set_bgr_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drawing or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_origin = event.position().toPoint()
        self._drag_current = self._drag_origin
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is None:
            return
        self._drag_current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is None:
            return
        end = event.position().toPoint()
        start = self._drag_origin
        self._drag_origin = None
        self._drag_current = None
        self.update()
        n1 = self._widget_to_norm(start)
        n2 = self._widget_to_norm(end)
        if n1 is None or n2 is None:
            return
        x1, x2 = sorted((n1[0], n2[0]))
        y1, y2 = sorted((n1[1], n2[1]))
        if x2 - x1 < 0.03 or y2 - y1 < 0.03:
            return
        self.roi_drawn.emit((x1, y1, x2, y2))

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._drag_origin is None or self._drag_current is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(40, 200, 255), 2))
        painter.setBrush(QColor(40, 200, 255, 40))
        painter.drawRect(QRect(self._drag_origin, self._drag_current).normalized())

    def _content_rect(self) -> QRect | None:
        if self._pixmap is None:
            return None
        scaled = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def _widget_to_norm(self, pos: QPoint) -> tuple[float, float] | None:
        rect = self._content_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return None
        x = (pos.x() - rect.x()) / rect.width()
        y = (pos.y() - rect.y()) / rect.height()
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
        return (x, y)

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MonitorWindow(QMainWindow):
    def __init__(self, config_path: str = "config.yaml") -> None:
        super().__init__()
        self.setWindowTitle("Package room monitor")
        self.resize(1280, 800)
        self.cfg = load_config(config_path)
        self.pipeline: Pipeline | None = None
        self.worker: StreamWorker | None = None
        self._build()
        self._apply_style()
        self.rtsp_edit.setText(str(self.cfg.get("source", {}).get("path") or ""))

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(12)

        title = QLabel("Package room")
        title.setObjectName("appTitle")
        side.addWidget(title)
        subtitle = QLabel("Live RTSP detection")
        subtitle.setObjectName("muted")
        side.addWidget(subtitle)

        behavior_box = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_box)
        self.behavior = QComboBox()
        for key in TASK_ORDER:
            self.behavior.addItem(TASK_LABELS[key], key)
        self.behavior.currentIndexChanged.connect(self._on_behavior_changed)
        hint = QLabel("Alerts and overlay follow the selected action.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        behavior_layout.addWidget(self.behavior)
        behavior_layout.addWidget(hint)
        side.addWidget(behavior_box)

        source_box = QGroupBox("Camera")
        source_layout = QVBoxLayout(source_box)
        self.rtsp_edit = QLineEdit()
        self.rtsp_edit.setPlaceholderText("rtsp://user:pass@host:554/Streaming/Channels/101/")
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_stream)
        self.rec_btn = QPushButton("REC")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self._toggle_record)
        btns = QHBoxLayout()
        btns.addWidget(self.connect_btn)
        btns.addWidget(self.rec_btn)
        source_layout.addWidget(self.rtsp_edit)
        source_layout.addLayout(btns)
        side.addWidget(source_box)

        roi_box = QGroupBox("Region of interest")
        roi_layout = QVBoxLayout(roi_box)
        self.draw_roi_btn = QPushButton("Draw ROI")
        self.draw_roi_btn.setCheckable(True)
        self.draw_roi_btn.toggled.connect(self._on_draw_roi_toggled)
        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.setObjectName("ghost")
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        roi_btns = QHBoxLayout()
        roi_btns.addWidget(self.draw_roi_btn)
        roi_btns.addWidget(self.clear_roi_btn)
        roi_hint = QLabel("Drag on the video. Person feet and boxes/bags outside the zone are ignored (carrying + YOLO person).")
        roi_hint.setObjectName("muted")
        roi_hint.setWordWrap(True)
        roi_layout.addLayout(roi_btns)
        roi_layout.addWidget(roi_hint)
        side.addWidget(roi_box)

        alert_box = QGroupBox("Alerts")
        alert_layout = QVBoxLayout(alert_box)
        self.desktop_alerts = QCheckBox("Desktop notification")
        self.sound_alerts = QCheckBox("Sound")
        self.desktop_alerts.setChecked(bool(self.cfg.get("alerts", {}).get("desktop", True)))
        self.sound_alerts.setChecked(bool(self.cfg.get("alerts", {}).get("sound", True)))
        self.desktop_alerts.toggled.connect(self._sync_alerter)
        self.sound_alerts.toggled.connect(self._sync_alerter)
        alert_layout.addWidget(self.desktop_alerts)
        alert_layout.addWidget(self.sound_alerts)
        side.addWidget(alert_box)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        self.fps_label = QLabel("0.0 fps")
        self.fps_label.setObjectName("muted")
        self.stats_label = QLabel("visits 0  interacts 0  carries 0  takeaways 0")
        self.stats_label.setObjectName("status")
        self.stats_label.setWordWrap(True)
        side.addWidget(self.status_label)
        side.addWidget(self.fps_label)
        side.addWidget(self.stats_label)
        side.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.alert_banner = QLabel("No alerts yet")
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.video = VideoView()
        self.video.roi_drawn.connect(self._on_roi_drawn)

        results_header = QHBoxLayout()
        results_title = QLabel("Results")
        results_title.setObjectName("sectionTitle")
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("ghost")
        clear_btn.clicked.connect(self._clear_results)
        results_header.addWidget(results_title)
        results_header.addStretch(1)
        results_header.addWidget(clear_btn)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Alert", "Message", "IDs"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(180)

        splitter = QSplitter(Qt.Orientation.Vertical)
        video_wrap = QFrame()
        video_wrap.setObjectName("videoWrap")
        video_l = QVBoxLayout(video_wrap)
        video_l.setContentsMargins(0, 0, 0, 0)
        video_l.addWidget(self.video, 1)
        splitter.addWidget(video_wrap)
        results_wrap = QWidget()
        results_l = QVBoxLayout(results_wrap)
        results_l.setContentsMargins(0, 0, 0, 0)
        results_l.addLayout(results_header)
        results_l.addWidget(self.table)
        splitter.addWidget(results_wrap)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        right_layout.addWidget(self.alert_banner)
        right_layout.addWidget(splitter, 1)

        layout.addWidget(sidebar)
        layout.addWidget(right, 1)

        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._dim_banner)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #12151c; color: #e8edf5; font-size: 13px; }
            #appTitle { font-size: 22px; font-weight: 700; color: #f4f7fb; }
            #sectionTitle { font-size: 15px; font-weight: 600; }
            #muted { color: #8b95a7; font-size: 12px; }
            #status { color: #c5d0e0; }
            QGroupBox {
                border: 1px solid #2a3140; border-radius: 8px; margin-top: 12px; padding: 12px 10px 10px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa6b8; }
            QLineEdit, QComboBox, QTableWidget {
                background: #1b2130; border: 1px solid #323b4e; border-radius: 6px; padding: 6px 8px;
                color: #e8edf5; selection-background-color: #3d4d6b;
            }
            QPushButton {
                background: #2f6fed; color: white; border: none; border-radius: 6px;
                padding: 8px 12px; font-weight: 600;
            }
            QPushButton:hover { background: #3b7cff; }
            QPushButton:disabled { background: #2a3140; color: #6d7788; }
            QPushButton#ghost { background: #243044; }
            QPushButton:checked { background: #d13b3b; }
            #videoView {
                background: #0b0d12; border: 1px solid #2a3140; border-radius: 10px;
                color: #6d7788; font-size: 15px;
            }
            #alertBanner {
                background: #1b2130; border: 1px solid #323b4e; border-radius: 8px;
                padding: 10px 14px; font-weight: 600; min-height: 22px;
            }
            QHeaderView::section {
                background: #1b2130; color: #9aa6b8; border: none; padding: 6px; font-weight: 600;
            }
            QCheckBox { spacing: 8px; }
            """
        )

    def _task_key(self) -> str:
        return str(self.behavior.currentData())

    def _toggle_stream(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._stop_stream()
            return
        source = self.rtsp_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "RTSP", "Enter an RTSP URL first.")
            return
        self.connect_btn.setEnabled(False)
        self.status_label.setText("Starting…")
        self.cfg.setdefault("source", {})["path"] = source
        apply_task_to_cfg(self.cfg, self._task_key())
        self.cfg.setdefault("visualizer", {})["show"] = False
        self.cfg.setdefault("visualizer", {})["write"] = False
        self.cfg.setdefault("alerts", {})["desktop"] = self.desktop_alerts.isChecked()
        self.cfg.setdefault("alerts", {})["sound"] = self.sound_alerts.isChecked()
        self.worker = StreamWorker(self.cfg, source, self._task_key(), pipeline=self.pipeline)
        self.worker.frame_ready.connect(self.video.set_bgr_frame)
        self.worker.event_ready.connect(self._on_event)
        self.worker.status.connect(self.status_label.setText)
        self.worker.fps_ready.connect(lambda fps: self.fps_label.setText(f"{fps:.1f} fps"))
        self.worker.stats_ready.connect(self.stats_label.setText)
        self.worker.failed.connect(self._on_failed)
        self.worker.recording_changed.connect(self.rec_btn.setChecked)
        self.worker.pipeline_ready.connect(self._on_pipeline_ready)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.rec_btn.setEnabled(True)

    def _on_draw_roi_toggled(self, enabled: bool) -> None:
        self.video.set_drawing(enabled)
        if enabled:
            self.status_label.setText("Drag a rectangle on the video to set the ROI")

    def _on_roi_drawn(self, roi: tuple) -> None:
        self.draw_roi_btn.setChecked(False)
        self.video.set_drawing(False)
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_roi(roi)
        elif self.pipeline is not None:
            self.pipeline.set_roi(roi)
        self.cfg.setdefault("roi", {})["xyxy"] = list(roi)
        self.cfg.setdefault("roi", {})["enabled"] = True
        self.status_label.setText("ROI set — person + boxes outside it are ignored")

    def _clear_roi(self) -> None:
        self.draw_roi_btn.setChecked(False)
        self.video.set_drawing(False)
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_roi(None)
        elif self.pipeline is not None:
            self.pipeline.set_roi(None)
        self.cfg.setdefault("roi", {})["xyxy"] = []
        self.cfg.setdefault("roi", {})["enabled"] = False
        self.status_label.setText("ROI cleared")

    def _stop_stream(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.connect_btn.setEnabled(False)
        self.status_label.setText("Stopping…")

    def _on_worker_finished(self) -> None:
        if self.worker is not None and self.worker.pipeline is not None:
            self.pipeline = self.worker.pipeline
        self.worker = None
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.rec_btn.setEnabled(False)
        self.rec_btn.setChecked(False)
        if self.status_label.text() in {"Stopping…", "Starting…"}:
            self.status_label.setText("Disconnected")

    def _on_pipeline_ready(self) -> None:
        if self.worker is not None:
            self.pipeline = self.worker.pipeline
        self._sync_alerter()

    def _on_behavior_changed(self) -> None:
        task = self._task_key()
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_task(task)
        elif self.pipeline is not None:
            self.pipeline.apply_task(task)
        self.alert_banner.setText(f"Watching for: {TASK_LABELS[task]}")

    def _toggle_record(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_record_toggle()

    def _sync_alerter(self) -> None:
        target = None
        if self.worker is not None and self.worker.pipeline is not None:
            target = self.worker.pipeline
        elif self.pipeline is not None:
            target = self.pipeline
        if target is None:
            return
        target.alerter.desktop = self.desktop_alerts.isChecked()
        target.alerter.sound = self.sound_alerts.isChecked()

    def _on_event(self, payload: dict, title: str, message: str) -> None:
        kind = str(payload.get("event", ""))
        stamp = time.strftime("%H:%M:%S")
        banner = BANNER_TEXT.get(kind, title.upper())
        self.alert_banner.setText(f"{banner}  —  {message}")
        color = {
            "put_box_in_bag": "#5c1515",
            "open_box": "#5a3510",
            "dual_entry": "#5c1424",
        }.get(kind, "#1b2130")
        self.alert_banner.setStyleSheet(
            f"#alertBanner {{ background: {color}; border: 1px solid #6a2a2a; "
            f"border-radius: 8px; padding: 10px 14px; font-weight: 600; }}"
        )
        self._banner_timer.start(8000)

        ids = []
        if payload.get("person_id") is not None:
            ids.append(f"person #{payload['person_id']}")
        if payload.get("box_id") is not None:
            ids.append(f"box #{payload['box_id']}")
        if payload.get("bag_id") is not None:
            ids.append(f"bag #{payload['bag_id']}")
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [stamp, title, message, ", ".join(ids)]
        bg = ALERT_ROW_COLORS.get(kind)
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if bg is not None:
                item.setBackground(bg)
            self.table.setItem(row, col, item)
        self.table.scrollToBottom()
        self.table.resizeColumnsToContents()

    def _dim_banner(self) -> None:
        self.alert_banner.setStyleSheet("")

    def _clear_results(self) -> None:
        self.table.setRowCount(0)
        self.alert_banner.setText("No alerts yet")
        self._dim_banner()

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Stream error", message)
        self.status_label.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(4000)
        event.accept()


def launch(config_path: str = "config.yaml") -> int:
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("Package room monitor")
    window = MonitorWindow(config_path)
    window.show()
    return app.exec()
