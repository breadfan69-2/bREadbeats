"""
bREadbeats - Qt Stylesheet definitions.
Extracted from main.py for modularization.
"""


def get_main_stylesheet() -> str:
    """Restim-Coyote3 darkmode theme with #3d3d3d background"""
    return """
        /* Main Window and Widgets */
        QMainWindow, QWidget {
            background-color: #3d3d3d;
            color: #e0e0e0;
        }

        QFrame {
            background-color: #3d3d3d;
            color: #e0e0e0;
        }

        /* Menu Bar */
        QMenuBar {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border-bottom: 1px solid #5d5d5d;
        }

        QMenuBar::item:selected {
            background-color: #5d5d5d;
        }

        /* Menus */
        QMenu {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
        }

        QMenu::item:selected {
            background-color: #008b8b;
            color: #ffffff;
        }

        /* Buttons */
        QPushButton {
            background-color: #565d7f;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 5px 15px;
        }

        QPushButton:hover {
            background-color: #6d6d8f;
        }

        QPushButton:checked {
            background-color: #008b8b;
            color: #ffffff;
        }

        QPushButton:checked:hover {
            background-color: #109b9b;
        }

        QPushButton:checked:pressed {
            background-color: #006f6f;
        }

        QPushButton:pressed {
            background-color: #4a4d6f;
        }

        QPushButton:disabled {
            background-color: #424242;
            color: #757575;
        }

        /* Labels */
        QLabel {
            color: #e0e0e0;
        }

        /* Line Edit */
        QLineEdit {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
            padding: 5px;
        }

        QLineEdit:focus {
            border: 1px solid #565d7f;
        }

        /* Spin Box */
        QSpinBox, QDoubleSpinBox {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
            padding: 5px;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #3d3d3d;
            border: 1px solid #2d2d2d;
            width: 20px;
        }

        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4d4d4d;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #565d7f;
        }

        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #5d5d5d;
            height: 8px;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background-color: #565d7f;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }

        QSlider::handle:horizontal:hover {
            background-color: #6d6d8f;
        }

        /* ComboBox */
        QComboBox {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
            padding: 5px;
        }

        QComboBox:focus {
            border: 1px solid #565d7f;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        /* CheckBox and RadioButton */
        QCheckBox, QRadioButton {
            color: #e0e0e0;
        }

        QCheckBox::indicator, QRadioButton::indicator {
            width: 18px;
            height: 18px;
        }

        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
            background-color: #4d4d4d;
            border: 1px solid #5d5d5d;
            border-radius: 3px;
        }

        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background-color: #008b8b;
            border: 1px solid #008b8b;
            border-radius: 3px;
        }

        /* GroupBox */
        QGroupBox {
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }

        QGroupBox::indicator {
            width: 0px;
            height: 0px;
        }

        /* Tabs */
        QTabBar::tab {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            padding: 8px 20px;
        }

        QTabBar::tab:selected {
            background-color: #008b8b;
            color: #ffffff;
        }

        QTabWidget::pane {
            border: 1px solid #5d5d5d;
        }

        /* ScrollBar */
        QScrollBar:vertical {
            background-color: #3d3d3d;
            width: 12px;
            border: none;
        }

        QScrollBar::handle:vertical {
            background-color: #626262;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #727272;
        }

        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
            height: 0;
            background: none;
        }

        QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {
            background: none;
        }

        QScrollBar:horizontal {
            background-color: #3d3d3d;
            height: 12px;
            border: none;
        }

        QScrollBar::handle:horizontal {
            background-color: #626262;
            border-radius: 6px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #727272;
        }

        QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
            width: 0;
            background: none;
        }

        QScrollBar::sub-page:horizontal, QScrollBar::add-page:horizontal {
            background: none;
        }

        /* ProgressBar */
        QProgressBar {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #565d7f;
            border-radius: 3px;
        }

        /* Text Edit */
        QTextEdit, QPlainTextEdit {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            border-radius: 4px;
        }

        /* List View and Table View */
        QListView, QTableView, QTreeView {
            background-color: #4d4d4d;
            color: #e0e0e0;
            border: 1px solid #5d5d5d;
            gridline-color: #5d5d5d;
        }

        QListView::item:selected, QTableView::item:selected, QTreeView::item:selected {
            background-color: #008b8b;
        }

        /* Dialogs */
        QDialog {
            background-color: #3d3d3d;
            color: #e0e0e0;
        }
    """
    


def get_thin_scrollbar_style() -> str:
    """Return thin minimal scrollbar CSS for NoWheelScrollArea tabs"""
    return """
        QScrollBar:vertical {
            background-color: transparent;
            width: 4px;
            border: none;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background-color: rgba(100, 100, 100, 0.5);
            border-radius: 2px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: rgba(150, 150, 150, 0.7);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
            background: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
    """


