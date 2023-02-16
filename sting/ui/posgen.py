
import sys
import copy
import pathlib
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, QFile, QThread
from sting.ui.qt_ui_classes.posgen_window_ui import Ui_PosGenWindow

from datetime import datetime

class PosGenWindow(QMainWindow):

    def __init__(self):
        super(PosGenWindow, self).__init__()
        self.ui = Ui_PosGenWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Grid position generation")

        # setup button handlers
        self.setup_button_handlers()
        
    def setup_button_handlers(self):
        pass


def main():
    app = QApplication(sys.argv)
    window = PosGenWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()