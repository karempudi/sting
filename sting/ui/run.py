# File that will launch the Ui that will run the experiment
# alternately you could run the thing with a script as well

from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QFileDialog, QMessageBox)
from qt_ui_classes.run_window_ui import Ui_RunWindow
import sys
import json
import yaml
import pathlib
from pathlib import Path

class RunWindow(QMainWindow):

    def __init__(self):
        super(RunWindow, self).__init__()
        self.ui = Ui_RunWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment run window")
     
        self.setup_button_handlers()

    def setup_button_handlers(self):
        # button handlers 
        self.ui.load_button.clicked.connect(self.load_expt_params)

        self.ui.start_button.clicked.connect(self.start_expt)

        self.ui.stop_button.clicked.connect(self.stop_expt)

    def load_expt_params(self):
        print('Loadinng experimental parameters')

    def start_expt(self):
        print('starting experiment')

    def stop_expt(self):
        print('stopping experiment')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RunWindow()
    window.show()
    sys.exit(app.exec())