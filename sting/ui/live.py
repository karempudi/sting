
import sys
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QFileDialog, QMessageBox)

from sting.utils.types import RecursiveNamespace
from sting.ui.qt_ui_classes.live_window_ui import Ui_LiveWindow
from sting.utils.param_io import load_params



class LiveWindow(QMainWindow):
    """
    An Ui class that sets up the tweezer window and 
    hooks up all the buttons to appropriate controls.
    
    All the code that runs when you press buttons on
    the live window should be in used from this file

    """
    def __init__(self, param: RecursiveNamespace = None):
        super(LiveWindow, self).__init__()
        self.ui = Ui_LiveWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Live analysis loop windwo ...")

        self.setup_button_handlers()

        self.param = param


        
    def setup_button_handlers(self):
        pass
    
    def set_params(self, param: RecursiveNamespace):
        self.param = param


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file',
                        help='params file in .json, .yaml or .yml format',
                        required=True)
    # Add more arguments here later, on how the live functions
    # should run
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("Using tweezer window from command line ....")
    params = load_params(args.param_file)

    app = QApplication(sys.argv)
    window = LiveWindow(param=params)
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()