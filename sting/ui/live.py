
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
        self.setWindowTitle("Live analysis loop window ...")

        # set default check state
        # by default, the image processing will do
        # cells, channels and bboxes around barcodes
        # if you change the check state, only the viewing 
        self.ui.cell_seg_check_box.setChecked(True)
        self.ui.channel_seg_check_box.setChecked(True)
        self.ui.reg_detect_check_box.setChecked(True)
        self.setup_button_handlers()

        self.ui.live_image_graphics.ui.histogram.hide()
        self.ui.live_image_graphics.ui.roiBtn.hide()
        self.ui.live_image_graphics.ui.menuBtn.hide()
        if param != None and 'Live' in param.keys():
            self.param = param.Live
        else:
            self.param = None

    # only called from outside
    def set_params(self, param: RecursiveNamespace):
        if param!= None and 'Live' in param.keys():
            self.param = param.Live
        else:
            self.param = None
        print(self.param)
        
    def setup_button_handlers(self):
        # start live acquisition
        self.ui.start_imaging_button.clicked.connect(self.start_acquire)
        # stop live acquistion
        self.ui.stop_imaging_button.clicked.connect(self.stop_acquire)
        # change set
    
    def start_acquire(self):
        sys.stdout.write(f"Started acquiring ... \n")
        sys.stdout.flush()


    def stop_acquire(self):
        sys.stdout.write(f"Stopped acquiring ... \n")
        sys.stdout.flush()

    
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