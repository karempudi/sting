
import sys
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QFileDialog, QMessageBox)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from sting.utils.types import RecursiveNamespace
from sting.ui.qt_ui_classes.tweezer_window_ui import Ui_TweezerWindow
from sting.utils.param_io import load_params

                
class TweezerWindow(QMainWindow):
    """
    A Ui class that sets up the tweezer window and
    hooks up all the buttons to approriate controls

    You will intialize an instance of this in the Mainwidow
    of the experiment viewer
    """
    def __init__(self, param: RecursiveNamespace = None):
        super(TweezerWindow, self).__init__()
        self.ui = Ui_TweezerWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Tweezer window, to view stuff to tweeze")

        self.setup_button_handlers()

        self.param = param

        self.current_pos = None
        self.current_ch_no = None
        self.position_no_validator =  None
        self.channel_no_validator = None

        self.expt_running = False

    
    def setup_button_handlers(self):
        # plotting settings hide defaults
        self.ui.image_plot.ui.histogram.hide()
        self.ui.image_plot.ui.roiBtn.hide()
        self.ui.image_plot.ui.menuBtn.hide()
        self.ui.barcode_plot_1.ui.histogram.hide()
        self.ui.barcode_plot_1.ui.roiBtn.hide()
        self.ui.barcode_plot_1.ui.menuBtn.hide()
        self.ui.barcode_plot_2.ui.histogram.hide()
        self.ui.barcode_plot_2.ui.roiBtn.hide()
        self.ui.barcode_plot_2.ui.menuBtn.hide()

    def set_params(self, param: RecursiveNamespace):
        self.param = param
        self.position_no_validator = QIntValidator(0, self.param.Experiment.max_positions, self.ui.pos_no_edit)
        self.channel_no_validator = QIntValidator(0, 120, self.ui, self.ui.ch_no_edit)
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file',
                        help='params file in .json, .yaml or .yml format',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("Using tweezer window from command line ....")
    params = load_params(args.param_file)


    app = QApplication(sys.argv)
    window = TweezerWindow(param=params)
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()