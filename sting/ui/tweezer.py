
import sys
import numpy as np
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QFileDialog, QMessageBox)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from sting.utils.types import RecursiveNamespace
from sting.ui.qt_ui_classes.tweezer_window_ui import Ui_TweezerWindow
from sting.utils.param_io import load_params
from sting.utils.disk_ops import read_files

class DataFetchThread(QThread):

    data_fetched = pyqtSignal()

    def __init__(self, read_type, param, position, channel_no, max_imgs):
        super(DataFetchThread, self).__init__()
        self.read_type = read_type
        self.param = param
        self.position = position
        self.channel_no = channel_no
        self.max_imgs = max_imgs
        self.data = None

    def run(self):
        sys.stdout.write(f"Image thread to get Pos: {self.position} Ch no: {self.channel_no}\n")
        sys.stdout.flush()
        try:
            a = 1 + 1
        except Exception as e:
            sys.stdout.write(f"Data couldn't be fetched due to {e}\n")
            sys.stdout.flush()
            self.data = {
                'image': np.random.normal(loc=0.0, scale=1.0, size=(100, 100))
            }
        finally:
            self.data_fetched.emit()

    def get_data(self):
        return self.data
                
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

        # which image to show
        self.show_phase = False
        self.show_seg = True
        self.show_tracks = False

        # max images to show
        self.max_imgs = 20

        # data fetch thread
        self.data_fetch_thread = None
        self.data_thread_running = False
    
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


        self.ui.pos_no_edit.textChanged.connect(self.position_changed)
        self.ui.ch_no_edit.textChanged.connect(self.channel_changed)
        # fetch data for the current settings
        self.ui.fetch_button.clicked.connect(self.fetch_data)

        # set what kind of image you want to read
        self.ui.phase_image.toggled.connect(self.set_image_type)
        self.ui.cell_seg_image.toggled.connect(self.set_image_type)
        self.ui.cell_tracks_image.toggled.connect(self.set_image_type)

        # how many images to get toggle
        self.ui.get_last20_radio.toggled.connect(self.set_no_images2get)
        self.ui.get_all_images_radio.toggled.connect(self.set_no_images2get)

    def set_params(self, param: RecursiveNamespace):
        self.param = param
        self.position_no_validator = QIntValidator(0, self.param.Experiment.max_positions, self.ui.pos_no_edit)
        self.channel_no_validator = QIntValidator(0, 120, self.ui, self.ui.ch_no_edit)
    
    def position_changed(self):
        position = self.ui.pos_no_edit.text()
        try:
            int_pos = int(position)
        except:
            self.ui.pos_no_edit.setText("")
            int_pos = None
        finally:
            self.current_pos = int_pos

        sys.stdout.write(f"Position set to {self.current_pos}\n")
        sys.stdout.flush()
    
    def channel_changed(self):
        ch_no = self.ui.ch_no_edit.text()
        try:
            int_ch_no = int(ch_no)
        except:
            self.ui.ch_no_edit.setText("")
            int_ch_no = None
        finally:
            self.current_ch_no = int_ch_no
        sys.stdout.write(f"Channel no set to {self.current_ch_no}\n")
        sys.stdout.flush()

    def set_image_type(self, clicked):
        self.show_phase = self.ui.phase_image.isChecked()
        self.show_seg = self.ui.cell_seg_image.isChecked()
        self.show_tracks = self.ui.cell_tracks_image.isChecked()
        sys.stdout.write(f"Phase: {self.show_phase}, Seg: {self.show_seg}, Tracks: {self.show_tracks}\n")
        sys.stdout.flush()
    
    def set_no_images2get(self):
        if self.ui.get_last20_radio.isChecked():
            self.max_imgs = 20
            sys.stdout.write(f"Getting only {self.max_imgs} images\n")
        else:
            self.max_imgs = None
            sys.stdout.write(f"Getting all images\n")
        sys.stdout.flush()
    
    def fetch_data(self):
        if self.show_phase:
            read_type = 'phase'
        elif self.show_seg:
            read_type = 'cell_seg'
        elif self.show_tracks:
            read_type = 'cell_tracks'
        
        if self.data_fetch_thread is None:
            self.data_fetch_thread = DataFetchThread(read_type, self.param, 
                                        self.current_pos, self.current_ch_no, self.max_imgs)

            self.data_fetch_thread.start()
            self.data_fetch_thread.data_fetched.connect(self.update_image)
    
    def update_image(self):
        sys.stdout.write(f"Image updated ...\n")
        sys.stdout.flush()
        self.data_fetch_thread.quit()
        self.data_fetch_thread.wait()
        self.data_fetch_thread = None
        return None

    
 
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