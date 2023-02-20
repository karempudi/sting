
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import (QTimer, pyqtSignal, QThread)

from sting.utils.types import RecursiveNamespace
from sting.ui.qt_ui_classes.live_window_ui import Ui_LiveWindow
from sting.utils.param_io import load_params
from sting.microscope.utils import fetch_mm_image
from tifffile import imread
from datetime import datetime
from sting.mm.detect import process_image, get_loaded_model
from sting.mm.utils import plot_inference_img_pyqtgraph
import cv2
from pycromanager import Core

class LiveImageFetch(QThread):

    dataFetched = pyqtSignal()

    def __init__(self, param):
        super(LiveImageFetch, self).__init__()
        self.data = None
        self.param = param

    def set_data(self, image, metadata):
        self.data = image

    def run(self):
        # try different things depending on the param
        try:
            if self.param.fake_image:
                self.data = imread(Path(self.param.fake_image_path))
            else:
                self.data = fetch_mm_image()

        except Exception as e:
            sys.stdout.write(f"Live image grabbing failed inside thread for reason: {e}\n")
            sys.stdout.flush()
            self.data = None
        finally:
            self.dataFetched.emit()

    def get_data(self):
        return self.data



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

        self.ui.scroll_bar.setMinimum(0)
        self.ui.scroll_bar.setMaximum(3)
        self.ui.scroll_bar.setSingleStep(1)
        self.ui.scroll_bar.setPageStep(1)
        self.ui.scroll_bar.setTracking(True)
        self.ui.scroll_bar.setEnabled(False)

        if param != None and 'Live' in param.keys():
            self.param_live = param.Live
            self.param = param
        else:
            self.param_live = None
            self.param = None
        
        if param != None and 'Analysis' in param.keys():
            self.analysis_params = param.Analysis
        else:
            self.analysis_params = None

        self.img_acq_thread = None
        self.acquiring = False
        self.interval = 1000
        self.timer = QTimer()
        self.timer.setInterval(self.interval) # Image will be grabbed every 300ms
        self.net = None

    def closeEvent(self, event):
        self.stop_acquire()
        sys.stdout.write(f"Live window closed ..\n")
        sys.stdout.flush()

    # only called from outside
    def set_params(self, param: RecursiveNamespace):
        if param != None and 'Live' in param.keys():
            self.param_live = param.Live
            self.param = param
        else:
            self.param_live = None
            self.param = None
        if param != None and 'Analysis'  in param.keys():
            self.analysis_params = param.Analysis
        else:
            self.analysis_params = None
        print(self.param.Live)
        print("Printing analysis parameters ....")
        print(self.analysis_params)
        
    def setup_button_handlers(self):
        # start live acquisition
        self.ui.start_imaging_button.clicked.connect(self.start_acquire)
        # stop live acquistion
        self.ui.stop_imaging_button.clicked.connect(self.stop_acquire)
        # slider position change
        self.ui.scroll_bar.valueChanged.connect(self.update_image)
    
    def start_acquire(self):
        if not self.acquiring:
            sys.stdout.write(f"Started acquiring ... \n")
            sys.stdout.flush()
            # Load networks
            self.net = get_loaded_model(self.param)
            
            sys.stdout.write(f"Nets loaded on device\n")
            sys.stdout.flush()
            self.ui.scroll_bar.setEnabled(True)
            self.timer.timeout.connect(self.grab_image)
            self.timer.start()

    def grab_image(self):
        try:
            # send only live parameters to live image grabbing thread
            self.img_acq_thread = LiveImageFetch(self.param_live)
            self.img_acq_thread.dataFetched.connect(self.update_image)
            self.img_acq_thread.start()
        except Exception as e:
            sys.stdout.write(f"Failed to launch image grabbing thread :( due to {e}\n")
            sys.stdout.flush()
        self.acquiring = True

    def stop_acquire(self):
        self.acquiring = False
        self.ui.scroll_bar.setEnabled(False)
        self.timer.stop() # stop the update timer
        if self.img_acq_thread != None:
            self.img_acq_thread.quit()
        # delete network objects and set them to none
        del self.net
        self.net = None
        # empty cuda cache
        torch.cuda.empty_cache()
        self.timer = None
        self.timer = QTimer()
        self.timer.setInterval(self.interval)
        sys.stdout.write(f"Stopped acquiring ... \n")
        sys.stdout.flush()

    def update_image(self):
        if self.acquiring and self.img_acq_thread.data is not None:
            # update the image with the data
            image = self.img_acq_thread.data.copy()
            datapoint = {
                'position': 0,
                'time': 0,
                'image': image
            }
            result = process_image(datapoint, self.net, self.param, visualize=True)
            #sys.stdout.write(f"Segmented: error: {result['error']} found {result['total_channels']}\n")
            #sys.stdout.flush()

            barcode_img = np.transpose(plot_inference_img_pyqtgraph(result), (1, 0, 2))
            if (self.ui.scroll_bar.value() == 0 and (not result['error'])):
                #sys.stdout.write(f"slider value is {self.ui.scroll_bar.value()} \n")
                #sys.stdout.flush()
                self.ui.live_image_graphics.setImage(image.T, autoLevels=True, autoRange=False)
            elif (self.ui.scroll_bar.value() == 1 and (not result['error'])):
                self.ui.live_image_graphics.setImage(result['cells'].T, autoLevels=True, autoRange=False)
            elif (self.ui.scroll_bar.value() == 2 and (not result['error'])):
                self.ui.live_image_graphics.setImage(result['channels'].T, autoLevels=True, autoRange=False)
            elif (self.ui.scroll_bar.value() == 3 and (not result['error'])):
                self.ui.live_image_graphics.setImage(barcode_img, autoLevels=True, autoRange=False)
                #sys.stdout.write(f"Barcodes are: {bboxes_numpy[0].shape} \n")
                #sys.stdout.flush()
            else:
                sys.stdout.write(f"slider value is {self.ui.scroll_bar.value()} \n")
                sys.stdout.flush()
                img = np.random.randint(low = 0, high = 2, size=(3000, 4096))
                self.ui.live_image_graphics.setImage(img, autoLevels=True, autoRange=False)
        else:
            img = np.random.randint(low = 0, high = 2, size=(3000, 4096))
            self.ui.live_image_graphics.setImage(img, autoLevels=True, autoRange=False)


    
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