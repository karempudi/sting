# File that will launch the Ui that will run the experiment
# alternately you could run the thing with a script as well

from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QFileDialog, QMessageBox)
from sting.ui.qt_ui_classes.run_window_ui import Ui_RunWindow
import time
import sys
import json
import yaml
import pathlib
from pathlib import Path
from sting.utils.param_io import load_params
from sting.liveanalysis.processes import ExptRun, start_live_experiment

class RunWindow(QMainWindow):

    def __init__(self):
        super(RunWindow, self).__init__()
        self.ui = Ui_RunWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment run window")
     
        self.setup_button_handlers()

        self.params = None
        self.expt_obj = None
        self.simulation = False

    def setup_button_handlers(self):
        # button handlers 
        self.ui.load_button.clicked.connect(self.load_expt_params)

        self.ui.start_button.clicked.connect(self.start_expt)

        self.ui.stop_button.clicked.connect(self.stop_expt)

        self.ui.simulation_check.stateChanged.connect(self.set_simulation_mode)

    def load_expt_params(self):
        sys.stdout.write("Loadinng experimental parameters. Please select file.\n")
        sys.stdout.flush()
        try:
            filename, _ = QFileDialog.getOpenFileName(self,
                                                   self.tr("Open an expt params setup file"),
                                                   '.',
                                                   self.tr("Expt json or yaml file (*.json *.yaml *.yml"))
            if filename == '':
                msg = QMessageBox()
                msg.setText("Expt setup params not selected")
                msg.exec()
            else:
                # load params
                self.params = load_params(filename)
                # call verify params and raise something if it 
                # is not correct
        
        except Exception as e:
            sys.stdout.write(f"Error in loading the experimental setup file -- {e}\n")
            sys.stdout.flush()
        
        finally:
            if self.params != None:
                sys.stdout.write("Parameters for the experiment set from file \n")
                sys.stdout.flush()

    def start_expt(self):
        sys.stdout.write("Setting up experiment object from the parameters.\n")
        sys.stdout.flush()

        if self.params != None:
            self.expt_obj = ExptRun(self.params)
            self.ui.start_button.setEnabled(False)
            self.ui.load_button.setEnabled(False)
            start_live_experiment(self.expt_obj, self.params, sim=False)
        else:
            msg = QMessageBox()
            msg.setText("Expt parameters not loaded")
            msg.exec()


    def stop_expt(self):
        sys.stdout.write("Stopping the experiment.\n")
        sys.stdout.flush()

        # Stop the experiment and finally set it to None
        self.expt_obj.stop()
        time.sleep(3)
        self.ui.load_button.setEnabled(True)
        self.ui.start_button.setEnabled(True)
        self.expt_obj = None

    def set_simulation_mode(self, state):
        self.simulation = state
        if self.simulation:
            sys.stdout.write("Expt running in simulation mode\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("Expt running in real mode\n")
            sys.stdout.flush()

        
def main():
    app = QApplication(sys.argv)
    window = RunWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()