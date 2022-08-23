
# The Entry point into the UI of the analysis
import sys
import copy
import pathlib
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QMessageBox, QFileDialog)
from sting.ui.qt_ui_classes.main_window_ui import Ui_MainWindow
from sting.utils.param_io import load_params, save_params
from sting.ui.tweezer import TweezerWindow 
from sting.ui.live import LiveWindow
from datetime import datetime
from sting.liveanalysis.processes import start_live_experiment
from sting.analysis.processes import start_post_analysis

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment Analysis")

        # setup button handlers()
        self.setup_button_handlers()

        # parameter namespace that will be passed around
        # to most functions in the code
        self.param = None
        self.written_param_file = None

        
        # initialize a tweezer window
        # set it's parameters when you have them
        self.tweezer_window = TweezerWindow() 

        
        # initialize a live window
        # set it's parameters when you have them
        self.live_window = LiveWindow()
    
    def setup_button_handlers(self):
        # setup group buttons
        self.ui.setup_button.clicked.connect(self.load_setup_file)
        
        # view setup parameters in a new window
        self.ui.view_setup_button.clicked.connect(self.view_setup_file)

        # write parameters in the experiment analysis dir
        self.ui.write_setup_button.clicked.connect(self.write_setup_file)

        # controls group buttons
        self.ui.start_status_button.clicked.connect(self.start_status_update)

        self.ui.stop_status_button.clicked.connect(self.stop_status_update)

        # View group buttons
        self.ui.tweezable_button.clicked.connect(self.show_tweeze_window)

        self.ui.live_button.clicked.connect(self.show_live_window)

        # Plots group buttons
        self.ui.growth_rates_button.clicked.connect(self.show_growth_rates)

        self.ui.dead_alive_button.clicked.connect(self.show_dead_alive)

        # Need to hook up the analysis progress bars here

        
    def load_setup_file(self):
        # read param file and load them
        filename, _  = QFileDialog.getOpenFileName(self,
                                                "Open an expt .yaml or json file", 
                                                '../data',
                                                "Parameter files (*.yaml *.yml *.json)",
                                                options=QFileDialog.DontUseNativeDialog)
        print(filename)
        if filename == '':
            msg = QMessageBox()
            msg.setText("Experiment setup file not selected")
            msg.exec()

        else:
            # load up the parameters into
            param = load_params(filename)
            self.param = param
            # write to logs here that you loaded parameters
            sys.stdout.write(f"Loaded parameters from {filename}\n")
            sys.stdout.flush()
            self.tweezer_window.set_params(copy.deepcopy(param))
            self.live_window.set_params(copy.deepcopy(param))


    def view_setup_file(self):
        # View setup parameters in a window
        # You can make it more fancier later
        param_dict = self.param.to_dict()
        string_to_show = ""
        for key, values in param_dict.items():
            string_to_show += str(key) + " : " + str(values) + "\n"

        msg = QMessageBox()
        msg.setWindowTitle("Setup and analysis parameters")
        msg.setText(string_to_show)
        msg.exec()

    def write_setup_file(self):
        # write the current setup file to the analysis directory
        data_dir = self.param.Save.directory
        data_dir = data_dir if isinstance(data_dir, pathlib.Path) else Path(data_dir)

        date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
        write_filename = self.param.Experiment.number + '_' + date + '.yaml'
        write_filename = data_dir / write_filename
        if data_dir.exists() and data_dir.is_dir():
            save_params(write_filename, self.param)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            save_params(write_filename, self.param)
            # log there that you create a directory and saved params
        self.written_param_file = write_filename
        sys.stdout.write(f"Parameters of this expt viewing are stored in {write_filename}\n")
        sys.stdout.flush()
    
    def start_status_update(self):
        pass
    
    def stop_status_update(self):
        pass

    def show_tweeze_window(self):
        self.tweezer_window.show()
    
    def show_live_window(self):
        self.live_window.show()

    def show_growth_rates(self):
        pass
    
    def show_dead_alive(self):
        pass
    

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())    

if __name__ == "__main__":
    main()