
# The Entry point into the UI of the analysis
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QMessageBox, QFileDialog)
from qt_ui_classes.main_window_ui import Ui_MainWindow

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)
        self.setWindowTitle("Experiment Analysis")

        # setup button handlers()
        self.setup_button_handlers()
    
    def setup_button_handlers(self):
        pass

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    