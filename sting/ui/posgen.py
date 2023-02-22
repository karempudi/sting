
import sys
import copy
import pathlib
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, QFile, QThread
from sting.ui.qt_ui_classes.posgen_window_ui import Ui_PosGenWindow

from datetime import datetime
from pycromanager import Core

class PosGenWindow(QMainWindow):

    def __init__(self):
        super(PosGenWindow, self).__init__()
        self.ui = Ui_PosGenWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Grid position generation")

        self.one_side = True
        self.two_sides = False
        self.corners_dict = {}
        if not self.one_side:
            self.enable_one_side_buttons(False)
        if not self.two_sides:
            self.enable_two_sides_buttons(False)
        # setup button handlers
        self.setup_button_handlers()
       
    def setup_button_handlers(self):
        # set up button handlers for all the buttons

        # hide some of the pyqtgraph plotting settings
        self.ui.positions_plot.ui.histogram.hide()
        self.ui.positions_plot.ui.roiBtn.hide()
        self.ui.positions_plot.ui.menuBtn.hide()

        self.ui.one_rect_button.toggled.connect(self.set_layout_type)
        self.ui.two_rect_button.toggled.connect(self.set_layout_type)

        self.ui.tl_button.clicked.connect(self.set_tl_position)
        self.ui.tr_button.clicked.connect(self.set_tr_position)
        self.ui.br_button.clicked.connect(self.set_br_position)
        self.ui.bl_button.clicked.connect(self.set_bl_position)

        self.ui.left_tl_button.clicked.connect(self.set_left_tl_position)
        self.ui.left_tr_button.clicked.connect(self.set_left_tr_position)
        self.ui.left_br_button.clicked.connect(self.set_left_br_position)
        self.ui.left_bl_button.clicked.connect(self.set_left_bl_position)
        self.ui.right_tl_button.clicked.connect(self.set_right_tl_position)
        self.ui.right_tr_button.clicked.connect(self.set_right_tr_position)
        self.ui.right_br_button.clicked.connect(self.set_right_br_position)
        self.ui.right_bl_button.clicked.connect(self.set_right_bl_position)

    def set_layout_type(self, clicked):
        self.one_side = self.ui.one_rect_button.isChecked()
        self.two_sides = self.ui.two_rect_button.isChecked()
        if not self.one_side:
            self.enable_one_side_buttons(False)
            self.enable_two_sides_buttons(True)

        if not self.two_sides:
            self.enable_two_sides_buttons(False)
            self.enable_one_side_buttons(True)
        self.corners_dict = {}

    def enable_one_side_buttons(self, value):
        self.ui.tl_button.setEnabled(value)
        self.ui.tr_button.setEnabled(value)
        self.ui.bl_button.setEnabled(value)
        self.ui.br_button.setEnabled(value)

    def enable_two_sides_buttons(self, value):
        self.ui.left_tl_button.setEnabled(value)
        self.ui.left_tr_button.setEnabled(value)
        self.ui.left_bl_button.setEnabled(value)
        self.ui.left_br_button.setEnabled(value)
        self.ui.right_tl_button.setEnabled(value)
        self.ui.right_tr_button.setEnabled(value)
        self.ui.right_bl_button.setEnabled(value)
        self.ui.right_br_button.setEnabled(value)

    def get_mm_current_position(self):
        # Check that micromanger is enabled, to 
        # grab positions
        core = None
        position_dict = None
        try:
            core = Core()
            if core.get_focus_device() != 'PFSOffset':
                raise ValueError("Foucs device is not PFSOffset")
            if core.get_xy_device() != 'XYStage':
                raise ValueError("XY device is not XYStage")
            x = core.get_x_position()
            y = core.get_y_position()
            z = core.get_foucs_position() 
            position_dict = {
                'X': x,
                'Y': y,
                'Z': z,
                'grid_row': 0,
                'grid_col': 0,
            }
        except Exception as e:
            msg = QMessageBox()
            msg.setText(f"Micromanger 2.0 position not grabbed due to: {e}")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
        finally:
            if core:
                del core
            return position_dict 

    def set_position_and_label(self, label):
        position_dict = self.get_mm_current_position()
        if position_dict != None:
            position_dict['label'] = 'Pos' + label
            self.corners_dict[label] = position_dict
        else:
            msg = QMessageBox()
            msg.setText(f"Position: {label} corner not set for some reason")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
        
    def set_tl_position(self):
        self.set_position_and_label('TL')

    def set_tr_position(self):
        self.set_position_and_label('TR')

    def set_br_position(self):
        self.set_position_and_label('BR')

    def set_bl_position(self):
        self.set_position_and_label('BL')

    def set_left_tl_position(self):
        self.set_position_and_label('TL1')
  
    def set_left_tr_position(self):
        self.set_position_and_label('TR1')
 
    def set_left_br_position(self):
        self.set_position_and_label('BR1')

    def set_left_bl_position(self):
        self.set_position_and_label('BL1')
       
    def set_right_tl_position(self):
        self.set_position_and_label('TL2')

    def set_right_tr_position(self):
        self.set_position_and_label('TR2')

    def set_right_br_position(self):
        self.set_position_and_label('BR2')

    def set_right_bl_position(self):
        self.set_position_and_label('BL2')


def main():
    app = QApplication(sys.argv)
    window = PosGenWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()