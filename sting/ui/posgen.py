
import sys
import copy
import pathlib
import json
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, 
                QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, QFile, QThread
from sting.ui.qt_ui_classes.posgen_window_ui import Ui_PosGenWindow

from datetime import datetime
from pycromanager import Core, Acquisition
from sting.microscope.motion import RectGridMotion, TwoRectGridMotion
from sting.microscope.utils import construct_pos_file
import multiprocessing as tmp

class PosGenWindow(QMainWindow):

    def __init__(self):
        super(PosGenWindow, self).__init__()
        self.ui = Ui_PosGenWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Grid position generation")

        self.one_side = True
        self.two_sides = False
        self.corners_dict = {}
        self.motion_object = None
        self.positions_to_write = None
        self.current_n_rows = None
        self.current_n_cols = None

        self.save_dir = None
        self.exposure = None
        self.mm_version=2.0

        if not self.one_side:
            self.enable_one_side_buttons(False)
            self.motion_object = TwoRectGridMotion()
        if not self.two_sides:
            self.enable_two_sides_buttons(False)
            self.motion_object = RectGridMotion()
        # setup button handlers
        self.setup_button_handlers()
       
    def setup_button_handlers(self):
        # set up button handlers for all the buttons

        # hide some of the pyqtgraph plotting settings
        #self.ui.positions_plot.ui.histogram.hide()
        #self.ui.positions_plot.ui.roiBtn.hide()
        #self.ui.positions_plot.ui.menuBtn.hide()

        self.ui.one_rect_button.toggled.connect(self.set_layout_type)
        self.ui.two_rect_button.toggled.connect(self.set_layout_type)

        self.ui.mm14_button.toggled.connect(self.set_mm_version)
        self.ui.mm20_button.toggled.connect(self.set_mm_version)

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

        self.ui.generate_pos_button.clicked.connect(self.generate_positions)

        self.ui.clear_button.clicked.connect(self.clear_data)
        self.ui.from_file_button.clicked.connect(self.load_positions_from_file)


        self.ui.save_pos_button.clicked.connect(self.save_positions_to_file)

        self.ui.n_rows_edit.textChanged.connect(self.n_rows_changed)
        self.ui.n_cols_edit.textChanged.connect(self.n_cols_changed)

        self.ui.update_plot_button.clicked.connect(self.update_plot)

        self.ui.save_dir_button.clicked.connect(self.set_save_dir)
        self.ui.exposure_button.clicked.connect(self.set_exposure)
        self.ui.exposure_edit.textChanged.connect(self.set_exposure)

        self.ui.dry_acquire_button.clicked.connect(self.acquire_dry_run)

        self.ui.quit_button.clicked.connect(self.quit)

    def set_layout_type(self, clicked):
        self.one_side = self.ui.one_rect_button.isChecked()
        self.two_sides = self.ui.two_rect_button.isChecked()
        if not self.one_side:
            self.enable_one_side_buttons(False)
            self.enable_two_sides_buttons(True)
            self.motion_object = TwoRectGridMotion()

        if not self.two_sides:
            self.enable_two_sides_buttons(False)
            self.enable_one_side_buttons(True)
            self.motion_object = RectGridMotion()
        self.corners_dict = {}

    def set_mm_version(self, clicked):
        self.mm14_true = self.ui.mm14_button.isChecked()
        self.mm20_true = self.ui.mm14_button.isChecked()

        if self.mm14_true:
            self.mm_version = 1.4
        else:
            self.mm_version = 2.0


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
            if core.get_xy_stage_device() != 'XYStage':
                raise ValueError("XY device is not XYStage")
            x = core.get_x_position()
            y = core.get_y_position()
            z = core.get_auto_focus_offset()
            position_dict = {
                'x': x,
                'y': y,
                'z': z,
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
            print(position_dict)
            return position_dict 

    def set_position_and_label(self, label):
        position_dict = self.get_mm_current_position()
        if position_dict != None:
            position_dict['label'] = 'Pos' + label
            self.corners_dict[label] = position_dict
            self.motion_object.set_corner_position(label, position_dict)
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

    def generate_positions(self):
        # code to generate positions based on the positions needed
        try:
            if self.motion_object.nrows != None and self.motion_object.ncols != None:
                self.motion_object.construct_grid() 
                self.positions_to_write = self.motion_object.positions
        except Exception as e:
            msg = QMessageBox()
            msg.setText(f"Positions not generated due to: {e}")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
        
    def save_positions_to_file(self):

        filename, _ = QFileDialog.getSaveFileName(self, "Save .pos positions file",
                            "../data", "Position files (*.pos)",
                            options=QFileDialog.DontUseNativeDialog)
        sys.stdout.write(f"Filename: {filename} selected\n")
        sys.stdout.flush()
        write_json = None
        try:
            core = Core()
            if core.get_focus_device() != 'PFSOffset':
                raise ValueError("Foucs device is not PFSOffset")
            if core.get_xy_stage_device() != 'XYStage':
                raise ValueError("XY device is not XYStage")
            
            write_json = construct_pos_file(self.positions_to_write, {
                'xy_device': core.get_xy_stage_device(),
                'z_device': core.get_focus_device(),
            }, version=self.mm_version)
        except Exception as e:
            msg = QMessageBox()
            msg.setText(f"Micromanger 2.0 position not grabbed due to: {e}")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
        finally:
            if core:
                del core

        if filename == '' or self.positions_to_write == None or write_json == None:
            msg = QMessageBox()
            msg.setText("Position file to save not selected or positions not generated correctly")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
        else:
            filename = Path(filename)
            with open(filename, 'w') as fh:
                fh.write(json.dumps(write_json))

    def clear_data(self):
        self.one_side = True
        self.two_sides = False
        self.corners_dict = {}
        self.motion_object = None
        self.positions_to_write = None
        self.current_n_rows = None
        self.current_n_cols = None
        self.save_dir = None
        self.exposure = None
        if not self.one_side:
            self.enable_one_side_buttons(False)
            self.motion_object = TwoRectGridMotion()
        if not self.two_sides:
            self.enable_two_sides_buttons(False)
            self.motion_object = RectGridMotion()
    
    def n_rows_changed(self):
        n_rows = self.ui.n_rows_edit.text()
        try:
            int_n_rows = int(n_rows)
        except:
            self.ui.n_rows_edit.setText("")
            int_n_rows = None
        finally:
            self.current_n_rows = int_n_rows
            self.motion_object.set_rows(self.current_n_rows)

        sys.stdout.write(f"Number of rows set to : {self.current_n_rows}\n")
        sys.stdout.flush()

    def n_cols_changed(self):
        n_cols = self.ui.n_cols_edit.text()
        try:
            int_n_cols = int(n_cols)
        except:
            self.ui.n_colss_edit.setText("")
            int_n_cols = None
        finally:
            self.current_n_cols = int_n_cols
            self.motion_object.set_cols(self.current_n_cols)

        sys.stdout.write(f"Number of columns set to : {self.current_n_cols}\n")
        sys.stdout.flush()
    
    def update_plot(self):
        plotItem = self.ui.positions_plot.getPlotItem()
        plotItem.invertY(b=True)
        plotItem.invertX(b=True)
        positions = self.positions_to_write
        x = [position['x'] for position in positions]
        y = [position['y'] for position in positions]
        plotItem.plot(x, y, symbol='o', pen=(0, 128,0), symbolBrush=(0, 0, 200))

        corners_x = []
        corners_y = []
        for corner in self.corners_dict:
            corners_x.append(self.corners_dict[corner]['x'])
            corners_y.append(self.corners_dict[corner]['y'])
        
        plotItem.plot(corners_x, corners_y, symbol='o', pen=None, symbolBrush=(200, 0, 0))

        plotItem.setLabel('bottom', 'X')
        plotItem.setLabel('left', 'Y')
    
    def quit(self):
        self.close()
    
    def load_positions_from_file(self):

        filename, _ = QFileDialog.getOpenFileName(self, "Load .pos positiions file",
                            "../data", "Position files (*.pos)", 
                            options=QFileDialog.DontUseNativeDialog)
        sys.stdout.write(f"Using: {filename} for loading corner positions\n")
        sys.stdout.flush()
        corner_positions = None
        try:
            if self.one_side:
                microscope_props, corner_positions = RectGridMotion().parse_position_file(Path(filename))
        except Exception as e:
            msg = QMessageBox()
            msg.setText(f"Corner positions couldn't be loaded due to {e}")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
        finally:
            if corner_positions != None:
                for corner in corner_positions:
                    self.corners_dict[corner['label'][3:]] = corner
                    self.motion_object.set_corner_position(corner['label'][3:], corner) 

    def set_save_dir(self):
        saving_dir = QFileDialog.getExistingDirectory(self, "Open Directory", 
                            "../data", options=QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)

        sys.stdout.write(f"Saving images of dry run set to: {saving_dir}")
        sys.stdout.flush()
        self.save_dir = Path(saving_dir)
        if saving_dir == '':
            msg = QMessageBox()
            msg.setText(f"Saving images for dry set to {self.save_dir}")
            msg.setIcon(QMessageBox.Warning)
            msg.exec()
        else:
            self.ui.save_dir_path_edit.setText(str(self.save_dir))

    def set_exposure(self):
        exp_ms = self.ui.exposure_edit.text()
        try:
            int_exp = int(exp_ms)
        except:
            self.ui.exposure_edit.setText("")
            int_exp = None
        finally:
            self.exposure = int_exp

        sys.stdout.write(f"Exposure set to : {self.exposure} (ms)\n")
        sys.stdout.flush()


    @staticmethod
    def generate_events_to_acq(self, positions):
        events = []
        for one_position in positions:
            event = {}
            event['axis'] = {'time': 0, 'position': int(one_position['label'][3:])}
            event['x'] = one_position['x']
            event['y'] = one_position['y']
            event['z'] = one_position['z']
            event['channel'] = {'group': 'imaging', 'config': 'phase_slow'}
            event['exposure'] = self.exposure
            event['min_start_time'] = 0
            events.append(event)
        return events

    def acquire_dry_run(self):
        print("Calle acquired function ... ")
        

def main():
    app = QApplication(sys.argv)
    window = PosGenWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()