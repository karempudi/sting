# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\qt_ui_files\run_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_RunWindow(object):
    def setupUi(self, RunWindow):
        RunWindow.setObjectName("RunWindow")
        RunWindow.resize(345, 307)
        self.centralwidget = QtWidgets.QWidget(RunWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(50, 60, 75, 23))
        self.start_button.setObjectName("start_button")
        self.stop_button = QtWidgets.QPushButton(self.centralwidget)
        self.stop_button.setGeometry(QtCore.QRect(50, 100, 75, 23))
        self.stop_button.setObjectName("stop_button")
        self.load_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_button.setGeometry(QtCore.QRect(50, 20, 75, 23))
        self.load_button.setObjectName("load_button")
        self.simulation_check = QtWidgets.QCheckBox(self.centralwidget)
        self.simulation_check.setGeometry(QtCore.QRect(50, 140, 70, 17))
        self.simulation_check.setObjectName("simulation_check")
        RunWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RunWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 345, 21))
        self.menubar.setObjectName("menubar")
        RunWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RunWindow)
        self.statusbar.setObjectName("statusbar")
        RunWindow.setStatusBar(self.statusbar)

        self.retranslateUi(RunWindow)
        QtCore.QMetaObject.connectSlotsByName(RunWindow)

    def retranslateUi(self, RunWindow):
        _translate = QtCore.QCoreApplication.translate
        RunWindow.setWindowTitle(_translate("RunWindow", "MainWindow"))
        self.start_button.setText(_translate("RunWindow", "Start"))
        self.stop_button.setText(_translate("RunWindow", "Stop"))
        self.load_button.setText(_translate("RunWindow", "Load"))
        self.simulation_check.setText(_translate("RunWindow", "Simulation"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RunWindow = QtWidgets.QMainWindow()
    ui = Ui_RunWindow()
    ui.setupUi(RunWindow)
    RunWindow.show()
    sys.exit(app.exec_())

