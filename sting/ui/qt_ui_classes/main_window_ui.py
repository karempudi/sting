# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\qt_ui_files\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(669, 535)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.setup = QtWidgets.QGroupBox(self.centralwidget)
        self.setup.setGeometry(QtCore.QRect(20, 10, 121, 151))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setup.setFont(font)
        self.setup.setObjectName("setup")
        self.setup_button = QtWidgets.QPushButton(self.setup)
        self.setup_button.setGeometry(QtCore.QRect(20, 30, 88, 27))
        self.setup_button.setObjectName("setup_button")
        self.view_setup_button = QtWidgets.QPushButton(self.setup)
        self.view_setup_button.setGeometry(QtCore.QRect(20, 70, 88, 27))
        self.view_setup_button.setObjectName("view_setup_button")
        self.write_setup_button = QtWidgets.QPushButton(self.setup)
        self.write_setup_button.setGeometry(QtCore.QRect(20, 110, 89, 25))
        self.write_setup_button.setObjectName("write_setup_button")
        self.progress = QtWidgets.QGroupBox(self.centralwidget)
        self.progress.setGeometry(QtCore.QRect(10, 170, 631, 271))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.progress.setFont(font)
        self.progress.setObjectName("progress")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.progress)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 30, 191, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.acquisition = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.acquisition.setContentsMargins(0, 0, 0, 0)
        self.acquisition.setObjectName("acquisition")
        self.acq_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.acq_label.sizePolicy().hasHeightForWidth())
        self.acq_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.acq_label.setFont(font)
        self.acq_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acq_label.setObjectName("acq_label")
        self.acquisition.addWidget(self.acq_label)
        self.acq_pos_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.acq_pos_label.setFont(font)
        self.acq_pos_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acq_pos_label.setObjectName("acq_pos_label")
        self.acquisition.addWidget(self.acq_pos_label)
        self.acq_position_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        self.acq_position_bar.setProperty("value", 24)
        self.acq_position_bar.setObjectName("acq_position_bar")
        self.acquisition.addWidget(self.acq_position_bar)
        self.acq_n_pos = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.acq_n_pos.setReadOnly(True)
        self.acq_n_pos.setObjectName("acq_n_pos")
        self.acquisition.addWidget(self.acq_n_pos)
        self.acq_time_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.acq_time_label.setFont(font)
        self.acq_time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acq_time_label.setObjectName("acq_time_label")
        self.acquisition.addWidget(self.acq_time_label)
        self.acq_time_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        self.acq_time_bar.setProperty("value", 24)
        self.acq_time_bar.setObjectName("acq_time_bar")
        self.acquisition.addWidget(self.acq_time_bar)
        self.acq_n_time = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.acq_n_time.setReadOnly(True)
        self.acq_n_time.setObjectName("acq_n_time")
        self.acquisition.addWidget(self.acq_n_time)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.progress)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(220, 30, 191, 221))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.image_processing = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.image_processing.setContentsMargins(0, 0, 0, 0)
        self.image_processing.setObjectName("image_processing")
        self.img_process_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.img_process_label.setFont(font)
        self.img_process_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_process_label.setObjectName("img_process_label")
        self.image_processing.addWidget(self.img_process_label)
        self.img_pos_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.img_pos_label.setFont(font)
        self.img_pos_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_pos_label.setObjectName("img_pos_label")
        self.image_processing.addWidget(self.img_pos_label)
        self.img_pos_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget_2)
        self.img_pos_bar.setProperty("value", 24)
        self.img_pos_bar.setObjectName("img_pos_bar")
        self.image_processing.addWidget(self.img_pos_bar)
        self.img_n_pos = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.img_n_pos.setReadOnly(True)
        self.img_n_pos.setObjectName("img_n_pos")
        self.image_processing.addWidget(self.img_n_pos)
        self.img_time_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.img_time_label.setFont(font)
        self.img_time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_time_label.setObjectName("img_time_label")
        self.image_processing.addWidget(self.img_time_label)
        self.img_time_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget_2)
        self.img_time_bar.setProperty("value", 24)
        self.img_time_bar.setObjectName("img_time_bar")
        self.image_processing.addWidget(self.img_time_bar)
        self.img_n_time = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.img_n_time.setReadOnly(True)
        self.img_n_time.setObjectName("img_n_time")
        self.image_processing.addWidget(self.img_n_time)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.progress)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(420, 30, 191, 221))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.channel_properties = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.channel_properties.setContentsMargins(0, 0, 0, 0)
        self.channel_properties.setObjectName("channel_properties")
        self.cprop_label = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cprop_label.setFont(font)
        self.cprop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cprop_label.setObjectName("cprop_label")
        self.channel_properties.addWidget(self.cprop_label)
        self.cprop_pos_label = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cprop_pos_label.setFont(font)
        self.cprop_pos_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cprop_pos_label.setObjectName("cprop_pos_label")
        self.channel_properties.addWidget(self.cprop_pos_label)
        self.cprop_pos_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.cprop_pos_bar.setProperty("value", 24)
        self.cprop_pos_bar.setObjectName("cprop_pos_bar")
        self.channel_properties.addWidget(self.cprop_pos_bar)
        self.cprop_n_pos = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.cprop_n_pos.setReadOnly(True)
        self.cprop_n_pos.setObjectName("cprop_n_pos")
        self.channel_properties.addWidget(self.cprop_n_pos)
        self.cprop_time_label = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.cprop_time_label.setFont(font)
        self.cprop_time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cprop_time_label.setObjectName("cprop_time_label")
        self.channel_properties.addWidget(self.cprop_time_label)
        self.cprop_time_bar = QtWidgets.QProgressBar(self.verticalLayoutWidget_3)
        self.cprop_time_bar.setProperty("value", 24)
        self.cprop_time_bar.setObjectName("cprop_time_bar")
        self.channel_properties.addWidget(self.cprop_time_bar)
        self.cprop_n_time = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.cprop_n_time.setReadOnly(True)
        self.cprop_n_time.setObjectName("cprop_n_time")
        self.channel_properties.addWidget(self.cprop_n_time)
        self.controls = QtWidgets.QGroupBox(self.centralwidget)
        self.controls.setGeometry(QtCore.QRect(150, 10, 141, 151))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.controls.setFont(font)
        self.controls.setObjectName("controls")
        self.start_expt_button = QtWidgets.QPushButton(self.controls)
        self.start_expt_button.setGeometry(QtCore.QRect(30, 30, 88, 27))
        self.start_expt_button.setObjectName("start_expt_button")
        self.stop_expt_button = QtWidgets.QPushButton(self.controls)
        self.stop_expt_button.setGeometry(QtCore.QRect(30, 70, 88, 27))
        self.stop_expt_button.setObjectName("stop_expt_button")
        self.tweeze = QtWidgets.QGroupBox(self.centralwidget)
        self.tweeze.setGeometry(QtCore.QRect(320, 10, 141, 151))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tweeze.setFont(font)
        self.tweeze.setObjectName("tweeze")
        self.tweezable_button = QtWidgets.QPushButton(self.tweeze)
        self.tweezable_button.setGeometry(QtCore.QRect(20, 30, 88, 27))
        self.tweezable_button.setObjectName("tweezable_button")
        self.live_button = QtWidgets.QPushButton(self.tweeze)
        self.live_button.setGeometry(QtCore.QRect(20, 70, 88, 27))
        self.live_button.setObjectName("live_button")
        self.plots = QtWidgets.QGroupBox(self.centralwidget)
        self.plots.setGeometry(QtCore.QRect(480, 10, 141, 151))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.plots.setFont(font)
        self.plots.setObjectName("plots")
        self.growth_rates_button = QtWidgets.QPushButton(self.plots)
        self.growth_rates_button.setGeometry(QtCore.QRect(20, 30, 88, 27))
        self.growth_rates_button.setObjectName("growth_rates_button")
        self.dead_alive_button = QtWidgets.QPushButton(self.plots)
        self.dead_alive_button.setGeometry(QtCore.QRect(20, 70, 88, 27))
        self.dead_alive_button.setObjectName("dead_alive_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 669, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.setup.setTitle(_translate("MainWindow", "Setup"))
        self.setup_button.setText(_translate("MainWindow", "Setup"))
        self.view_setup_button.setText(_translate("MainWindow", "View "))
        self.write_setup_button.setText(_translate("MainWindow", "Write"))
        self.progress.setTitle(_translate("MainWindow", "Experiment Progess"))
        self.acq_label.setText(_translate("MainWindow", "Acquisition"))
        self.acq_pos_label.setText(_translate("MainWindow", "Positions"))
        self.acq_time_label.setText(_translate("MainWindow", "Time points"))
        self.img_process_label.setText(_translate("MainWindow", "Image Processing"))
        self.img_pos_label.setText(_translate("MainWindow", "Positions"))
        self.img_time_label.setText(_translate("MainWindow", "Time points"))
        self.cprop_label.setText(_translate("MainWindow", "Channel Properties"))
        self.cprop_pos_label.setText(_translate("MainWindow", "Positions"))
        self.cprop_time_label.setText(_translate("MainWindow", "Time points"))
        self.controls.setTitle(_translate("MainWindow", "Controls"))
        self.start_expt_button.setText(_translate("MainWindow", "Start"))
        self.stop_expt_button.setText(_translate("MainWindow", "Stop"))
        self.tweeze.setTitle(_translate("MainWindow", "View"))
        self.tweezable_button.setText(_translate("MainWindow", "Tweezable"))
        self.live_button.setText(_translate("MainWindow", "Live"))
        self.plots.setTitle(_translate("MainWindow", "Plots"))
        self.growth_rates_button.setText(_translate("MainWindow", "Growth Rates"))
        self.dead_alive_button.setText(_translate("MainWindow", "Dead Alive"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

