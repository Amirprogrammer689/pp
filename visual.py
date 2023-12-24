from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 400)
        MainWindow.setStyleSheet("background-color: rgb(39, 39, 39);\n"
"color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(102, 255, 152);")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.button_annotation = QtWidgets.QPushButton(self.tab)
        self.button_annotation.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_annotation.setObjectName("button_annotation")
        self.horizontalLayout.addWidget(self.button_annotation)
        self.button_init_start_path = QtWidgets.QPushButton(self.tab)
        self.button_init_start_path.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_init_start_path.setObjectName("button_init_start_path")
        self.horizontalLayout.addWidget(self.button_init_start_path)
        self.button_dataset_copy = QtWidgets.QPushButton(self.tab)
        self.button_dataset_copy.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_dataset_copy.setObjectName("button_dataset_copy")
        self.horizontalLayout.addWidget(self.button_dataset_copy)
        self.button_dataset_rand = QtWidgets.QPushButton(self.tab)
        self.button_dataset_rand.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_dataset_rand.setObjectName("button_dataset_rand")
        self.horizontalLayout.addWidget(self.button_dataset_rand)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.image = QtWidgets.QLabel(self.tab_2)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setObjectName("image")
        self.verticalLayout_2.addWidget(self.image)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.button_polar = QtWidgets.QPushButton(self.tab_2)
        self.button_polar.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_polar.setObjectName("button_polar")
        self.horizontalLayout_2.addWidget(self.button_polar)
        self.button_open_annotation = QtWidgets.QPushButton(self.tab_2)
        self.button_open_annotation.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_open_annotation.setObjectName("button_open_annotation")
        self.horizontalLayout_2.addWidget(self.button_open_annotation)
        self.button_brown = QtWidgets.QPushButton(self.tab_2)
        self.button_brown.setStyleSheet("background-color: rgb(45, 45, 45);\n"
"color: rgb(255, 50, 255);")
        self.button_brown.setObjectName("button_brown")
        self.horizontalLayout_2.addWidget(self.button_brown)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.setStretch(0, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Медведи"))
        self.button_annotation.setText(_translate("MainWindow", "Создание аннотации"))
        self.button_init_start_path.setText(_translate("MainWindow", "Инициализация пути к датасету"))
        self.button_dataset_copy.setText(_translate("MainWindow", "Копирование датасета (class_0000.jpg)"))
        self.button_dataset_rand.setText(_translate("MainWindow", "Копирование датасета (random names)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Аннотации"))
        self.image.setText(_translate("MainWindow", "TextLabel"))
        self.button_polar.setText(_translate("MainWindow", "Следующий белый медведь"))
        self.button_open_annotation.setText(_translate("MainWindow", "Открыть датасет"))
        self.button_brown.setText(_translate("MainWindow", "Следующий бурый медведь"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Обозреватель датасета"))