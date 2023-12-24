import os
import sys
import typing
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QFileSystemModel

from visual import Ui_MainWindow
import create_annotation, dataset_copy, create_dataset_random, iterator

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.polar_iterator = None
        self.brown_iterator = None
        self.dirModel = None
        self.start_path = None
        
        self.ui.button_annotation.clicked.connect(self.click_button_annotation)
        self.ui.button_init_start_path.clicked.connect(self.click_init_start_path)
        self.ui.button_dataset_copy.clicked.connect(self.click_button_dataset_copy)
        self.ui.button_dataset_rand.clicked.connect(self.click_button_dataset_rand)

        self.ui.button_polar.clicked.connect(self.click_next_polar)
        self.ui.button_brown.clicked.connect(self.click_next_brown)
        self.ui.button_open_annotation.clicked.connect(self.click_open_annotation)


    def click_init_start_path(self):
        self.start_path = QFileDialog.getExistingDirectory(self, "Выберите путь к датасету")


    '''Функционал кнопки для пролистывания белых медведей
    Parameters
    —--------   
    self: Новый объект
    '''
    def click_next_polar(self):
        path = next(self.polar_iterator)
        self.resize_image(path)

    '''Функционал кнопки для пролистывания бурых медведей
    Parameters
    —--------
    self: Новый объект
    '''
    def click_next_brown(self):
        path = next(self.brown_iterator)
        self.resize_image(path)

    '''Функция для масштабирования картинок
    Parameters
    —--------
    self: Новый объект
    path: Путь
    '''
    def resize_image(self, path):
        pixmap = QPixmap(path)
        self.ui.image.setPixmap(pixmap)

    '''Функционал для получения аннотации
    Parameters
    —--------
    self: Новый объект
    '''
    def click_open_annotation(self):
        filter = "csv(*.csv)"
        path = QFileDialog.getOpenFileName(filter=filter)
        self.polar_iterator = iterator.Iterator("polar bear", path[0])
        self.brown_iterator = iterator.Iterator("brown bear", path[0])

    '''Функционал для создания аннотации
    Parameters
    —--------
    self: Новый объект
    '''
    def click_button_annotation(self):
        if self.start_path:
            path_to_annotation: str = "annotations_1.csv"
            create_annotation.create_annotation(os.path.relpath(self.start_path), path_to_annotation)
        else:
            print("Путь к датасету не выбран")

    '''Функционал копирования аннотации и dataset 
    Parameters
    —--------
    self: Новый объект
    '''
    def click_button_dataset_copy(self):
        if self.start_path:
            end_path: str = "dataset_copy"
            file_name: str = "annotations_2.csv"
            dataset_copy.copy_files(os.path.relpath(self.start_path), end_path)
            dataset_copy.create_annotation(os.path.relpath(self.start_path), os.path.relpath(end_path), file_name)
        else:
            print("Путь к датасету не выбран")

    '''Функционал копирования аннотации и dataset с рандомными номерами
    Parameters
    —--------
    self: Новый объект
    '''
    def click_button_dataset_rand(self):
        if self.start_path:
            end_path: str = "dataset_random"
            file_name: str = "annotations_3.csv"
            create_dataset_random.create_dataset_random_and_annotations(os.path.relpath(self.start_path), os.path.relpath(end_path), file_name)
        else:
            print("Путь к датасету не выбран")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())