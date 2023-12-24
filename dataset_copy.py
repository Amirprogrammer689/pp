import csv
import os
import shutil
import typing


def write_file(file_name: str, data: list[list[str]]) -> None:
    '''Записывает данные в csv файл
    Args:
        file_name (str): Имя файла аннотации 
        data (list[list[str]]): Данные 
    '''
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for row in data:
            writer.writerow(row)


def copy_files(start_path: str, end_path: str) -> None:
    '''Копирует файлы
    Args:
        start_path (str): Исходный путь
        end_path (str): Целевой путь
    '''
    if not os.path.exists(end_path):
        os.makedirs(end_path)

    for root, dirs, files in os.walk(start_path):
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(end_path, os.path.basename(root) + "_" + file)
            shutil.copyfile(source_file, destination_file)


def create_annotation(start_path: str, end_path: str, file_name: str) -> None:
    '''Создает аннотацию
    Args:
        start_path (str): Исходный путь
        end_path (str): Целевой путь
        file_name (str): Имя файла аннотации
    '''
    data = [["full_path", "path", "class"]]

    for root, dirs, files in os.walk(start_path):
        for file in files:
            full_path = os.path.abspath(os.path.join(end_path, os.path.basename(root) + "_" + file))
            relative_path = os.path.relpath(full_path, end_path)
            class_name = os.path.basename(root)
            data.append([full_path, relative_path, class_name])

    write_file(file_name, data)


if __name__ == "__main__":
    start_path = "dataset"
    end_path = "dataset_copy"
    file_name = "annotations_2.csv"
    copy_files(start_path, end_path)
    create_annotation(start_path, end_path, file_name)
