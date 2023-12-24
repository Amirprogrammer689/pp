import csv
import os
import typing

def write_file(file_name: str, data: list[list[str]]) -> None:
    '''Записывает данные в csv файл
    Args:
        file_name (str): Имя файла аннотации
        data (list[list[str]]): Данные для записи
    '''
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for row in data:
            writer.writerow(row)

def create_annotation(path_data: str, path_to_annotation: str) -> None:
    '''Создаёт аннотацию
    Args:
        path_data (str): Путь до данных
        path_to_annotation (str): Путь до файла аннотации
    '''
    data = [["full_path", "path", "class"]]

    for dirpath, dirnames, filenames in os.walk(path_data):
        for file in filenames:
            full_path = os.path.abspath(os.path.join(dirpath, file))
            relative_path = os.path.relpath(full_path, path_data)
            class_name = os.path.basename(dirpath)
            data.append([full_path, relative_path, class_name])

    write_file(path_to_annotation, data)

if __name__ == "__main__":
    path_to_annotation = "annotations_1.csv"
    path_data = "dataset"
    create_annotation(path_data, path_to_annotation)