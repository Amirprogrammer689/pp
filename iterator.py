import csv
import typing
from typing import Optional

def read_file(path: str) -> list[list[str]]:
    '''
    Читает файл CSV и возвращает список списков, содержащих пути к файлам и классы.

    Аргументы:
    path (str): Путь к файлу CSV.

    Возвращает:
    list[list[str]]: Список списков, содержащих пути к файлам и классы.
    '''
    files: list[list[str]] = []
    with open(path, "r") as csvfile:
        reader: csv.DictReader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            files.append([row["full_path"], row["class"]])
    return files


class Iterator:
    def __init__(self, files_class: str, path: str) -> None:
        '''
        Инициализирует итератор с классом файла и путем к файлу.
        
        Аргументы:
        files_class (str): Класс файлов для перебора.
        path (str): Путь к файлу CSV.
        '''
        self.file_class: str = files_class
        csv_f: list[list[str]] = read_file(path)
        self.annotation: list[list[str]] = [i for i in csv_f if i[1] == self.file_class]

    def __next__(self) -> Optional[str]:
        '''
        Получает следующий путь к файлу на основе указанного класса.

        Возвращает:
        Optional[str]: Следующий путь к файлу или None, если больше нет путей.
        '''
        if self.annotation:
            item_0: str = self.annotation[0][0]
            self.annotation: list[list[str]] = self.annotation[1:]
            return item_0
        else:
            return None


if __name__ == "__main__":
    path_to_annotation: str = "annotations_1.csv"
    fileIterator: Iterator = Iterator("brown bear", path_to_annotation)
    fileIterator: Iterator = Iterator("polar bear", path_to_annotation)
    while True:
        next_val: Optional[str] = next(fileIterator)
        print(next_val)
        if next_val is None:
          break
    