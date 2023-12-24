import csv
import os
import shutil
import random


def write_file(file_name: str, data: list[list[str]]) -> None:
    '''Записывает данные в csv файл'''
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for row in data:
            writer.writerow(row)


def create_dataset_random_and_annotations(start_path: str, end_path: str, file_name: str) -> None:
    '''Копирует файлы и создает файл аннотации'''
    random_names = []
    data = [["full_path", "path", "class"]]
    dirs = os.listdir(start_path)

    if not os.path.isdir(end_path):
        os.makedirs(end_path)

    for dir in dirs:
        path = os.path.join(start_path, dir)
        files = os.listdir(path)

        for file in files:
            random_file_name = random.randint(0, 10000)

            source_file = os.path.join(start_path, dir, file)
            destination_file = os.path.join(end_path, str(random_file_name) + ".jpg")

            shutil.copyfile(source_file, destination_file)

            full_path = os.path.abspath(os.path.join(end_path, str(random_file_name) + ".jpg"))
            relative_path = os.path.join(end_path, str(random_file_name) + ".jpg")
            data.append([full_path, relative_path, dir])

    write_file(file_name, data)


if __name__ == "__main__":
    start_path = "dataset"
    end_path = "dataset_random"
    file_name = "annotations_3.csv"
    create_dataset_random_and_annotations(start_path, end_path, file_name)
