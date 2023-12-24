import csv
import pandas as pd
import cv2 
import matplotlib.pyplot as plt

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

def create_df_two_columns(list_file_info: list[list[str]]) -> pd.DataFrame:
    df_two_columns = pd.DataFrame(columns=["Name", "Absolute Path"])
    for file_info in list_file_info:
        df_two_columns.loc[len(df_two_columns)] = [file_info[1], file_info[0]]
    return df_two_columns

def create_df(list_file_info: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(columns=["Name", "Absolute Path", "Class Id", "Width", "Height", "Depth"])
    for file_info in list_file_info:
        class_id = 0 if file_info[1] == "brown bear" else 1
        im = cv2.imread(file_info[0])
        h, w, c = im.shape
        df.loc[len(df)] = [file_info[1], file_info[0], class_id, w, h, c]
    return df

def calculate_static_info(df: pd.DataFrame) -> dict:
    static_info = {"Height": df["Height"].mean(), "Width": df["Width"].mean(),
                   "Depth":  df["Depth"].mean(), "Class Id": df["Class Id"].mean()}
    return static_info

def filter_df_by_class_id(df: pd.DataFrame, class_id: int) -> pd.DataFrame:
    filtered_df = df.loc[df["Class Id"] == class_id]
    return filtered_df

def filter_df_by_size_and_class_id(df: pd.DataFrame, class_id: int, max_width: int, max_height: int) -> pd.DataFrame:
    filtered_df_size = df.loc[(df["Class Id"] == class_id) & (df["Width"] <= max_width) & (df["Height"] <= max_height)]
    return filtered_df_size

def calculate_pixel_count_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['Pixel Count'] = df['Height'] * df['Width'] * df['Depth']
    grouped_df = df.groupby('Class Id')['Pixel Count'].agg(['max', 'mean', 'min'])
    return grouped_df

def read_and_split_image(df: pd.DataFrame, class_id: int) -> tuple:
    filter = df.loc[df["Class Id"] == class_id]
    path = filter["Absolute Path"].sample(1).values[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    r, g, b = cv2.split(img)
    return r, g, b

def plot_histograms(r, g, b):
    plt.plot(range(256), cv2.calcHist([r], [0], None, [256], [0, 256]), color='red')
    plt.plot(range(256), cv2.calcHist([g], [0], None, [256], [0, 256]), color='green')
    plt.plot(range(256), cv2.calcHist([b], [0], None, [256], [0, 256]), color='blue')

    plt.xlabel("OX")
    plt.ylabel("OY")
    plt.title("Histogramm")

    plt.show()
    
def main():
    file_info = read_file("annotations_1.csv")
    df_two_columns = create_df_two_columns(file_info)
    df = create_df(file_info)
    static_info = calculate_static_info(df)
    filtered_df = filter_df_by_class_id(df, 1)
    filtered_df_size = filter_df_by_size_and_class_id(df, 1, 1000, 1000)
    pixel_count_stats = calculate_pixel_count_stats(df)
    r, g, b = read_and_split_image(df, 1)
    plot_histograms(r, g, b)

    print("2 columns:")
    print(df_two_columns)
    print("\n6 columns:")
    print(df)
    print("\nstatic info:")
    print(static_info)
    print("\nfiltered df:")
    print(filtered_df)
    print("\nfiltered df size:")
    print(filtered_df_size)
    print("\npixel count stats:")
    print(pixel_count_stats)

if __name__ == "__main__":
    main()
