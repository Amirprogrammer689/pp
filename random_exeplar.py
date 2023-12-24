import os

def get_next(class_name) -> str:

    '''
    Возвращает следующий путь к файлу из указанного класса.

    Args:
        class_name (str): Название класса, для которого нужно получить следующий путь.

    Returns:
        str: Следующий путь к файлу из указанного класса, или None, если путь не найден.
    '''

    path = os.path.join('dataset', class_name)
    class_names = os.listdir(path)
    for i in range (len(class_names)):
        if class_names[i] is not None:
            print(os.path.join(path, class_names[i]))
        else:
            return None
    
def main():

    '''
    Основная функция программы
    '''

    get_next("brown bear")
    get_next("polar bear")

if __name__ == "__main__":
    main()