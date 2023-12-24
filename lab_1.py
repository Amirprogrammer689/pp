import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def get_image(image_link, name, index):
    if not os.path.isdir(name):
        os.mkdir(name)
    picture = requests.get(image_link)
    with open(os.path.join(f"{name}/{str(index).zfill(4)}.jpg"), "wb") as saver:
        saver.write(picture.content)

def get_url_images(path, search_query):
    os.chdir(path)
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    os.chdir(dataset_dir)

    count = 0
    page = 0
    while count < 1000:
        search_query1 = search_query.replace(" ", "%20")
        url = f"https://yandex.ru/images/search?p={page}&text={search_query1}"

        driver = webdriver.Chrome()
        driver.get(url=url)
        time.sleep(5)

        try:
            driver.find_element(By.CLASS_NAME, 'CheckboxCaptcha')
            input('Пройдите капчу и нажмите Enter') 
            driver.get(url=url)
            time.sleep(5)
        except NoSuchElementException as e:
            print('Не удалось найти капчу:', e)

        images = driver.find_elements(By.CLASS_NAME, 'SimpleImage-Image')
        print(images)

        for i in images:
            image_link = i.get_attribute('src')
            get_image(image_link, search_query, count)
            count += 1
            print(image_link)
        page += 1
    driver.close()
    driver.quit()

def main():
    directory = os.getcwd()
    search_query = 'brown bear'
    #search_query = 'polar bear'
    get_url_images(directory, search_query)


if __name__ == "__main__":
    main()