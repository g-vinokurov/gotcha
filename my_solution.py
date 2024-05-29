# -*- coding: utf-8 -*-

import base64
import io
from PIL import Image, ImageChops
import numpy as np

import requests
import os

SERVER = 'http://localhost:5003'

# Данная программа является решателем капч
# Задача возникшая передо мной:
# Я участвовал в отборочном этапе GreyCTF
# GreyCTF - соревнование по информационной безопасности (CTF - capture the flag)
# Одна из задач была такой:

# Есть веб-приложение со страницей, на которой находится капча
# Также на странице отображается Score пользователя
# Пользователь решает капчу и отправляет текст в ответ
# Если всё верно, его Score увеличивается на 1, иначе он не меняется
# Когда Score станет 100, пользователь получит награду - Флаг (особую строку)
# Нужно решить 100 капч за 120 секунд

# Руками решить такую задачу практически невозможно
# Поэтому этот процесс требует автоматизации

# Какие есть решения:
# - использовать библиотеки для распознавания текста на свёрточных нейросетях
# - использовать алгоритмы компьютерного зрения
# - использовать шаблонное сопоставление (некое подобие ML)

# На момент решения задачи я попытался использовать первый вариант
# Я попробовал использовать библиотеку Pytesseract, но ничего не получилось
# Капча читалась слишком долго
# Возможно, где-то накосячил я
# Возможно просто не хватало мощности слабенького ноутбука
# В общем спустя множество попыток, мне пришлось отказаться от этой идеи

# Алгоритмы для компьютерного зрения показались мне тогда весьма сложными
# Капчи были довольно простыми, сложнее было понять, как эти алгоритмы работают

# В итоге я решил написать свой скрипт, который смог бы решить эту задачу решать
# Мне не была важна особая точность, главное была скорость
# Собственно, суть работы скрипта:
# - скрипт делает запрос к серверу и получает в ответ HTML-страницу с капчей
# - посмотрев на эту HTML-страницу, мы можем точно сказать, где капча
# - изображение капчи закодировано в виде строки в кодировке base64
# - мы извлекаем из HTML-содержимого данную base64-строку
# - далее преобразовываем эту строку в объект Image библиотеки Pillow
# - буквы капчи весьма в плохом качестве и находятся на расстоянии от краев
# - поэтому мы сначала делаем наше изображение монохромным
# - потом бинаризуем его
# - удаляем фон и мелкие артефакты (тонкие линии в 1 пиксель, точки) вокруг букв
# - попиксельно сравниваем куски изображения с сохраненными образцами букв
# - для каждой буквы в алфавите указывается доля совпавших пикселей
# - список букв алфавита сортируется по этой доле
# - самая правдоподобная буква имеет самый большую долю совпадений
# - буквы, которые не удалось декодировать, отбрасываются
# - это можно сделать, так как при неверной капче сервер просто даст новую
# - отправляем результат, проверяем корректность и повторяем уже для новой капчи

# Примечание: реальный скрипт выглядел проще и хуже по качеству кода
# Скрипт отредактирован, оптимизирован и подготовлен как работа по обработке изображений
# Он полностью работоспособен, но, условно говоря, "одет на парад"


def extract_base64_from_response(resp):
    # Ищем наше закодированное в base64 изображение и возвращаем его
    start = resp.text.find('src="data:image/jpeg;base64,') + 28
    end = resp.text.find('"', start)
    data = resp.text[start:end]
    return data


def base64_to_img(base64data):
    # Конвертируем base64-строку в объект Image библиотеки Pillow
    msg = base64.b64decode(base64data)
    buf = io.BytesIO(msg)
    return Image.open(buf)


def grayscale(img: Image):
    # Берёт среднее значение каналов RGB и заменяет их на него
    # Вообще формула: (R + G + B) / 3
    # Но из-за ограничения до макс. значения при сложении компонент
    # Формула становится другой: R / 3 + G / 3 + B / 3
    Z = ImageChops.constant(img, 0)
    R, G, B = img.split()  # Разделение на каналы
    R = ImageChops.add(R, Z, scale=3)  # (R + 0) / 3
    G = ImageChops.add(G, Z, scale=3)  # (G + 0) / 3
    B = ImageChops.add(B, Z, scale=3)  # (B + 0) / 3
    M = ImageChops.add(ImageChops.add(R, G), B)
    img = Image.merge(img.mode, (M, M, M))
    return img


@np.vectorize
def binarize_array(a, alpha):
    # Функция для бинаризации массива
    # Если значение меньше порогового - это 0, иначе - 1
    if a < alpha:
        return 0
    return 1


def binarize(gray_img: Image, alpha=128):
    # Бинаризует изображение и возвращает массив из 0 и 1
    R, _, _ = gray_img.split()  # Выделение одного канала
    return binarize_array(np.asarray(R), alpha).T


def get_letters(img):
    letters = []

    img = grayscale(img)
    data = binarize(img)

    width, height = img.size

    start = 0
    stop = 0
    for i in range(width):
        col = data[i]

        # Подсчет кол-ва белых пикселей в столбце
        white_pxs = np.sum(col)

        # Если кол-во черных пекселей в столбце больше 1, то это буква
        # Иначе - это пространство между буквами
        if height - white_pxs > 1:
            stop += 1
        else:
            letters.append(data[start:stop])
            start = i
            stop = i
    letters.append(data[start:stop])

    # Убираем пустые значения - не буквы
    letters = [letter for letter in letters if letter.size != 0]

    # Превращаем двумерный массив в линейный
    letters = [letter.flatten() for letter in letters]
    return letters[:]


def get_character(letter, abc):
    possible_keys = set()

    for key, code in abc:
        # Если кол-во пикселей в букве не равно их кол-ву в образце - пропускаем
        # Здесь тонкое место - из-за дефекта изображения
        # Одинаковые буква и образец могут не совпасть по кол-ву пикселей
        # Но, в прицнипе, это решается тем, что у нас достаточно образцов
        if key.size != letter.size:
            continue

        # Считаем кол-во совпавших пикселей
        equal = letter.size - np.sum(np.logical_xor(key, letter))

        # Считаем их долю от общего числа пикселей в букве
        ratio = equal / letter.size
        possible_keys.add((code, ratio))

    # Сортируем декодированные символы по убыванию доли совпадения буквы с ними
    possible_keys = sorted(possible_keys, key=lambda x: x[1], reverse=True)
    print(possible_keys)

    if possible_keys:
        # Выбириаем наиболее вероятный символ
        return possible_keys[0][0]
    return None


def load_examples():
    # Загружает сохранненые файлы с образцами изображений
    # Из файла считывается строка в кодировке base64
    # Затем она конвертируется в объект Image библиотеки Pillow
    examples = []
    for filename in os.listdir("examples"):
        with open(os.path.join("examples", filename)) as file:
            base64data = file.read()
            image = base64_to_img(base64data)
        examples.append((image, filename))
    return examples


def load_abc(examples):
    # Загруженные образцы представлены в виде списка пар:
    # examples = [
    #     (<Image>, <str>)
    # ]
    # <Image> - изображене в формате объекта Image библиотеки Pillow
    # <str> - текст на картинке (code)

    abc = []

    for image, code in examples:
        # Извлекаем из образца "буквы", которые удалось декодировать
        # "Буква" - это некий объект, которому сопоставляется символ алфавита
        # Фактически, "буква" - это один из шаблонов какого-то символа
        # Что это за объект, зависит от нашей реализации распознавателя капчи
        # В моём случае - это 1D-массивы numpy
        letters = get_letters(image)

        # Если количество букв не равно кол-ву символов, то образец некорректен
        if len(letters) != len(code):
            continue

        # Иначе сохраняем для "буквы" её символ
        for i in range(len(letters)):
            abc.append((letters[i], code[i]))
    return abc


# ----- TRAINING -----

def train():
    # Чтобы декодировать буквы, нам нужны их образцы
    # Наш скрипт в режиме обучения запрашивает страницу с капчей
    # Извлекает изображение и показывает нам
    # Мы вводим в программу текст, который прочитали на капче
    # Программа сохраняет изображение в файл, его название - текст на капче
    resp = requests.get(f"{SERVER}")

    while True:
        try:
            base64data = extract_base64_from_response(resp)
            img = base64_to_img(base64data)
            img.show()

            # Вводим текст с капчи и сохраняем её в файл
            filename = input()
            with open(os.path.join("examples", filename), "w") as dump:
                dump.write(base64data)

        except Exception as error:
            print("EXCEPTION")
            print(error.with_traceback(None))
        finally:
            # Отправляем новый запрос на сервер (можно get вместо post)
            resp = requests.post(f"{SERVER}/submit", data={"captcha": ""})


# ----- TESTING -----

def test():
    # Рабочий режим - загружаем образцы и пытаемся решить 100 капч
    abc = load_abc(load_examples())

    resp = requests.get(f"{SERVER}")

    i = 0
    steps = 0
    while i != 100:
        steps += 1
        print(f"Processing: {i}")
        try:
            base64data = extract_base64_from_response(resp)
            img = base64_to_img(base64data)

            letters = get_letters(img)

            word = "".join(get_character(letter, abc) for letter in letters)
        except Exception as error:
            print(error.with_traceback(None))
            return

        # Отправляем решенную капчу на сервер
        resp = requests.post(
            f"{SERVER}/submit",
            data={"captcha": word},
            cookies=resp.cookies
        )

        # Проверяем, что капча решена верна
        # Сервер возвращает страницу с сообщением: верно или неверно
        ok = resp.text.find('<div class="alert alert-success" role="alert">')
        if ok != -1:
            # Если верно, увеличиваем счётчик верно решенных капч
            i += 1

            print("\u001b[32m SUCCESS \033[0m")
            score_start = resp.text.find("Score: ")
            score_legth = resp.text[score_start:].find("<")
            print(resp.text[score_start:score_start + score_legth])
        else:
            # Иначе предупреждаем о несоответствии
            print("\u001b[33m MISMATCH \033[0m")

        # Проверяем наличие флага в HTML-документе
        # Если он есть, то выводим его
        flag_start = resp.text.find("grey{")
        flag_stop = resp.text.find("}")
        if flag_start != -1 and flag_stop != -1:
            print(resp.text[flag_start:flag_stop + 1])

    # Выводим долю верно решенных капч
    print(i / steps)


train()
# test()
