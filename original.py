# -*- coding: utf-8 -*-

import base64
import io
from PIL import Image

import requests
import os

SERVER = 'http://localhost:5003'

# You need solve 100 captcha in 120 seconds


def base64_to_img(base64data):
    msg = base64.b64decode(base64data)
    buf = io.BytesIO(msg)
    return Image.open(buf)


def extract_base64_from_response(resp):
    start = resp.text.find('src="data:image/jpeg;base64,') + 28
    end = resp.text.find('"', start)
    data = resp.text[start:end]
    return data


def get_letters(img):
    letters = [[]]

    w, h = img.size
    for i in range(w):
        # read pixels in row
        row = [img.getpixel((i, j)) for j in range(h)]

        # binarize pixels:
        # dark grey pixels are black (0)
        # light grey pixels are white (1)
        row = [0 if x[0] < 128 else 1 for x in row]

        # count white pixels in row
        white_pxs = sum(row)

        # check if row has only white pixels
        # white row means that we are in space between letters
        if white_pxs != h:
            # letter actually is the list of stringified digits 0 and 1
            letters[-1].extend([str(x) for x in row])
        else:
            # if row contains only white pixels, we prepare place to other letter
            if letters[-1]:
                letters.append([])

    # filter empty arrays (rows with only white pixels)
    letters = [x for x in letters if x]
    return letters[:]


def cmp(letter, abc):
    possible_keys = set()

    for key in abc.keys():
        # align strings
        common_length = max(len(key), len(letter))

        # feel up to common length by white pixels
        i1 = letter.ljust(common_length, "1")
        i2 = key.ljust(common_length, "1")

        # count number of equal pixels
        count = sum(1 if i1[i] == i2[i] else 0 for i in range(common_length))

        # we save key and percent of equal pixels
        # there can be more than one decoded symbol
        activation = count / common_length
        possible_keys.add((abc[key], activation))

    # we sort symbols by descending order by their percents
    possible_keys = sorted(possible_keys, key=lambda x: x[1], reverse=True)
    print(possible_keys)

    if possible_keys:
        # we choose the most likely symbol
        return possible_keys[0][0]
    return None


def load_examples():
    examples = []
    for filename in os.listdir("examples"):
        with open(os.path.join("examples", filename)) as file:
            base64data = file.read()
            image = base64_to_img(base64data)
        examples.append((image, filename))
    return examples


def load_abc(examples):
    abc = dict()

    # for each example we split our image on 'letters' and assign it with symbol
    # for each 'letter' - string of 0 and 1 we save decoded ABC symbol
    for image, code in examples:
        letters = get_letters(image)
        for i in range(len(letters)):
            as_str = "".join(letters[i])
            abc[as_str] = code[i]
    return abc


# ----- TRAINING -----

def train():
    resp = requests.get(f"{SERVER}")

    while True:
        try:
            base64data = extract_base64_from_response(resp)
            img = base64_to_img(base64data)
            img.show()

            # create training set manually
            filename = input()
            with open(os.path.join("examples", filename), "w") as dump:
                dump.write(base64data)
        except Exception as error:
            print('EXCEPTION')
            print(error.with_traceback())
        finally:
            # server returns new page with captcha as response
            resp = requests.post(f"{SERVER}/submit", data={"captcha": ""})


# ----- TESTING -----

def test():
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
            letters = ["".join(x) for x in letters]

            # compare current letters with training set
            letters = "".join([cmp(x, abc) for x in letters])
        except Exception:
            print("\u001b[31m FAILURE \033[0m")
            return

        # send cracked captcha to server
        resp = requests.post(
            f"{SERVER}/submit",
            data={"captcha": letters},
            cookies=resp.cookies
        )

        # check on success
        ok = resp.text.find('<div class="alert alert-success" role="alert">')
        if ok != -1:
            i += 1

            print("\u001b[32m SUCCESS \033[0m")
            score_start = resp.text.find("Score: ")
            score_legth = resp.text[score_start:].find("<")
            print(resp.text[score_start:score_start + score_legth])
        else:
            print("\u001b[33m MISMATCH \033[0m")

        flag_start = resp.text.find("grey{")
        flag_stop = resp.text.find("}")
        if flag_start != -1 and flag_stop != -1:
            print(resp.text[flag_start:flag_stop + 1])

    print(i / steps)


# train()
test()
