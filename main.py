# -*- coding: utf-8 -*-

import base64
import io
from PIL import Image, ImageChops
import numpy as np

import requests
import os

SERVER = 'http://localhost:5003'

# This program is captcha cracker
# Problem to solve:
# I took part in qualification round of GreyCTF
# GreyCTF is task based information security competition (CTF - capture the flag)
# That is task description:

# You have web-application with HTML-page with captcha
# Page contains your Score
# You solve captcha and send solved captcha text as answer
# If all is ok, your Score will increase on 1, else it won't change
# If Score is 100, you will get flag (your reward, special string)
# But you have to solve these 100 captchas in 120 seconds

# Manually it is almost impossible
# It has to be solved automatically

# Possible solutions:
# - to use text recognizing library based on neural networks
# - to use computer vision algorithms
# - to use pattern based comparation

# I tried to use first case: python library Pytesseract
# Unsuccessfully
# Captcha solving was too long
# Maybe, I made mistakes
# Maybe, it was my low-power laptop
# So, after many attempts, I left this idea

# That time computer vision algorithms seemed too complex for me
# Captchas were actually easy-crackable, algorithms were more misunderstandable

# So, I decided to write my own solver
# Precision was less important for me than speed
# My solver work process description:
# - script makes GET-request to server and server returns HTML-page with captcha
# - if we looks at this page content, we will see where captcha is located
# - captcha image is encoded in base64 string 
# - we find and extract this base64-content from HTML
# - next step we convert this string to Image object of Pillow library
# - captcha's letters are in bad quality and have background pixels around self
# - so, firstly, we make our image monochromic
# - then, we binarize the image
# - remove background pixels and thin lines around letters
# - compare every pixels of image with previosly saved patterns
# - for each symbol in alphabet we count percent of equal pixels
# - sort alphabet symbols by this percent
# - the most probable symbol has the greatest percent value
# - skip letters, which are impossible to be recognized
# - it is possible, because if even we make a mistake, server just will generate new captcha
# - send answer, check correctness and repeat for new captcha

# Note: real script looked more simple and had bad code quality
# It was edited and optimized as NSU Image Processing laboratory work 
# It works, but it doesn't look as battle program


def extract_base64_from_response(resp):
    # Find base64-encoded image and extract it
    start = resp.text.find('src="data:image/jpeg;base64,') + 28
    end = resp.text.find('"', start)
    data = resp.text[start:end]
    return data


def base64_to_img(base64data):
    # Convert base64 string to Image object of Pillow library
    msg = base64.b64decode(base64data)
    buf = io.BytesIO(msg)
    return Image.open(buf)


def grayscale(img: Image):
    # Count mean value of red, green and blue channels
    # Actually, formula is (R + G + B) / 3
    # But due to the pixel value limitation (technical aspect of Pillow library)
    # Real formula becomes R / 3 + G / 3 + B / 3
    Z = ImageChops.constant(img, 0)
    R, G, B = img.split()  # Split channels
    R = ImageChops.add(R, Z, scale=3)  # (R + 0) / 3
    G = ImageChops.add(G, Z, scale=3)  # (G + 0) / 3
    B = ImageChops.add(B, Z, scale=3)  # (B + 0) / 3
    M = ImageChops.add(ImageChops.add(R, G), B)
    img = Image.merge(img.mode, (M, M, M))
    return img


@np.vectorize
def binarize_array(a, alpha):
    # Array binarizing function
    # If value is less than alpha - then 0, else - 1
    if a < alpha:
        return 0
    return 1


def binarize(gray_img: Image, alpha=128):
    # Binarize image and return array of 0 and 1
    R, _, _ = gray_img.split()  # Choose only one channel
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

        # Count white pixels number in column
        white_pxs = np.sum(col)

        # If black pixels number in column is greater than 1 it is part of letter
        # Else it is part of space around letter
        if height - white_pxs > 1:
            stop += 1
        else:
            letters.append(data[start:stop])
            start = i
            stop = i
    letters.append(data[start:stop])

    # Remove empty values
    letters = [letter for letter in letters if letter.size != 0]

    # Convert 2D-array to 1D-array
    letters = [letter.flatten() for letter in letters]
    return letters[:]


def get_character(letter, abc):
    possible_keys = set()

    for key, code in abc:
        # Skip, if pixels number in letter is not equal to pixels number in pattern
        # There is possible place for mistakes because of letter defects
        # Actually equal letter and pattern can be not equal by pixels number
        # But we just have to load enough number of examples
        if key.size != letter.size:
            continue

        # Count equal pixels number
        equal = letter.size - np.sum(np.logical_xor(key, letter))

        # Count their ratio to all pixels number in image
        ratio = equal / letter.size
        possible_keys.add((code, ratio))

    # Sort decoded letters in descending order by their percent of equalness
    possible_keys = sorted(possible_keys, key=lambda x: x[1], reverse=True)
    print(possible_keys)

    if possible_keys:
        # Choose the most probable symbol
        return possible_keys[0][0]
    return None


def load_examples():
    # Load files with patterns
    # A file contains string base64-encoding
    # Convert this string to Image object of Pillow library
    examples = []
    for filename in os.listdir("examples"):
        with open(os.path.join("examples", filename)) as file:
            base64data = file.read()
            image = base64_to_img(base64data)
        examples.append((image, filename))
    return examples


def load_abc(examples):
    # Patterns are represented as pairs
    # examples = [
    #     (<Image>, <str>)
    # ]
    # <Image> - Image object of Pillow library
    # <str> - captcha text (code)

    abc = []

    for image, code in examples:
        # Extract "letters" from pattern, which were decoded
        # "Letter" is some object that corresponds some alphabet symbol
        # In my case it is numpy 1D-array
        letters = get_letters(image)

        # Skip, if letters number is not equal to letters number in pattern
        if len(letters) != len(code):
            continue

        # Else save symbol for letter
        for i in range(len(letters)):
            abc.append((letters[i], code[i]))
    return abc


# ----- TRAINING -----

def train():
    # To decode letters we need their examples
    # In train mode script asks captcha from server
    # Then extracts and shows to us
    # We input captcha text to program interface
    # Script saves image as file with decoded text in filename 
    resp = requests.get(f"{SERVER}")

    while True:
        try:
            base64data = extract_base64_from_response(resp)
            img = base64_to_img(base64data)
            img.show()

            filename = input()
            with open(os.path.join("examples", filename), "w") as dump:
                dump.write(base64data)

        except Exception as error:
            print("EXCEPTION")
            print(error.with_traceback(None))
        finally:
            # Send new request to server (it can be GET-request, not POST)
            resp = requests.post(f"{SERVER}/submit", data={"captcha": ""})


# ----- TESTING -----

def test():
    # Attack mode - load 100 captchas and solve them
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

        # Send answer to server
        resp = requests.post(
            f"{SERVER}/submit",
            data={"captcha": word},
            cookies=resp.cookies
        )

        # Check that solution is correct
        # Server returns page with message about solution correctness
        ok = resp.text.find('<div class="alert alert-success" role="alert">')
        if ok != -1:
            # If ok, we'll increase counter of solved captchas
            i += 1

            print("\u001b[32m SUCCESS \033[0m")
            score_start = resp.text.find("Score: ")
            score_legth = resp.text[score_start:].find("<")
            print(resp.text[score_start:score_start + score_legth])
        else:
            # Else display warning about miscorresponding
            print("\u001b[33m MISMATCH \033[0m")

        # Check flag existness in document and display it
        flag_start = resp.text.find("grey{")
        flag_stop = resp.text.find("}")
        if flag_start != -1 and flag_stop != -1:
            print(resp.text[flag_start:flag_stop + 1])

    # Show accuracy
    print(i / steps)


train()
# test()
