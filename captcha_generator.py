from captcha.image import ImageCaptcha
import numpy as np
from PIL import  Image
import random
import sys
import itertools

number = ['0','1','2','3','4','5','6','7','8','9']


def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def length_N_comb_captcha_text(char_set=number,captcha_size=4):
    count = 0
    result = []
    for comb in itertools.product(char_set,repeat=captcha_size):
        count += 1
        result.append(list(comb))
    return result


def gen_captcha_text_and_image():
    image = ImageCaptcha()
    result = length_N_comb_captcha_text()
    for x in range(len(result)):
        captcha_text = result[x]
        captcha_text = ''.join(captcha_text)
        captcha = image.generate(captcha_text)
        image.write(captcha_text,'captcha/images/'+captcha_text+'.jpg')
        sys.stdout.write('\r>> Creating image %d/%d' % (x+1, len(result)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("done")


if __name__ == '__main__':
    gen_captcha_text_and_image()