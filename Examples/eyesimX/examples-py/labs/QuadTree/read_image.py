#!/usr/bin/env python
# Lab R4
# Image reading helper function (Python)
# Written by Jai Castle

from eye import *
import ctypes


def read_image_file(filename):
    """
    :param filename: The name of the PBM image file to read
    :return: The image as an array of integers, the width, the height
    """
    f = open(filename, 'r')
    f.readline()
    header = f.readline().strip()
    header = header.split(' ')
    length = int(header[0]) * int(header[1])
    img = []
    line = f.readline()

    while len(line) > 0:
        for x in line.split():
            img.append(int(x))
        line = f.readline()

    if length != len(img):
        raise Exception("Size mismatch in file between header and contents.")

    return img, int(header[0]), int(header[1])


if __name__ == '__main__':
    image, width, height = read_image_file('corner.pbm')
    LCDImageStart(0, 0, width, height)
    LCDImageSize(len(image))
    byte_array = [0 if i == 0 else 255 for i in image]
    x = (ctypes.c_byte * len(byte_array))(*byte_array)
    LCDImageBinary(x)
input()