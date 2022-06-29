# Reference : https://github.com/safreita1/malnet-image
import json
import math
import os
import sys

from PIL import Image as im


def read_json_list(file_path):
    json_list = None
    if os.path.isfile(file_path):
        with open(file_path, 'r') as in_file:
            json_list = list(in_file)
    return json_list


def createGreyScaleImage(filename, binary_data, width=None):
    """
    Create greyscale image from binary data. Use given with if defined or create square size image from binary data.
    :param filename: image filename
    """
    size = get_size(len(binary_data), width)
    save_file(filename, binary_data, size, 'L')


def createRGBImage(filename, binary_data, width=None):
    """
    Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
    :param filename: image filename
    """
    index = 0
    rgb_data = []

    # Create R,G,B pixels
    while (index + 3) < len(binary_data):
        R = binary_data[index]
        G = binary_data[index + 1]
        B = binary_data[index + 2]
        index += 3
        rgb_data.append((R, G, B))
    size = get_size(len(rgb_data), width)
    save_file(filename, rgb_data, size, 'RGB')


def save_file(filename, data, size, image_type):
    try:
        image = im.new(image_type, size)
        image.putdata(data)
        imagename = filename + '_' + image_type + '.png'

        image.save(imagename)
        print('The file', imagename, 'saved.')
    except Exception as err:
        print(err)


def get_size(data_length, width=None):
    # source Malware images: visualization and automatic classification by L. Nataraj
    # url : http://dl.acm.org/citation.cfm?id=2016908
    if width is None:  # with don't specified any with value
        size = data_length
        if (size < 10240):
            width = 32
        elif (10240 <= size <= 10240 * 3):
            width = 64
        elif (10240 * 3 <= size <= 10240 * 6):
            width = 128
        elif (10240 * 6 <= size <= 10240 * 10):
            width = 256
        elif (10240 * 10 <= size <= 10240 * 20):
            width = 384
        elif (10240 * 20 <= size <= 10240 * 50):
            width = 512
        elif (10240 * 50 <= size <= 10240 * 100):
            width = 768
        else:
            width = 1024

        height = int(size / width) + 1
    else:
        width = int(math.sqrt(data_length)) + 1
        height = width

    return (width, height)


def run(arg_list):
    json_file_path = arg_list[0]
    json_data_list = read_json_list(json_file_path)

    for json_str in json_data_list:
        json_data = json.loads(json_str)
        label = json_data['label']
        class_name = "class_{}".format(label)
        if not os.path.isdir(class_name):
            os.makedirs(class_name)
        byte_entropy = json_data['byteentropy']
        hash_val = json_data["sha256"]
        createGreyScaleImage(str(hash_val), byte_entropy)
        createRGBImage(str(hash_val), byte_entropy)
        os.system("mv *.png {}".format(class_name))


if __name__ == "__main__":
    run(sys.argv[1:])
