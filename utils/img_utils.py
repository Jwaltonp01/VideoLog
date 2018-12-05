# coding=utf-8

import math
import os
import ntpath
import urllib

import cv2
import numpy as np
import shutil

import requests

from utils import labels

num_image_in_file = 4800


def resize_img(img_path, img_width, img_height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_width, img_height)
    cv2.imwrite(img, img_path)


# AIが特定できるラベル
def save_images_to_cache(npy_path, X, paths):
    print("Saved processed images to cache file: " + npy_path)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    np.save(npy_path + "/" + "paths.npy", paths)
    np.save(npy_path + "/" + "X.npy", X)


def save_images_to_cache(npy_path, X, paths, file_no):
    print("Saved processed images to cache file: " + npy_path)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    np.save(npy_path + "/" + str(file_no) + "_paths.npy", paths)
    np.save(npy_path + "/" + str(file_no) + "_X.npy", X)


def rotateByAngle(img, angle, cols, rows):
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def rotateByMatrix(img, cols, rows, M):
    return cv2.warpAffine(img, M, (cols, rows))


# 渡したパス（path)を確認して、そのパスの中の画像の全てを返す
# path：パス
# ext：ファイル・タイプ
def get_files(path, ext=".jpg"):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(ext)]


#
def permutation(x):
    if len(x) != 0:
        for i in range(5):
            x = np.random.permutation(x)
    return x


def load_and_resize(paths, img_rows, img_cols, cache_path):
    X = []
    files = []
    count = 0
    file_count = 0

    for i in range(len(paths)):
        path = paths[i]
        try:
            img = cv2.imread(path)
            img = square_wrap(img)
            img = cv2.resize(img, (img_rows, img_cols))
            X.append(img)
            files.append(path)
            count += 1

            if count % num_image_in_file == 0 or (count == len(paths)):
                print("Saving processed image: " + str(count) + "/" + str(len(paths)))
                save_images_to_cache(cache_path, X, files, file_count)
                X = []
                files = []
                file_count += 1

        except:
            print("warning: cannot load " + path)

    return np.array(X), np.array(files)


# Load images to training cache
def load_resize_argument(paths, img_rows, img_cols, cache_path, enable_mods):
    # enable_mods:
    #   add rotated images cache (add modifications)

    X = []
    files = []
    count = 0
    file_count = 0

    angles = [-90, -45, 45, 90]
    angles2 = [-90, -60, -30, 30, 60, 90]
    M1 = [cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), angle, 1) for angle in angles]
    M2 = [cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), angle, 1) for angle in angles]

    for i in range(len(paths)):
        path = paths[i]
        try:
            img = cv2.imread(path)
            img = square_wrap(img)
            img = cv2.resize(img, (img_rows, img_cols))

            X.append(img)
            files.append(path)
            count += 1

            if enable_mods:
                if count % num_image_in_file == 0:
                    print("Saving processed image: " + str(count) + "/" + str(len(paths)))
                    save_images_to_cache(cache_path, X, files, file_count)
                    X = []
                    files = []
                    file_count += 1
                    M = M2

                for m in M:
                    rotated_img = rotateByMatrix(img, img_cols, img_rows, m)

                    X.append(rotated_img)
                    files.append(path)
                    count += 1
                    if count % num_image_in_file == 0:
                        print("Saving processed image: " + str(count) + "/" + str(len(paths)))
                        save_images_to_cache(cache_path, X, files, file_count)
                        X = []
                        files = []
                        file_count += 1

        except:
            print("warning: cannot load " + path)

    if len(X) > 0:
        print("Saving processed image: " + str(count) + "/" + str(len(paths)))
        save_images_to_cache(cache_path, X, files, file_count)
        file_count += 1

    return cache_path, file_count


# Load single image file
# 渡されたパス（npy_path）を確認して、npy_pathにある画像ファイルの全てをロードする
#
# npy_path：ファイルのバス（numpyのフォーマット）
# file_no：ファイル数
def load_cache_data(cache_path, file_no):
    # 新しいファイルパスを作って、npyのフォーマットに変更する
    path = cache_path + "/" + str(file_no) + "_paths.npy"

    # ファイルのオブジェクト
    file = cache_path + "/" + str(file_no) + "_X.npy"

    # ファイルのパスとファイルが存在しているかの確認
    if os.path.exists(path) and os.path.exists(file):
        # ファイルのパスをロードする（numpyのフォーマット）
        paths = np.load(path)
        # 画像ファイルをロードする（numpyのフォーマットで）
        img_data = np.load(file)

        return img_data, paths

    # ファイルが存在していなければ、空配列を返す
    return [], []


# Loads single image file
# Load images within the cache_path path for training sessions
# トレーニング・セッションのためにcache_pathにある画像をロードする
#
# 参考：
# https://stackoverflow.com/questions/46820500/how-to-handle-large-amouts-of-data-in-tensorflow/47040165#47040165
def load_train_data(model_type, cache_path, file_no):
    # X：画像ファイル
    # paths：ファイルのパス
    # multi_class:
    X, paths = load_cache_data(cache_path, file_no)

    # Assigns a numpy array containing all labels for each image.
    # 全てのラベルが入っているnumpyのarrayを作る
    Y = labels.get_label_ids(paths, model_type)

    # X：画像のファイル数
    # Y：ラベル
    # paths：画像が保存されているパス
    return X, Y, paths


# Load single image based on path in local drive
def load_img(img_path, img_rows, img_cols):
    # 画像ファイルのロード
    img = cv2.imread(img_path)
    # Square Wrap
    img = square_wrap(img)
    img = cv2.resize(img, (img_rows, img_cols))
    return img


# Load single image based on path in local drive
def load_pred_img(img_path, img_rows, img_cols):
    # 画像ファイルのロード
    img = cv2.imread(img_path)
    # Square Wrap
    img = square_wrap(img)
    img = cv2.resize(img, (img_rows, img_cols))
    test_img = np.asarray(img)
    pred_img = np.reshape(a=test_img, newshape=(-1, img_rows, img_cols, 3))
    return [pred_img]


def prep_pred_img(img, img_rows, img_cols):
    # Square Wrap
    img = square_wrap(img)
    img = cv2.resize(img, (img_rows, img_cols))
    test_img = np.asarray(img)
    pred_img = np.reshape(a=test_img, newshape=(-1, img_rows, img_cols, 3))
    return [pred_img]


# 画像の復活する（ロードする）
def load_validation_img(model_type, valid_cache):
    img_data = []
    label_data = []
    paths_data = []
    file_no = 0

    while (True):
        X, Y, paths = load_train_data(model_type=model_type, cache_path=valid_cache, file_no=file_no)
        file_no += 1

        if len(X) == 0:
            break

        img_data.extend(X)
        label_data.extend(Y)
        paths_data.extend(paths)

    img_data = np.asarray(img_data)
    label_data = np.asarray(label_data)
    paths_data = np.asarray(paths_data)

    return img_data, label_data, paths_data


# CACHEから画像データを復活する
def load_cached_img(cache_path):
    img_data = []
    paths_data = []
    file_no = 0

    while (True):
        X, paths = load_cache_data(cache_path=cache_path, file_no=file_no)
        file_no += 1

        if len(X) == 0:
            break

        img_data.extend(X)
        paths_data.extend(paths)

    img_data = np.asarray(img_data)
    paths_data = np.asarray(paths_data)

    return img_data, paths_data


def square_wrap(img, rand=True):
    width = len(img[0])
    height = len(img)

    if width == height:
        return img

    size = max(width, height)
    if rand:
        new_img = np.round(255 * np.random.rand(size, size, 3))
        new_img = np.array(new_img, dtype=np.uint8)
    else:
        new_img = 128 * np.ones((size, size, 3), dtype=np.uint8)

    if width > height:
        delta = int(np.divide((size - height), 2))
        new_img[delta:delta + height, :] = img[:, :, 0:3]

    if height > width:
        delta = int(np.divide((size - width), 2))
        new_img[:, delta:delta + width] = img[:, :, 0:3]

    return new_img


def create_train_cache(path, img_rows, img_cols, train_fraction, enable_mods):
    train_cache = path + "_cache_train_" + str(img_cols)
    valid_cache = path + "_cache_valid_" + str(img_cols)

    if not os.path.exists(train_cache) or not os.path.exists(valid_cache):
        paths = get_files(path)
        # Generate possible array combinations
        paths = permutation(paths)
        max_train = int(math.ceil(float(train_fraction) * float(len(paths))))
        path_train = paths[:max_train]
        path_valid = paths[max_train:]
        load_resize_argument(path_train, img_rows, img_cols, train_cache, enable_mods=enable_mods)
        load_and_resize(path_valid, img_rows, img_cols, valid_cache)

    return train_cache, valid_cache


def create_img_cache(path, img_rows, img_cols):
    __cache_path = path + "_cache_img_" + str(img_cols)

    if not os.path.exists(__cache_path):
        paths = get_files(path)
        # Generate possible array combinations
        paths = permutation(paths)
        _path = paths[:]
        load_resize_argument(_path, img_rows, img_cols, __cache_path, enable_mods=False)

    return __cache_path


# Fetch image from url URL
def download_url(URL):
    resp = urllib.urlopen(URL)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# Fetch image from url URL
def download_img_to_path(URL, path, filename):
    r = requests.get(URL, allow_redirects=True)
    open(path + filename + ".jpg", 'wb').write(r.content)


def download_bulk_img(img_args):
    __tmp_path = "tmp/img_cache/"
    n = 0

    # Check if image data is available for training
    if len(img_args) < 1:
        return

    # Check if tmp for exist for current rank_pt index
    if not os.path.exists(__tmp_path):
        os.mkdir(__tmp_path)

    # TODO Check is tmp path is empty
    if not os.listdir(__tmp_path):
        print("Downloading bulk images...")
        print("Number of images: " + str(len(img_args)) + "\n")

    # Cycle through all images
    for i in img_args:
        download_img_to_path(URL=i, path=__tmp_path, filename=str("_tmp_img_" + str(n)))
        n = n + 1


if __name__ == "__main__":
    t, v = create_train_cache("C:\\Users\\Jwaltonp\\PycharmProjects\\AI\\VideoLog\\tmp\\landscape1\\", 224, 224, 1.0, False)
    print(str(t))

