import os
from pathlib import Path
import cv2


def add_end_slash(path):
    if path[-1] is not '/':
        return path + '/'
    return path


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_file_name(path):
    base_dir = os.path.dirname(path)
    file_name, ext = os.path.splitext(os.path.basename(path))
    ext = ext.replace(".", "")
    return (base_dir, file_name, ext)
    

def showimg(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (700,700))
	cv2.imshow('img', img)
	cv2.waitKey()

