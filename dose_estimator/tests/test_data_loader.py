from pathlib import Path
import cv2

from data_loader.data_loader import dataLoader


def test_data_loader():
    dl = dataLoader()
    assert 0 == 0

def test_load_folder():
    dl = dataLoader()
    folderName = "lungs"
    inputPath, outputPath = dl.load_folder(folderName)
    assert inputPath == Path(r"../../data/lungs/2d/images") and outputPath == Path(r"../..\data/lungs/2d/masks")

def test_read_files():
    dl = dataLoader()
    folderName = "lungs"
    inputPath, outputPath = dl.load_folder(folderName)

    inputImgs = dl.read_files(inputPath)
    outputImgs = dl.read_files(outputPath)
    assert inputImgs[0].parts[-1] == r"ID_0000_Z_0142.tif" and outputImgs[0].parts[-1] == r"ID_0000_Z_0142.tif"

def test_data_split():
    dl = dataLoader()
    folderName = "lungs"
    inputPath, outputPath = dl.load_folder(folderName)
    inputImgs = dl.read_files(inputPath)
    outputImgs = dl.read_files(outputPath)

    trainPerc = 0.8
    shfl = True
    trainIn, trainOut, testIn, testOut = dl.data_split(inputImgs, outputImgs, trainPerc, shfl)
    assert len(trainIn) == 214 and len(testIn) == 53 and (trainIn[0].parts[-1] != r"ID_0000_Z_0142.tif" or (trainIn[1].parts[-1] != r"ID_0001_Z_0146.tif"))

def test_load_images():
    dl = dataLoader()
    folderName = "lungs"
    inputPath, outputPath = dl.load_folder(folderName)
    inputImgs = dl.read_files(inputPath)
    outputImgs = dl.read_files(outputPath)
    trainPerc = 0.8
    shfl = True
    trainIn, trainOut, testIn, testOut = dl.data_split(inputImgs, outputImgs, trainPerc, shfl)

    trainIn_imgs = dl.load_images(trainIn)
    assert trainIn_imgs[0].shape[0] == 64
