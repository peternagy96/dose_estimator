from pathlib import Path
from random import shuffle
import cv2

class dataLoader:

    def __init__(self):
        self.input = 0
        self.output = 0

    def load_folder(self, folderName):
        p = Path("../../data/{}/2d".format(folderName))
        inputPath, outputPath = sorted([x for x in p.iterdir()])
        return inputPath, outputPath

    def read_files(self, path):
        imgs = sorted([x for x in path.iterdir()])
        return imgs

    def data_split(self, input, output, trainPerc, shfl):
        l = len(input)
        if l != len(output):
            raise TypeError("Input and output image lists are not the same size!")
        if shfl:
            shuffle(input)
            shuffle(output)
        split = round(trainPerc * l)
        trainIn = input[:split]
        trainOut = output[:split]
        testIn = input[split:]
        testOut = output[split:]
        return trainIn, trainOut, testIn, testOut

    def load_images(self, imgs):
        img_list = []
        for img in imgs:
            img_list.append(cv2.resize(cv2.imread(str(img)), (64, 64)))
        return img_list

