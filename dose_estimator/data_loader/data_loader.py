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

    def data_split(self, input, output, trainPerc, shfl, testset):
        l = len(input)
        if l != len(output):
            raise TypeError("Input and output image lists are not the same size!")
        if shfl:
            shuffle(input)
            shuffle(output)
        if testset:
            split1 = round(trainPerc * l)
            split2 = round((trainPerc+0.1) * l)
            trainIn = input[:split1]
            trainOut = output[:split1]
            valIn = input[split1:split2]
            valOut = output[split1:split2]
            testIn = input[split2:]
            testOut = output[split2:]
            return trainIn, trainOut, valIn, valOut, testIn, testOut
        else:
            split = round(trainPerc * l)
            trainIn = input[:split]
            trainOut = output[:split]
            valIn = input[split:]
            valOut = output[split:]
            return trainIn, trainOut, valIn, valOut

    def load_images(self, imgs):
        img_list = []
        for img in imgs:
            img_list.append(cv2.resize(cv2.imread(str(img)), (64, 64)))
        return img_list

