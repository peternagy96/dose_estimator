from pathlib import Path

class dataLoader:

    def __init__(self):
        self.input = 0
        self.output = 0

    def load_folder(self, folderName):
        p = Path("../../data/{}/2d".format(folderName))
        inputPath, outputPath = sorted([x for x in p.iterdir() if x.is_dir()])
        return inputPath, outputPath
