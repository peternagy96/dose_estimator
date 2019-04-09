from pathlib import Path

from data_loader.data_loader import dataLoader


def test_data_loader():
    dl = dataLoader()
    assert 0 == 0

def test_load_folder():
    dl = dataLoader()
    folderName = "lungs"
    inputPath, outputPath = dl.load_folder(folderName)
    assert inputPath == Path(r"../../data/lungs/2d/images") and outputPath == Path(r"../..\data/lungs/2d/masks")
