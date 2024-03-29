class Config:
    def __init__(self):
        self.WIDTH = 256
        self.HEIGHT = 256
        self.OVERLAP = 128

        self.FILE_NAME = r"256_128"
        self.ORI_DIR = r"/data/dataset/update/train"
        self.CLIP_DIR = r"/data/data/update/"
        self.RES_DIR = r"/data/data/update/"
        
        # base project dir
        self.CLIP_BASEDIR = rf"{self.CLIP_DIR}/{self.FILE_NAME}"
        self.RES_BASEDIR = rf"{self.RES_DIR}/{self.FILE_NAME}"