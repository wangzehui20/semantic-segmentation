class Config:
    def __init__(self):
        self.WIDTH = 256
        self.HEIGHT = 256
        self.OVERLAP = 128
        self.IMAGE_STEP = 100000

        self.FILE_NAME = r"256_128"
        self.ORI_DIR = r"/data/dataset/change_detection/origin_merge"
        self.CLIP_DIR = r"/data/dataset/change_detection/merge"
        self.RES_DIR = r"/data/dataset/change_detection/merge"
        
        # base project dir
        self.CLIP_BASEDIR = rf"{self.CLIP_DIR}/{self.FILE_NAME}"
        self.RES_BASEDIR = rf"{self.RES_DIR}/{self.FILE_NAME}"