class Config:
    def __init__(self):
        self.WIDTH = 512
        self.HEIGHT = 512
        self.OVERLAP = 128
        self.IMAGE_STEP = 100000

        self.FILE_NAME = r"512_128"
        self.ORI_DIR = r"/data/dataset/semi_compete/origin_integrate"
        self.EXP_NAME = r"labeled_train"
        self.CLIP_DIR = r"/data/data/semi_compete/clip_integrate"
        self.RES_DIR = r"/data/data/semi_compete"
        
        # base project dir
        self.CLIP_BASEDIR = rf"{self.CLIP_DIR}/{self.EXP_NAME}/{self.FILE_NAME}"
        self.RES_BASEDIR = rf"{self.RES_DIR}/{self.EXP_NAME}/{self.FILE_NAME}"