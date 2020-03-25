import cv2
class Dataloader():
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []
            
            
    def load(self, image_paths, verbose=-1):
        data = []
        labels = []
        for (i, image_path) in enumerate(image_paths):
            image = 
        