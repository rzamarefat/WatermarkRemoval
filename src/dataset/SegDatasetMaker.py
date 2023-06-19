import cv2
import numpy as np
from config import config
import random
from glob import glob
import os

class SegDatasetMaker:
    def __init__(self):
        self._chars = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYX1234567890!@#$%^&*()_+="]
        self._max_text_length = 15
        self._aug_methods = ["rotate", "void", "sharpness"]
        self._watermark_methods = ["tile", "single"]

        self._root_to_read_original_images = config["root_to_read_original_images"]

        if not(os.path.isdir(self._root_to_read_original_images)):
            raise RuntimeError("Verify a valid dir to background images")

        self._bg_images_p = [f for f in sorted(glob(os.path.join(self._root_to_read_original_images, "*")))]

        if len(self._bg_images_p) == 0:
            raise RuntimeError("There is no images in the specified dir for backgroung images. Check the config file ('root_to_read_original_images') and set a valid path with at least 1 image inside.")        


    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _generate_text(self, length_of_text):
        
        text = "".join(random.sample(self._chars, length_of_text))

        return text 

    def _select_random_bg(self):
        image = cv2.imread(random.sample(self._bg_images_p, 1)[0])
        return image

    def _make_mask_image(self, bg_image, watermark_method, texts):
        black_image = np.zeros_like(bg_image)


        if watermark_method == "tile":
            mask_images = []
            for text in texts:
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                maks_image = cv2.putText(black_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                mask_images.append(mask_images)

            return mask_images

        elif watermark_method == "single":
            return None 


    def _select_log(self):
        logo = None
        return logo

    def make(self):
        rand_watermark_method = random.sample([self._watermark_methods], 1)[0]
        rand_text_length = random.sample([f for f in range(self._max_text_length)], 1)[0]
        rand_text = self._generate_text(rand_text_length)
        rand_bg = self._select_random_bg()
        
        if config["do_augmentation_on_bgs"]:
            toss_for_aug_method = random.sample([i for i in range(len(self._aug_methods))], 1)[0]


        self._make_mask_image(rand_bg, rand_watermark_method, rand_text)
            
        


if __name__ == "__main__":
    sd = SegDatasetMaker()

    sd.make()




    