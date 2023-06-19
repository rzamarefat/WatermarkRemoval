import cv2
import numpy as np
from config import config
import random
from glob import glob
from tqdm import tqdm
import os
from uuid import uuid1


class SegDatasetMaker:
    def __init__(self):
        self._chars = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYX1234567890!@#$%^&*()_+="]
        self._max_text_length = 15
        self._aug_methods = ["rotate", "void", "sharpness"]
        self._watermark_methods = ["tile", "single"]
        self.save_mask_images = config["save_mask_images"]

        if self.save_mask_images:
            if not(os.path.isdir(config["root_to_save_watermarked_images_gt"])):
                raise RuntimeError("The path you provided for saving watermarked images is not valid.")


        self._fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        ]

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

    def _select_log(self):
        logo = None
        return logo

    def _generate_rand_font(self):
        return random.sample(self._fonts, 1)[0]

    def _melt_mask_to_edges(self, mask_image):
        """
        ref: https://medium.com/@rootaccess/how-to-detect-edges-of-a-mask-in-python-with-opencv-4bcdb3049682
        """
        
        mask_image = mask_image > 128

        mask_image = np.asarray(mask_image[:, :, 0], dtype=np.double)
        gx, gy = np.gradient(mask_image)

        temp_edge = gy * gy + gx * gx
        temp_edge
        temp_edge[temp_edge != 0.0] = 255.0
        temp_edge = np.asarray(temp_edge, dtype=np.uint8)

        return temp_edge


    def _save_seg_bounding_yolo_format(mask_image, abs_path_to_save_seg_bounding):
        pass



    def _make_single(self):
        rand_text_length = random.sample([f for f in range(self._max_text_length)], 1)[0]
        rand_text = self._generate_text(rand_text_length)
        rand_bg = self._select_random_bg()
        
        if config["do_augmentation_on_bgs"]:
            toss_for_aug_method = random.sample([i for i in range(len(self._aug_methods))], 1)[0]

        black_image = np.zeros_like(rand_bg)

        font = self._generate_rand_font()
        padding = 20
        org = (random.randint(1, int(rand_bg.shape[0])) + padding,  random.randint(1, int(rand_bg.shape[1])) - padding)
        fontScale = random.sample([1, 2, 3, 4, 5], 1)[0]
        color = (255, 0, 0)
        color_for_mask = (255, 255, 255)
        thickness = 2
        mask_image = cv2.putText(black_image, rand_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        mask_image_for_save = cv2.putText(black_image, rand_text, org, font, fontScale, color_for_mask, thickness, cv2.LINE_AA)
        mask_image_for_save = np.where(mask_image_for_save > 127.5, 255, 0)
        roi = rand_bg
        result = cv2.addWeighted(roi, 1, mask_image, 0.6, 0)


        random_name = f"single__{str(uuid1())[0:8]}.jpg"
        
        if self.save_mask_images:
            try:
                abs_path_watermarked_gt = os.path.join(config["root_to_save_watermarked_images_gt"], random_name)
                cv2.imwrite(abs_path_watermarked_gt, mask_image_for_save)
            except Exception as e:
                print("Sth went wrong in saving watermarked 'mask' images.The full error")
                print(e)
                exit()

            try:    
                abs_path_watermarked_img = os.path.join(config["root_to_save_watermarked_images"], random_name)
                cv2.imwrite(abs_path_watermarked_img, result)
            except Exception as e:
                print("Sth went wrong in saving watermarked images.The full error")
                print(e)
                exit()
        

        abs_path_to_save_seg_bounding = os.path.join(config["root_to_save_yolo_format_seg_boudning"], random_name)
        self._save_seg_bounding_yolo_format(mask_image, abs_path_to_save_seg_bounding)
            




        
        


    def _make_tile(self):
        pass

    def make(self):
        
        print("Making data")
        for i in range(config["num_required_images"]):
            self._make_single()
            self._make_tile()
            
        


if __name__ == "__main__":
    sd = SegDatasetMaker()

    sd.make()




    