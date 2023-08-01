import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svgstr2paths, Document, paths2Drawing
import cairosvg
import pickle
import os
from tqdm import tqdm

with open('./data_text.pickle', mode='rb') as f:
    data = pickle.load(f)


data_image_list = []
path = 'C:/Users/admin/Documents/Python/unsplash/'

for filename in os.listdir('../../unsplash/animals/'):
    data_image_list.append(path + "animals/" + filename)
for filename in os.listdir('../../unsplash/architecture/'):
    data_image_list.append(path + "architecture/" + filename)
for filename in os.listdir('../../unsplash/film/'):
    data_image_list.append(path + "film/" + filename)
for filename in os.listdir('../../unsplash/nature/'):
    data_image_list.append(path + "nature/" + filename)
for filename in os.listdir('../../unsplash/street/'):
    data_image_list.append(path + "street/" + filename)

for iter in tqdm(range(len(data))):
    try:
        # Modes of watermark such as grid, small, big
        MODE = random.randint(0, 2)
        # watermark scale
        SCALE = 1  # 0.45
        #watermark rotation angle
        ANGLE = 90
        # n-of rows and columns in `grid mode`
        ROWS = 4
        COLS = 4
        # watermark opacity
        OPACITY = 0.5  # 0.15 - 0.85
        # watermark colors
        COLORS = ["#fff", "#fff", "#fff", "#fff", "#fff", "#fff",
                  "#0ff", "#f0f", "#ff0", "#00f", "#f00", "#0f0"]
        bboxes = []
        all_segments = []

        # rows and columns to be randomly selected
        MODES = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3),
                 (3, 1), (2, 3), (3, 2), (3, 3)]

        ANGLE = random.randint(-90, 90)
        OPACITY = random.randint(60, 90) / 100

        if MODE == 2:
            watermarksvg = data[iter]['big']
            ROWS, COLS = (1, 1)
        else:
            ROWS, COLS = MODES[random.randint(0, 8)]
            watermarksvg = data[iter]['small']

        paths, attributes, svg_attributes = svgstr2paths(
            watermarksvg, return_svg_attributes=True)
        SVG_WIDTH = int(math.ceil(float(svg_attributes['width'])) // (1/SCALE))
        SVG_HEIGHT = int(
            math.ceil(float(svg_attributes['height'])) // (1/SCALE))

        SAMPLE_IMAGE_PATH = data_image_list[random.randint(
            0, len(data_image_list) - 1)]
        background = cv2.imread(SAMPLE_IMAGE_PATH)

        if ANGLE < 0:
            ANGLE = - ANGLE
        elif ANGLE > 90:
            ANGLE = 180 - ANGLE

        for attr in attributes:
            attr['transform'] = f'translate({SVG_HEIGHT/2},{SVG_WIDTH/2}) rotate({ANGLE},{SVG_WIDTH/2},{SVG_HEIGHT/2}) scale({SCALE})'

        svg_attributes['width'] = SVG_WIDTH + SVG_HEIGHT
        svg_attributes['height'] = SVG_WIDTH + SVG_HEIGHT
        svg_attributes['viewBox'] = f'0 0 {SVG_WIDTH + SVG_HEIGHT} {SVG_WIDTH + SVG_HEIGHT}'

        # if iter % 2 == 0:

        # select color by the image average color
        avg_color = background.mean(axis=0).mean(axis=0).astype('int8')
        svg_attributes['fill'] = '#%02x%02x%02x' % (
            255 - avg_color[0], 255 - avg_color[1], 255 - avg_color[2])
        # else:
        #     svg_attributes['fill'] = COLORS[random.randint(0,11)]

        dr = paths2Drawing(paths, attributes=attributes,
                           svg_attributes=svg_attributes)

        cairosvg.svg2png(bytestring=dr.tostring(), write_to='temp.png',
                         output_height=SVG_WIDTH + SVG_HEIGHT, output_width=SVG_WIDTH + SVG_HEIGHT)

        rect_width = math.ceil((math.cos(math.radians(
            ANGLE)) * SVG_WIDTH) + (math.cos(math.radians(90-ANGLE)) * SVG_HEIGHT)) + 5
        rect_height = math.ceil((math.sin(math.radians(
            ANGLE)) * SVG_WIDTH) + (math.sin(math.radians(90-ANGLE)) * SVG_HEIGHT)) + 5
        viewpoint_width = SVG_WIDTH + SVG_HEIGHT
        x1 = (viewpoint_width//2) - (rect_width//2)
        y1 = (viewpoint_width//2) - (rect_height//2)
        x2 = (x1 + rect_width)
        y2 = (y1 + rect_height)

        dots = []
        ncols = []
        nradii = []
        segments = []
        for path in paths:
            # print(path)
            path = path.scaled(SCALE, SCALE)
            path = path.rotated(ANGLE, (SVG_WIDTH/2) + 1j*(SVG_HEIGHT/2))
            path = path.translated((SVG_HEIGHT/2) + 1j*(SVG_WIDTH/2))
            temp = []
            for seg in path:
                ln = int(seg.length())
                for i in range(0, ln, 4):
                    # print(seg.point(i/(ln)))
                    temp.append(seg.point(i/(ln)))

                    dots += [seg.point(i/(ln))]
                    ncols += ['red']
                    nradii += [0.5]

            segments.append(temp.copy())
            # break


        img = np.zeros((viewpoint_width, viewpoint_width))
        mask_bg = np.zeros((background.shape[0], background.shape[1]))
        watermark = cv2.imread('./temp.png', cv2.IMREAD_UNCHANGED)

        # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        cut_x = background.shape[0]//ROWS
        cut_y = background.shape[1]//COLS

        random_offset_x = random.randint(0, cut_x-watermark.shape[0] - 1)
        random_offset_y = random.randint(0, cut_y-watermark.shape[1] - 1)

        for cut_y_index in range(COLS):
            for cut_x_index in range(ROWS):

                if MODE == 0:
                    offset_x = random_offset_x + (cut_x_index) * cut_x
                    offset_y = random_offset_y + (cut_y_index) * cut_y
                else:
                    offset_x = random.randint(
                        cut_x_index*cut_x, (cut_x_index+1)*cut_x-watermark.shape[1] - 1)
                    offset_y = random.randint(
                        cut_y_index*cut_y, (cut_y_index+1)*cut_y-watermark.shape[1] - 1)

                # # Show COL ROW RECATNGLE verbose
                # background = cv2.rectangle(background, (cut_y_index*cut_y, cut_x_index*cut_x), ( (cut_y_index+1)*cut_y, (cut_x_index+1)*cut_x), (255, 0, 0), 2)
                # offset_y = background.shape[1] - 100

                for seg in segments:
                    segments_temp = []
                    for dot in seg:
                        img[int(dot.imag)][int(dot.real)] = 1

                        # # Show SEGMENTATION verbose
                        # background = cv2.circle(background, ((int(dot.real) + offset_y),(int(dot.imag) + offset_x)),0 ,(0, 255, 0),1)

                        segment_x = int(dot.real) + offset_y
                        segment_y = int(dot.imag) + offset_x
                        segment_x = segment_x / background.shape[1]
                        segment_y = segment_y / background.shape[0]
                        segment_x = "{:.2f}".format(segment_x)
                        segment_y = "{:.2f}".format(segment_y)
                        segments_temp.append((segment_x, segment_y))
                    all_segments.append(segments_temp)

                for channel in range(3):
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            i2 = i + offset_x
                            j2 = j + offset_y
                            if i2 < background.shape[0] and j2 < background.shape[1]:
                                # if background[i2, j2, channel] < (img[i, j]*255):
                                #     background[i2, j2, channel] = (img[i, j]*255)

                                # if background[i2, j2, channel] < (watermark[i, j,channel]):
                                if (watermark[i, j, channel]) > 0:
                                    background[i2, j2, channel] = (
                                        (OPACITY * (watermark[i, j, channel]/255.)) + ((1-OPACITY) * (background[i2, j2, channel]/255.))) * 255
                                    background[i2, j2, channel] = watermark[i, j, channel]
                                    mask_bg[i2, j2] = 255

                # # Show BOUNDING BOX verbose
                # background = cv2.rectangle(background,(offset_y + x2,offset_x + y2),(offset_y + x1,offset_x + y1),(255,0,0),0)
                # background = cv2.circle(background, (offset_y + x1+rect_width//2, offset_x + y1), radius=2, color=(0, 0, 255), thickness=5)
                # background = cv2.circle(background, (offset_y + x1,offset_x + y1+rect_height//2), radius=2, color=(0, 0, 255), thickness=5)
                # background = cv2.circle(background, (offset_y + x1 +rect_width//2 ,offset_x + y1+rect_height//2), radius=5, color=(0, 0, 255), thickness=5)

                center_x = offset_y + x1 + rect_width//2
                center_y = offset_x + y1+rect_height//2
                center_x = "{:.2f}".format(center_x / background.shape[1])
                center_y = "{:.2f}".format(center_y / background.shape[0])
                normalized_rect_width = "{:.2f}".format(
                    rect_width / background.shape[1])
                normalized_rect_height = "{:.2f}".format(
                    rect_height / background.shape[0])
                bboxes.append(
                    (center_x, center_y, normalized_rect_width, normalized_rect_height))


        # Save IMAGE
        cv2.imwrite(f'./dataset2/image/{iter}.png', background)

        # Save MASK
        cv2.imwrite(f'./dataset2/mask/{iter}.png', mask_bg)

        # Save SEGMENTATION
        with open(f'./dataset2/segbox/{iter}.txt', mode='w') as f:
            segs = []
            for item in all_segments:
                sexbox_text = "0 "
                for seg in item:
                    sexbox_text += " " + " ".join(list(seg))
                segs.append(sexbox_text)
            f.write("\n".join(segs))

        # Save BBOX
        with open(f'./dataset2/bbox/{iter}.txt', mode='w') as f:
            bb = []
            for item in bboxes:
                bbox = f'0 {item[0]} {item[1]} {item[2]} {item[3]}'
                bb.append(bbox)
            f.write("\n".join(bb))
    except KeyboardInterrupt as e:
        exit()
    except Exception as e:
        print(e)
