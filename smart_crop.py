import click
import numpy as np
import math
import json
import glob
import os

from collections import namedtuple
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from PIL import Image, ImageDraw

Rectangle = namedtuple('Rectangle', 'x y w h')

# метод для получения ограничивающей рамки для множества прямоугольников
def get_bbox(*rects):
    xmin = 1e9
    ymin = 1e9
    xmax = -1e9
    ymax = -1e9
    for rect in rects:
        xmin = min(xmin, rect[0], rect[0]+rect[2])
        ymin = min(ymin, rect[1], rect[1]+rect[3])
        xmax = max(xmax, rect[0], rect[0]+rect[2])
        ymax = max(ymax, rect[1], rect[1]+rect[3])
        
    return Rectangle(xmin, ymin, xmax-xmin, ymax-ymin)

# проверка, пересекаются ли рамки
def bboxes_intersects(a, b):
    return abs((a.x + a.w/2) - (b.x + b.w/2)) * 2 < (a.w + b.w) and \
           abs((a.y + a.h/2) - (b.y + b.h/2)) * 2 < (a.h + b.h)

# специальная метрика для расчета "расстояния" между прямоугольниками
def bounded_rect_distance(a, b, max_bbox_size):
    bbox = get_bbox(a, b)
    if bbox.w > max_bbox_size or bbox.h > max_bbox_size:
        return 10e9
    return max(bbox[2]-a[2], bbox[2]-b[2], bbox[3]-a[3], bbox[3]-b[3])

def cluster_rectangles(rects, max_bbox_size):
    rect_dist = lambda a, b: bounded_rect_distance(a, b, max_bbox_size)
    linked = linkage(rects, 'complete', metric=rect_dist)
    clusters = fcluster(linked, 10e8, criterion="distance") - 1

    # Разбиение всех прямоугольников на отдельные массивы
    rect_clusters = []
    for i in range(clusters.max() + 1):
        mask = clusters == i
        _rects = [r for j, r in enumerate(rects) if mask[j]]
        rect_clusters.append(_rects)

    return rect_clusters

def get_filename(path):
    _, tail = os.path.split(path)
    return ".".join(tail.split(".")[:-1])

def markup_to_rect(markup, w, h):
    x = (markup["xc"] - markup["w"] / 2) * w
    y = (markup["yc"] - markup["h"] / 2) * h
    w = markup["w"] * w
    h = markup["h"] * h
    return Rectangle(x, y, w, h)

def rect_to_yolo(rect, w, h):
    xc = (rect.x + rect.w / 2) / w
    yc = (rect.y + rect.h / 2) / h
    w = rect.w / w
    h = rect.w / h
    return (xc, yc, w, h)   

def get_bbox_with_margin(b, margin):
    return Rectangle(b.x - margin, b.y - margin, b.w + margin * 2, b.h + margin * 2)

@click.command()
@click.option("--dataset-path", "-d", help="Path to dataset folder")
@click.option("--output-path", "-o", help="Path to output folder")
@click.option("--max-frag-size", "-s", help="Max size of output image fragment", type=int)
@click.option("--padding", "-p", help="Space in pixels between bounding box of objectas and borders of the fragment", type=int)
def process_data(dataset_path, output_path, max_frag_size, padding):
    for markup_path in glob.glob(f"{dataset_path}/*.txt"):
        markup = []
        filename = get_filename(markup_path)

        with open(markup_path, "r") as file:
            for line in file:
                line_items = line.split(" ")
                markup.append({
                    "label": int(line_items[0]),
                    "xc": float(line_items[1]),
                    "yc": float(line_items[2]),
                    "w": float(line_items[3]),
                    "h": float(line_items[4])
                })
            
        files_with_same_name = glob.glob(f"{dataset_path}/{filename}.*")
        image_path = list(filter(lambda s: s != markup_path, files_with_same_name))[0]

        image = np.array(Image.open(image_path), dtype=np.uint8)
        img_h, img_w, _ = image.shape

        max_bbox_size = max_frag_size - 2 * padding

        rects = [markup_to_rect(m, img_w, img_h) for m in markup]

        rect_clusters = cluster_rectangles(rects, max_bbox_size)
        bboxes = [get_bbox(*rc) for rc in rect_clusters]
        bboxes = [get_bbox_with_margin(bbox, padding) for bbox in bboxes]

        for i, bbox in enumerate(bboxes):
            frag = image[int(bbox.y):int(bbox.y+bbox.h), int(bbox.x):int(bbox.x+bbox.w)]

            file_lines = []
            pil_frag = Image.fromarray(frag)
            draw = ImageDraw.Draw(pil_frag)

            for m in markup:
                rect = markup_to_rect(m, img_w, img_h)

                if not bboxes_intersects(rect, bbox):
                    continue

                rect = Rectangle(rect.x - bbox.x, rect.y - bbox.y, rect.w, rect.h)
                draw.rectangle([rect.x, rect.y, rect.x + rect.w, rect.y + rect.h], outline="red")
                yt = rect_to_yolo(rect, frag.shape[1], frag.shape[0])
                file_lines.append(f"{m['label']} {yt[0]} {yt[1]} {yt[2]} {yt[3]}")

            file_content = "\n".join(file_lines)
            with open(f"{output_path}/{filename}__{i}.txt", "w") as file:
                file.write(file_content)
            
            pil_frag.save(f"{output_path}/{filename}__{i}.jpg")
        
if __name__ == "__main__":
    process_data()
