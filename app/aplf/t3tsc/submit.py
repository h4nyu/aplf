# type: ignore
"""
requirements:
    python: 3.6.3
    library:
        numpy: 1.16.3
        opencv-python: 4.1.0
"""
import numpy as np
import os
import json
import argparse
import time
from skimage import io

def make_json(annotations_dir, categories):
    count = 0
    annotation_files = os.listdir(annotations_dir)[:1]
    json_data = {}
    s = time.clock()
    for annotation_file in annotation_files:
        print(annotation_file)
        name = annotation_file.split('.')[0]
        json_data[name] = {}
        img = io.imread(os.path.join(annotations_dir, annotation_file))
        print(img)
        for category in categories:
            category_segments = {}
            x, y = np.where(img==categories[category])
            category_pix = {}
            for i, j in zip(x, y):
                if i not in category_pix:
                    category_pix[i] = []
                category_pix[i].append(j)
            for l in category_pix:
                segments = []
                num_segments = 0
                for i,v in enumerate(sorted(category_pix[l])):
                    if i == 0:
                        start = v
                        end = v
                    else:
                        if v == end + 1:
                            end = v
                        else:
                            segments.append([int(start), int(end)])
                            start = v
                            end = v
                            num_segments += 1
                segments.append([int(start), int(end)])
                category_segments[int(l)]=segments
            if len(category_pix):
                json_data[name][category]=category_segments
        count+=1
        print(count, time.clock()-s)

    return json_data
