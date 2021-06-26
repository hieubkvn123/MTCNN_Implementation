import os
import glob
import json
import xmltodict
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse_pascal_voc(directory, img_dir='images', annot_dir='annotations'):
    annot_dir = os.path.join(directory, annot_dir)
    img_dir = os.path.join(directory, img_dir)
    xml_files = glob.glob(os.path.join(annot_dir, '*.xml'))

    img_paths = []
    bboxes = []
    labels = []

    for file in xml_files:
        with open(file, 'r') as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            annot = data_dict['annotation']

            filename = annot['filename']
            objects = annot['object']
            W = float(annot['size']['width'])
            H = float(annot['size']['height'])
            class_id = 1 

            if(isinstance(objects, list)):
                for obj in objects:
                    bbox = obj['bndbox']
                    left  = float(bbox['xmin']) / W
                    top   = float(bbox['ymin']) / H
                    right = float(bbox['xmax']) / W
                    bottom= float(bbox['ymax']) / H
                    class_id = obj['name']

                    img_paths.append(os.path.join(img_dir, filename))
                    bboxes.append([left, top, right-left, bottom-top])
                    labels.append(class_id)

            elif(isinstance(objects, dict)):
                obj = objects
                bbox = objects['bndbox']
                left  = float(bbox['xmin']) / W
                top   = float(bbox['ymin']) / H
                right = float(bbox['xmax']) / W
                bottom= float(bbox['ymax']) / H
                class_id = obj['name']

                img_paths.append(os.path.join(img_dir, filename))
                bboxes.append([left, top, right-left, bottom-top])
                labels.append(class_id)


    labels = LabelEncoder().fit_transform(labels)
    return np.array(img_paths), np.array(bboxes), np.array(labels)

def parse_darknet(directory):
    img_extensions = ['png', 'jpg', 'jpeg', 'ppm']
    img_files = []
    annot_files = []
    file_ext = None

    img_paths = []
    labels = []
    bboxes = []

    for ext in img_extensions:
        img_files += glob.glob(os.path.join(directory, f'*.{ext}'))
        if(len(glob.glob(os.path.join(directory, f'*.{ext}'))) > 0):
            file_ext = ext

    annot_files = glob.glob(os.path.join(directory, '*.txt'))

    if(len(img_files) != len(annot_files)):
        raise Exception('Number of image files and annotation files mismatch')

    for annot in annot_files:
        lines = open(annot, 'r').readlines()
        for line in lines:
            details = line.strip().split(' ')

            class_id, x, y, w, h = details
            class_id = int(class_id)
            x, y, w, h = float(x), float(y), float(w), float(h)

            img_path = f"{annot.split('.')[0]}.{file_ext}"

            img_paths.append(img_path)
            bboxes.append([x, y, w, h])
            labels.append(class_id)

    return np.array(img_paths), np.array(bboxes), np.array(labels)

def pascal_voc_to_darknet(directory, img_dir='images', annot_dir='annotations'):
    annot_dir = os.path.join(directory, annot_dir)
    img_dir = os.path.join(directory, img_dir)
    xml_files = glob.glob(os.path.join(annot_dir, '*.xml'))
    classes = {}

    for file in xml_files:
        with open(file, 'r') as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            annot = data_dict['annotation']

            filename = annot['filename']
            out_file = open(os.path.join(img_dir, f'{filename.split(".")[0]}.txt'), 'w')
            objects = annot['object']
            W = float(annot['size']['width'])
            H = float(annot['size']['height'])
            class_id = 1 

            if(isinstance(objects, list)):
                for obj in objects:
                    bbox = obj['bndbox']
                    left  = float(bbox['xmin']) / W
                    top   = float(bbox['ymin']) / H
                    right = float(bbox['xmax']) / W
                    bottom= float(bbox['ymax']) / H
                    class_id = obj['name']

                    if(classes == {}):
                        classes[class_id] = 0
                        class_id = classes[class_id]
                    
                    if(class_id in classes):
                        class_id = classes[class_id]
                    else:
                        latest_class = classes[max(classes, key=classes.get)]
                        classes[class_id] = latest_class + 1
                        class_id = classes[class_id]

                    line = f'{class_id} {left:.6f} {top:.6f} {right-left:.6f} {bottom-top:.6f}\n'
                    out_file.write(line)

            elif(isinstance(objects, dict)):
                obj = objects
                bbox = objects['bndbox']
                left  = float(bbox['xmin']) / W
                top   = float(bbox['ymin']) / H
                right = float(bbox['xmax']) / W
                bottom= float(bbox['ymax']) / H
                class_id = obj['name']

                if(classes == {}):
                    classes[class_id] = 0
                    class_id = classes[class_id]
                
                if(class_id in classes):
                    class_id = classes[class_id]
                else:
                    latest_class = classes[max(classes, key=classes.get)]
                    classes[class_id] = latest_class + 1
                    class_id = classes[class_id]

                line = f'{class_id} {left:.6f} {top:.6f} {right-left:.6f} {bottom-top:.6f}\n'
                out_file.write(line)
        out_file.close()