import json

import h5py
import numpy as np

source_metadata_list = json.load(open('train_sceneGraphs.json'))
target_metadata_list = [] 
all_image_names = list(source_metadata_list.keys())
i = 0
M = len(all_image_names)
for im in range(0,M):
    image_name = all_image_names[im]
    str_name = str(image_name).split('.')[0]
    target_metadata = {}
    target_metadata['image_name'] = str_name
    target_metadata['height'] = source_metadata_list[image_name]['height']
    target_metadata['width'] = source_metadata_list[image_name]['width']
    target_metadata['image_id'] = i
    target_metadata['org_status'] = 'train'
    #target_metadata['no_relations'] = len(relation_info)
    target_metadata_list.append(target_metadata)
    i += 1

source_metadata_list = json.load(open('val_sceneGraphs.json'))
all_image_names = list(source_metadata_list.keys())
M = len(all_image_names)
for im in range(0,M):
    image_name = all_image_names[im]
    str_name = str(image_name).split('.')[0]
    target_metadata = {}
    target_metadata['image_name'] = str_name
    target_metadata['height'] = source_metadata_list[image_name]['height']
    target_metadata['width'] = source_metadata_list[image_name]['width']
    target_metadata['image_id'] = i  
    target_metadata['org_status'] = 'test'
    #target_metadata['no_relations'] = len(relation_info)
    target_metadata_list.append(target_metadata)
    i += 1 
with open('image_data.json', 'w') as outfile:
    json.dump(target_metadata_list,outfile)

