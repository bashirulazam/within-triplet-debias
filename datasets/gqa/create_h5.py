import json
import numpy as np
import h5py as h5
from utils import encode_box
train_file = open('train_sceneGraphs.json')
train_data = json.load(train_file)
dict_file = open('GQA-SGG-dicts.json')
gqa_dict = json.load(dict_file)
#obj_dict = list(dict['label_to_idx'].keys())
obj_count = {}
rel_count = {}
image_ids = list(train_data.keys())
labels = []
predicates = []
relationships = np.empty(2,dtype=int) 
img_to_first_box = []
img_to_last_box = []
img_to_first_rel = []
img_to_last_rel = []
active_object_mask = []
split = []
boxes_1024 = np.empty(4,dtype=int)
boxes_512 = np.empty(4,dtype=int)
attributes = np.empty(10,dtype=int)
M = len(image_ids)
k = 0
r = 0
#M = 100
for im in range(0,M):
    print(im)
    split = np.append(split,0)
    obj_ids = list(train_data[image_ids[im]]['objects'].keys())
    NO = len(obj_ids)
    img_to_first_box = np.append(img_to_first_box,k)
    org_width = train_data[image_ids[im]]['width']
    org_height = train_data[image_ids[im]]['height']
    for ob in range(0,NO):
        obj_name1 = train_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 in gqa_dict['label_to_idx'].keys():
            train_data[image_ids[im]]['objects'][obj_ids[ob]]['relative_id'] = k
            #print(obj_name1 + str(k))
            labels =  np.append(labels,gqa_dict['label_to_idx'][obj_name1])
            box_xmin = abs(train_data[image_ids[im]]['objects'][obj_ids[ob]]['x'])
            box_ymin = abs(train_data[image_ids[im]]['objects'][obj_ids[ob]]['y'])
            box_width = abs(train_data[image_ids[im]]['objects'][obj_ids[ob]]['w'])
            box_height = abs(train_data[image_ids[im]]['objects'][obj_ids[ob]]['h'])
            box_gqa = [box_xmin, box_ymin, box_width, box_height]
            encoded_box_1024 = encode_box(box_gqa, org_height, org_width, 1024,image_ids[im])
            encoded_box_512 = encode_box(box_gqa, org_height, org_width, 512,image_ids[im])
            boxes_1024 = np.vstack([boxes_1024,[encoded_box_1024]])
            boxes_512 = np.vstack([boxes_512,[encoded_box_512]])
            active_object_mask = np.append(active_object_mask,True)
            attr = train_data[image_ids[im]]['objects'][obj_ids[ob]]['attributes']
            attributes_row = np.zeros((10,),dtype=int)
            for at in range(0,len(attr)):
                if attr[at] in gqa_dict['attribute_to_idx']:
                    at_index = gqa_dict['attribute_to_idx'][attr[at]]
                    if at_index <= 10:
                        attributes_row[at_index-1] = 1
            attributes = np.vstack([attributes,[attributes_row]])
            k = k + 1 
    img_to_last_box = np.append(img_to_last_box,k-1)
    if img_to_last_box[-1] == img_to_first_box[-1] -1:
        img_to_first_box[-1] = -1
        img_to_last_box[-1] = -1
    #print('Img_to_first_box')
    #print(img_to_first_box[im])
    #print('Img_to_last_box')
    #print(img_to_last_box[im])

    img_to_first_rel = np.append(img_to_first_rel,r)
    for ob in range(0,NO):
        obj_name1 = train_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 in gqa_dict['label_to_idx'].keys():
            obj1_rel_all = train_data[image_ids[im]]['objects'][obj_ids[ob]]['relations']
            N_obj1_rel = len(obj1_rel_all)
            for re in range(0,N_obj1_rel):
                obj1_rel = obj1_rel_all[re]['name']
                obj2_id = obj1_rel_all[re]['object']
                obj_name2 = train_data[image_ids[im]]['objects'][obj2_id]['name']
                if obj1_rel in gqa_dict['predicate_to_idx'].keys() and obj_name2 in gqa_dict['label_to_idx'].keys():
                    id1 = train_data[image_ids[im]]['objects'][obj_ids[ob]]['relative_id']
                    id2 = train_data[image_ids[im]]['objects'][obj2_id]['relative_id']
                    predicates = np.append(predicates,gqa_dict['predicate_to_idx'][obj1_rel])
                    relationships = np.vstack([relationships,[id1,id2]])
                    #print(obj_name1 + ''+ obj1_rel + '' + obj_name2  + str(r))
                    r = r + 1
    img_to_last_rel = np.append(img_to_last_rel,r-1)
    if img_to_last_rel[-1] == img_to_first_rel[-1] -1:
        img_to_first_rel[-1] = -1
        img_to_last_rel[-1] = -1
    #print('Img_to_first_rel')
    #print(img_to_first_rel[im])
    #print('Img_to_last_rel')
    #print(img_to_last_rel[im])


#print(labels)
#print(predicates)
#print(relationships)
#print(boxes_1024)
#print(len(labels))
#print(len(predicates))
#print(attributes)

#Now extract from the validation set
val_file = open('val_sceneGraphs.json')
val_data = json.load(val_file)
dict_file = open('GQA-SGG-dicts.json')
gqa_dict = json.load(dict_file)
#obj_dict = list(dict['label_to_idx'].keys())
image_ids = list(val_data.keys())
M = len(image_ids)
#M = 100
for im in range(0,M):
    print(im)
    split = np.append(split,2)
    obj_ids = list(val_data[image_ids[im]]['objects'].keys())
    NO = len(obj_ids)
    img_to_first_box = np.append(img_to_first_box,k)
    org_width = val_data[image_ids[im]]['width']
    org_height = val_data[image_ids[im]]['height']
    for ob in range(0,NO):
        obj_name1 = val_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 in gqa_dict['label_to_idx'].keys():
            val_data[image_ids[im]]['objects'][obj_ids[ob]]['relative_id'] = k
            #print(obj_name1 + str(k))
            labels =  np.append(labels,gqa_dict['label_to_idx'][obj_name1])
            box_xmin = abs(val_data[image_ids[im]]['objects'][obj_ids[ob]]['x'])
            box_ymin = abs(val_data[image_ids[im]]['objects'][obj_ids[ob]]['y'])
            box_width = abs(val_data[image_ids[im]]['objects'][obj_ids[ob]]['w'])
            box_height = abs(val_data[image_ids[im]]['objects'][obj_ids[ob]]['h'])
            box_gqa = [box_xmin, box_ymin, box_width, box_height]
            encoded_box_1024 = encode_box(box_gqa, org_height, org_width, 1024,image_ids[im])
            encoded_box_512 = encode_box(box_gqa, org_height, org_width, 512,image_ids[im])
            boxes_1024 = np.vstack([boxes_1024,[encoded_box_1024]])
            boxes_512 = np.vstack([boxes_512,[encoded_box_512]])
            attr = val_data[image_ids[im]]['objects'][obj_ids[ob]]['attributes']
            active_object_mask = np.append(active_object_mask,True)
            attributes_row = np.zeros((10,),dtype=int)
            for at in range(0,len(attr)):
                if attr[at] in gqa_dict['attribute_to_idx']:
                    at_index = gqa_dict['attribute_to_idx'][attr[at]]
                    if at_index <= 10:
                        attributes_row[at_index-1] = 1
            attributes = np.vstack([attributes,[attributes_row]])
            k = k + 1 
    img_to_last_box = np.append(img_to_last_box,k-1)
    if img_to_last_box[-1] == img_to_first_box[-1] -1:
        img_to_first_box[-1] = -1
        img_to_last_box[-1] = -1

    img_to_first_rel = np.append(img_to_first_rel,r)
    for ob in range(0,NO):
        obj_name1 = val_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 in gqa_dict['label_to_idx'].keys():
            obj1_rel_all = val_data[image_ids[im]]['objects'][obj_ids[ob]]['relations']
            N_obj1_rel = len(obj1_rel_all)
            for re in range(0,N_obj1_rel):
                obj1_rel = obj1_rel_all[re]['name']
                obj2_id = obj1_rel_all[re]['object']
                obj_name2 = val_data[image_ids[im]]['objects'][obj2_id]['name']
                if obj1_rel in gqa_dict['predicate_to_idx'].keys() and obj_name2 in gqa_dict['label_to_idx'].keys():
                    id1 = val_data[image_ids[im]]['objects'][obj_ids[ob]]['relative_id']
                    id2 = val_data[image_ids[im]]['objects'][obj2_id]['relative_id']
                    predicates = np.append(predicates,gqa_dict['predicate_to_idx'][obj1_rel])
                    relationships = np.vstack([relationships,[id1,id2]])
                    #print(obj_name1 + ''+ obj1_rel + '' + obj_name2  + str(r))
                    r = r + 1
    img_to_last_rel = np.append(img_to_last_rel,r-1)
    if img_to_last_rel[-1] == img_to_first_rel[-1] -1:
        img_to_first_rel[-1] = -1
        img_to_last_rel[-1] = -1
    #print('Img_to_first_rel')
    #print(img_to_first_rel[im])
    #print('Img_to_last_rel')
    #print(img_to_last_rel[im])



relationships = relationships[1:,:]
boxes_1024 = boxes_1024[1:,:]
boxes_512 = boxes_512[1:,:]
attributes = attributes[1:,:]
#print(relationships)
#print(boxes_1024)

f = h5.File('GQA-SGG.h5', 'w')
labels = np.reshape(labels,(-1,1))
predicates = np.reshape(predicates,(-1,1))
f.create_dataset('labels', data=labels.astype(int))
f.create_dataset('boxes_1024', data=boxes_1024)
f.create_dataset('boxes_512', data=boxes_512)
f.create_dataset('img_to_first_box', data=img_to_first_box.astype(int))
f.create_dataset('img_to_last_box', data=img_to_last_box.astype(int))
f.create_dataset('predicates', data=predicates.astype(int))
f.create_dataset('relationships', data=relationships)
f.create_dataset('img_to_first_rel', data=img_to_first_rel.astype(int))
f.create_dataset('img_to_last_rel', data=img_to_last_rel.astype(int))
f.create_dataset('split', data=split.astype(int))
f.create_dataset('attributes', data=attributes)
f.create_dataset('active_object_mask', data=active_object_mask.astype(bool))




