import json
import numpy as np
train_file = open('train_sceneGraphs.json')
train_data = json.load(train_file)
dict_file = open('VG-SGG-dicts.json')
dict = json.load(dict_file)
#obj_dict = list(dict['label_to_idx'].keys())
obj_count = {}
rel_count = {}
att_count = {}
image_ids = list(train_data.keys())
M = len(image_ids)
for im in range(0,M):
    obj_ids = list(train_data[image_ids[im]]['objects'].keys())
    NO = len(obj_ids)
    for ob in range(0,NO):
        obj_name1 = train_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 not in obj_count:
            #print(obj_name)
            obj_count[obj_name1] = 0
        else:
            obj_count[obj_name1] = obj_count[obj_name1] + 1

        obj1_rel_all = train_data[image_ids[im]]['objects'][obj_ids[ob]]['relations']
        N_obj1_rel = len(obj1_rel_all)
        for re in range(0,N_obj1_rel):
            obj1_rel = obj1_rel_all[re]['name']
            if obj1_rel not in rel_count:
                rel_count[obj1_rel] = 0 
            else:
                rel_count[obj1_rel] = rel_count[obj1_rel] + 1
        obj1_attr_all = train_data[image_ids[im]]['objects'][obj_ids[ob]]['attributes']
        N_obj1_at = len(obj1_attr_all)
        for at in range(0,N_obj1_at):
            obj1_at = obj1_attr_all[at]
            if obj1_at not in att_count:
                att_count[obj1_at] = 0
            else:
                att_count[obj1_at] = att_count[obj1_at] + 1


val_file = open('val_sceneGraphs.json')
val_data = json.load(val_file)
dict_file = open('VG-SGG-dicts.json')
dict = json.load(dict_file)
#obj_dict = list(dict['label_to_idx'].keys())
image_ids = list(val_data.keys())
M = len(image_ids)
for im in range(0,M):
    obj_ids = list(val_data[image_ids[im]]['objects'].keys())
    NO = len(obj_ids)
    for ob in range(0,NO):
        obj_name1 = val_data[image_ids[im]]['objects'][obj_ids[ob]]['name']
        if obj_name1 not in obj_count:
            #print(obj_name)
            obj_count[obj_name1] = 0
        else:
            obj_count[obj_name1] = obj_count[obj_name1] + 1
        obj1_rel_all = val_data[image_ids[im]]['objects'][obj_ids[ob]]['relations']
        N_obj1_rel = len(obj1_rel_all)
        for re in range(0,N_obj1_rel):
            obj1_rel = obj1_rel_all[re]['name']
            if obj1_rel not in rel_count:
                rel_count[obj1_rel] = 0
            else:
                rel_count[obj1_rel] = rel_count[obj1_rel] + 1

        obj1_attr_all = val_data[image_ids[im]]['objects'][obj_ids[ob]]['attributes']
        N_obj1_at = len(obj1_attr_all)
        for at in range(0,N_obj1_at):
            obj1_at = obj1_attr_all[at]
            if obj1_at not in att_count:
                att_count[obj1_at] = 0
            else:
                att_count[obj1_at] = att_count[obj1_at] + 1



#print(rel_count)
#print(obj_count)


rel_count = sorted(rel_count.items(),key = lambda item: item[1],reverse = True)
obj_count = sorted(obj_count.items(),key = lambda item: item[1],reverse = True)
att_count = sorted(att_count.items(),key = lambda item: item[1],reverse = True)
#print(all_rels)
#print(all_objs)

NO = 150
NR = 50
NA = 200
rel_dict = []
obj_dict = []
att_dict = []
#Skipping left/right
for r in range(2,NR+2):
    rel_dict.append(rel_count[r][0])

for o in range(0,NO):
    obj_dict.append(obj_count[o][0])

for a in range(0,NA):
    att_dict.append(att_count[a][0])

print(rel_dict)
print(obj_dict)
print(att_dict)

gqa_dict = {}
gqa_dict['idx_to_label'] = {}
gqa_dict['label_to_idx'] = {}
gqa_dict['idx_to_predicate'] = {}
gqa_dict['predicate_to_idx'] = {}
gqa_dict['idx_to_attribute'] = {}
gqa_dict['attribute_to_idx'] = {}
for r in range(0,NR):
    gqa_dict['idx_to_predicate'][str(r+1)] = rel_dict[r]
    gqa_dict['predicate_to_idx'][rel_dict[r]] = r+1

for o in range(0,NO):
    gqa_dict['idx_to_label'][str(o+1)] = obj_dict[o]
    gqa_dict['label_to_idx'][obj_dict[o]] = o+1

for a in range(0,NA):
    gqa_dict['idx_to_attribute'][str(a+1)] = att_dict[a]
    gqa_dict['attribute_to_idx'][att_dict[a]] = a+1



outfile = open('GQA-SGG-dicts.json','w')
json.dump(gqa_dict,outfile)
#print(gqa_dict)
