import json
import numpy as np
from scipy.io import savemat,loadmat
import h5py as h5

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
from scipy import spatial


rel_data = json.load(open('train_sceneGraphs.json'))
M = len(rel_data)


dict_sgg = json.load(open('../GQA-SGG-dicts.json'))

#
val_entities = list(dict_sgg['label_to_idx'].keys())
all_sentences = []
all_image_keys = list(rel_data.keys())
for i in range(0, M):
    print(i)
    image_key = all_image_keys[i]
    print(image_key)
    rels = rel_data[image_key]
    obj_keys = list(rels['objects'].keys())
    total_objs = len(obj_keys)
    for k in range(0,total_objs):
        obj_data = rels['objects'][obj_keys[k]]
        sub_name = obj_data['name']
        if sub_name in val_entities:
            sub_rels = obj_data['relations']
            total_rel = len(sub_rels)
            for r in range(0,total_rel):
                relation_name = sub_rels[r]['name']
                obj_name = rels['objects'][sub_rels[r]['object']]['name']
                if obj_name in val_entities:
                    rth_sentence_kth_obj_ith_im = sub_name + ' ' + relation_name + ' ' + obj_name
                    print(rth_sentence_kth_obj_ith_im)
                    all_sentences.append(rth_sentence_kth_obj_ith_im)


#
unique_sentence = list(set(all_sentences))
unique_sentence.sort()
# new_unique_sentence = []
count_sentence = []
# Nr = 50
# map_from_sub_to_full = np.zeros([Nr])
# for r in range(0, Nr):
#      rel = dict_sgg['idx_to_predicate'][str(r+1)]
#      sentence = sub + ' ' + rel + ' ' + obj
#      if sentence in unique_sentence:
#          full_index = unique_sentence.index(sentence)
#          map_from_sub_to_full[r] = full_index+1 #because of matlab
#      else:
#          map_from_sub_to_full[r] = -1
#
for r in range(0,len(unique_sentence)):
    print(r)
    count_sentence.append(all_sentences.count(unique_sentence[r]))
#
#
#
# #Now generate embedding for all the new unique sentences of man and shirt
Nf = 768
New_Nr = len(unique_sentence)
embeddings_rel_val_gqa = np.ones([Nf, New_Nr])
for t in range(0, New_Nr):
    print(t)
    sentence = unique_sentence[t]
    print(sentence)
    embeddings_rel_val_gqa[:,t] = model.encode(sentence)

savemat('embeddings_rel_val_gqa.mat',{'embeddings_rel_val_gqa':embeddings_rel_val_gqa,'unique_sentence':unique_sentence,'count_sentence':count_sentence})
