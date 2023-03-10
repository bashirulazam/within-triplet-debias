import json
import numpy as np
from scipy.io import savemat,loadmat
import h5py as h5

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
from scipy import spatial
#
def get_image_info(image_data,rel_data):
    image_info = []
    rel_info = []
    corrupted_ims = ['1592', '1722', '4616', '4617']
    for i,item in enumerate(image_data):
        print(i)
        if str(item['image_id']) not in corrupted_ims:
            image_info.append(item)
            rel_info.append(rel_data[i])
    return image_info,rel_info
#
image_data = json.load(open('image_data.json'))
rel_data_all = json.load(open('relationships.json'))
data = h5.File("VG-SGG-with-attri.h5",'r')
img_info, rel_data = get_image_info(image_data,rel_data_all)
split = data['split'][:]
M = np.where(split == 0)[0][-1] + 1


dict_sgg = json.load(open('VG-SGG-dicts.json'))
rel_table = loadmat('all_triplets.mat')['rel_table']
sub_obj_set = rel_table[:,[0,2]]
unique_sub_obj_set = np.unique(sub_obj_set,axis=0)
#
val_entities = list(dict_sgg['label_to_idx'].keys())
all_sentences = []

for i in range(0, M):
    print(i)
    Nt = len(rel_data[i]['relationships'])
    for j in range(0,Nt):
        jth_triplet_ith_image = rel_data[i]['relationships'][j]
        if 'name' in jth_triplet_ith_image['subject']:
            jth_sub  = jth_triplet_ith_image['subject']['name']
        elif 'names' in jth_triplet_ith_image['subject']:
            jth_sub = jth_triplet_ith_image['subject']['names'][0]
        else:
            print('no name for subject or object')

        if 'name' in jth_triplet_ith_image['object']:
            jth_obj = jth_triplet_ith_image['object']['name']
        elif 'names' in jth_triplet_ith_image['object']:
            jth_obj = jth_triplet_ith_image['object']['names'][0]
        else:
            print('no name for subject or object')

        jth_predicate = jth_triplet_ith_image['predicate']

        if jth_sub in val_entities and jth_obj in val_entities:
            jth_sentence = jth_sub.lower() + ' ' + jth_predicate.lower() + ' ' + jth_obj.lower()
            all_sentences.append(jth_sentence)
            print(jth_sentence)

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
embeddings_rel_val_vg = np.ones([Nf, New_Nr])
for t in range(0, New_Nr):
    print(t)
    sentence = unique_sentence[t]
    print(sentence)
    embeddings_rel_val_vg[:,t] = model.encode(sentence)
#
savemat('embeddings_rel_val_vg.mat',{'embeddings_rel_val_vg':embeddings_rel_val_vg,'unique_sentence':unique_sentence,'count_sentence':count_sentence})
