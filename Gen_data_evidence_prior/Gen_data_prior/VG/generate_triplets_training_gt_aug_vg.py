import json
import numpy as np
from scipy.io import savemat,loadmat
import h5py as h5

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-mpnet-base-v2')
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

val_entities = list(dict_sgg['label_to_idx'].keys())
eps_aug = 0.95

#Remember to load the saved values instead of running it directly
embeddings_rel_val_vg = loadmat('embeddings_rel_val_vg.mat')['embeddings_rel_val_vg']
unique_sentence = loadmat('embeddings_rel_val_vg.mat')['unique_sentence']
count_sentence = loadmat('embeddings_rel_val_vg.mat')['count_sentence']

#find out the subject list from unique_sentences
pair_list = []
for l in range(0,len(unique_sentence)):
    pair_list.append(unique_sentence[l].split()[0]+ ' ' +unique_sentence[l].split()[-1])



unique_sentence_sub_obj = [[[] for x in range(150)] for y in range(150)]
count_sentence_sub_obj = [[[] for x in range(150)] for y in range(150)]
embeddings_sub_obj = [[[] for x in range(150)] for y in range(150)]
val_sentence_sub_obj = [[[] for x in range(150)] for y in range(150)]
map_from_unique_to_valid = [[[] for x in range(150)] for y in range(150)]
unique_sen_sub_obj_nice = [[[] for x in range(150)] for y in range(150)]

for i in range(0,150):
    for j in range(0,150):
        print(i)
        print(j)
        sub = dict_sgg['idx_to_label'][str(i+1)]
        obj = dict_sgg['idx_to_label'][str(j+1)]
        pair = sub + ' ' + obj
        indices = [k for k, x in enumerate(pair_list) if x == pair]
        unique_sentence_sub_obj[i][j].append(unique_sentence[indices])
        count_sentence_sub_obj[i][j].append(count_sentence[0,indices])
        embeddings_sub_obj[i][j].append(embeddings_rel_val_vg[:,indices])



Nr = 50
Ns = 150
No = 150
pr_r_so_cat = np.zeros([Nr,Ns,No])
pr_r_so_emb = np.zeros([Nr,Ns,No])
marginal_pr_cat = np.zeros([Nr])
marginal_pr_emb = np.zeros([Nr])
for i in range(0,Ns):
    for j in range(0,No):
        print(i)
        print(j)
        if len(count_sentence_sub_obj[i][j][0]) > 0:

            sub = dict_sgg['idx_to_label'][str(i+1)]
            obj = dict_sgg['idx_to_label'][str(j+1)]

            count_vector_full_index = count_sentence_sub_obj[i][j][0]
            Np = len(count_vector_full_index)
            similarity_matrix = np.identity(Np)
            embeddings = embeddings_sub_obj[i][j][0]
            unique_sen_i_j = unique_sentence_sub_obj[i][j][0]
            unique_sen_i_j_nice = []

            #Calculate the similarity matrix
            for p in range(0,Np):
                unique_sen_i_j_nice.append(unique_sen_i_j[p].strip())
                for q in range(0,Np):
                    pth_embedding = embeddings[:,p]
                    qth_embedding = embeddings[:,q]
                    similarity_matrix[p,q] = 1-spatial.distance.cosine(pth_embedding, qth_embedding)

            similarity_matrix = similarity_matrix - np.identity(Np)
            unique_sen_sub_obj_nice[i][j].append(unique_sen_i_j_nice)
            #Calculate the valid set
            good_index_full_set = []
            corresponding_good_index_sub_set = []

            #Find the map of good_index_set and corresponding good_index_set
            for r in range(0,Nr):
                rel = dict_sgg['idx_to_predicate'][str(r + 1)]
                sentence_sub_set = sub + ' ' + rel + ' ' + obj
                val_sentence_sub_obj[i][j].append(sentence_sub_set)
                if sentence_sub_set in unique_sen_i_j_nice:
                    good_index_full_set.append(unique_sen_i_j_nice.index(sentence_sub_set))
                    corresponding_good_index_sub_set.append(r)



            good_index_full_set = np.asarray(good_index_full_set)
            corresponding_good_index_sub_set = np.asarray(corresponding_good_index_sub_set)
            actual_count_vector = np.zeros([Nr])
            emb_count_vector = np.zeros([50])

            if len(good_index_full_set) > 0:
                map_vector = -1*np.ones([Np])
                for p in range(0,Np):
                    sentence_full_set = unique_sen_i_j_nice[p]
                    if sentence_full_set in val_sentence_sub_obj[i][j]:
                        sub_set_index = val_sentence_sub_obj[i][j].index(sentence_full_set)
                        emb_count_vector[sub_set_index] = emb_count_vector[sub_set_index] + count_vector_full_index[p]
                        actual_count_vector[sub_set_index] = count_vector_full_index[p]
                        marginal_pr_emb[sub_set_index] = marginal_pr_emb[sub_set_index] + emb_count_vector[sub_set_index]
                        marginal_pr_cat[sub_set_index] = marginal_pr_cat[sub_set_index] + actual_count_vector[sub_set_index]
                        map_vector[p] = sub_set_index
                    else:
                        max_sim_with_good_indices = np.max(similarity_matrix[p,good_index_full_set])
                        if max_sim_with_good_indices > eps_aug:
                            #candidate_good_index = good_index_full_set[np.argmax(similarity_matrix[p,good_index_full_set])]
                            sub_set_index = corresponding_good_index_sub_set[np.argmax(similarity_matrix[p,good_index_full_set])]
                            emb_count_vector[sub_set_index] = emb_count_vector[sub_set_index] +  count_vector_full_index[p]
                            marginal_pr_emb[sub_set_index] = marginal_pr_emb[sub_set_index] + emb_count_vector[sub_set_index]
                            map_vector[p] = sub_set_index

                map_from_unique_to_valid[i][j].append(map_vector)
                pr_r_so_cat[:,i,j] = actual_count_vector/sum(actual_count_vector)
                pr_r_so_emb[:,i,j] = emb_count_vector/sum(emb_count_vector)
            else:
                pr_r_so_cat[:, i, j] = 0.02 * np.ones(50)
                pr_r_so_emb[:, i, j] = 0.02 * np.ones(50)
        else:
            pr_r_so_cat[:, i, j] = 0.02*np.ones(50)
            pr_r_so_emb[:, i, j] = 0.02*np.ones(50)

#savemat('prior_and_emb_prior_val.mat',{'pr_r_so_cat':pr_r_so_cat,'pr_r_so_emb':pr_r_so_emb,'marginal_pr_cat':marginal_pr_cat,'marginal_pr_emb':marginal_pr_emb})


training_triplets = []
training_triplets_box_list = []
val_img_ids = []
for i in range(0, M):
    print(i)
    triplets = []
    triplets_boxes = []
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
            subi = dict_sgg['label_to_idx'][jth_sub]
            obji = dict_sgg['label_to_idx'][jth_obj]

            jth_sentence = jth_sub.lower() + ' ' + jth_predicate.lower() + ' ' + jth_obj.lower()
            if len(map_from_unique_to_valid[subi-1][obji-1]) > 0:
                uni_index = unique_sen_sub_obj_nice[subi-1][obji-1][0].index(jth_sentence)
                mapped_index = int(map_from_unique_to_valid[subi-1][obji-1][0][uni_index])

                if mapped_index != -1:
                    print(jth_sentence)
                    s = subi
                    o = obji
                    r = mapped_index + 1
                    #subject box
                    sx = jth_triplet_ith_image['subject']['x']
                    sy = jth_triplet_ith_image['subject']['y']
                    sw = jth_triplet_ith_image['subject']['w']
                    sh = jth_triplet_ith_image['subject']['h']
                    #object box
                    ox = jth_triplet_ith_image['object']['x']
                    oy = jth_triplet_ith_image['object']['y']
                    ow = jth_triplet_ith_image['object']['w']
                    oh = jth_triplet_ith_image['object']['h']
                    triplets.append([s,r,o])
                    triplets_boxes.append([sx, sy, sw, sh, ox, oy, ow, oh])

    if len(triplets) > 0:
        training_triplets.append(triplets)
        val_img_ids.append(rel_data[i]['image_id'])
        training_triplets_box_list.append(triplets_boxes)


savemat('training_data_vg_emb.mat', {'training_triplets':training_triplets, 'val_img_ids':val_img_ids, 'training_triplets_box_list':training_triplets_box_list})