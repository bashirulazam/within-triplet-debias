import json
import numpy as np
from scipy.io import savemat,loadmat
import h5py as h5

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-mpnet-base-v2')
from scipy import spatial


rel_data = json.load(open('train_sceneGraphs.json'))
M = len(rel_data)


dict_sgg = json.load(open('GQA-SGG-dicts.json'))

#
val_entities = list(dict_sgg['label_to_idx'].keys())
all_sentences = []
all_image_keys = list(rel_data.keys())


#Remember to load the saved values instead of running it directly
embeddings_rel_val_gqa = loadmat('embeddings_rel_val_gqa.mat')['embeddings_rel_val_gqa']
unique_sentence = loadmat('embeddings_rel_val_gqa.mat')['unique_sentence']
count_sentence = loadmat('embeddings_rel_val_gqa.mat')['count_sentence']

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
        embeddings_sub_obj[i][j].append(embeddings_rel_val_gqa[:,indices])



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
                        if max_sim_with_good_indices > 0.95:
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

#savemat('GQA_data/prior_and_emb_prior_val_gqa.mat',{'pr_r_so_cat':pr_r_so_cat,'pr_r_so_emb':pr_r_so_emb,'marginal_pr_cat':marginal_pr_cat,'marginal_pr_emb':marginal_pr_emb})


training_triplets = []
training_triplets_box_list = []
val_img_ids = []
for i in range(0, M):
    print(i)
    triplets = []
    image_key = all_image_keys[i]
    print(image_key)
    rels = rel_data[image_key]
    obj_keys = list(rels['objects'].keys())
    total_objs = len(obj_keys)
    for k in range(0, total_objs):
        obj_data = rels['objects'][obj_keys[k]]
        sub_name = obj_data['name']
        if sub_name in val_entities:
            sub_rels = obj_data['relations']
            total_rel = len(sub_rels)
            for r in range(0, total_rel):
                relation_name = sub_rels[r]['name']
                obj_name = rels['objects'][sub_rels[r]['object']]['name']
                if obj_name in val_entities:
                    rth_sentence_kth_obj_ith_im = sub_name + ' ' + relation_name + ' ' + obj_name
                    print(rth_sentence_kth_obj_ith_im)
                    subi = dict_sgg['label_to_idx'][sub_name]
                    obji = dict_sgg['label_to_idx'][obj_name]
                    if len(map_from_unique_to_valid[subi-1][obji-1]) > 0:
                        uni_index = unique_sen_sub_obj_nice[subi-1][obji-1][0].index(rth_sentence_kth_obj_ith_im)
                        mapped_index = int(map_from_unique_to_valid[subi-1][obji-1][0][uni_index])

                        if mapped_index != -1:
                            print(rth_sentence_kth_obj_ith_im)
                            s = subi
                            o = obji
                            r = mapped_index + 1
                            # #subject box
                            # sx = jth_triplet_ith_image['subject']['x']
                            # sy = jth_triplet_ith_image['subject']['y']
                            # sw = jth_triplet_ith_image['subject']['w']
                            # sh = jth_triplet_ith_image['subject']['h']
                            # #object box
                            # ox = jth_triplet_ith_image['object']['x']
                            # oy = jth_triplet_ith_image['object']['y']
                            # ow = jth_triplet_ith_image['object']['w']
                            # oh = jth_triplet_ith_image['object']['h']
                            triplets.append([s,r,o])
                            # triplets_boxes.append([sx, sy, sw, sh, ox, oy, ow, oh])

    if len(triplets) > 0:
        training_triplets.append(triplets)
        val_img_ids.append(image_key)
        # training_triplets_box_list.append(triplets_boxes)


savemat('training_data_GQA_emb.mat', {'training_triplets':training_triplets, 'val_img_ids':val_img_ids})