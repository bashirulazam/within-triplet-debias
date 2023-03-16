import sys
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

results = torch.load('checkpoints/trained_vctree_predcls/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch')
groundtruths = results['groundtruths']
M = len(groundtruths)
triplets_list = []
triplets_scores_list = []
labels_list = []
boxes_list = []
score_list = []
triplets_box_list =[]
range_start = 0
range_end = 26446
setting = 'predcls_'
method = 'vctree_'
indices = np.arange(range_start,range_end)
for i in indices:
    gt = groundtruths[i]
    image_width = gt.size[0]
    image_height = gt.size[1]
    obj_rois = gt.bbox
    obj_labels = gt.extra_fields['labels']
    rel_tuple = gt.extra_fields['relation_tuple']
    rel_inds = rel_tuple[:,:2]
    predicates = rel_tuple[:,2]
    obj_rois = obj_rois.cpu()
    obj_labels = obj_labels.cpu()
    rel_inds = rel_inds.cpu()
    box_preds = obj_rois.numpy()
    num_boxes = box_preds.shape[0]
    relations = rel_inds.numpy()
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    classes = obj_labels.numpy()
    boxes = box_preds
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    for j in range(num_relations):
        triplets[j, 1] = predicates[j]
        sub_j, obj_j = relations[j,:2]
        triplets[j, 0] = classes[sub_j]
        triplets[j, 2] = classes[obj_j]
        triplet_boxes[j, :4] = boxes[sub_j, :]
        triplet_boxes[j, 4:] = boxes[obj_j, :]
    subjects = np.zeros([1, num_relations]).astype(np.int32)
    relations = np.zeros([1, num_relations]).astype(np.int32)
    objects = np.zeros([1, num_relations]).astype(np.int32)
    #for j, index in enumerate(sorted_triplet_scores_inds):
        #print(triplets[index])
    print("Appending " + str(i))
    triplets_list.append(triplets)
    labels_list.append(np.reshape(classes,(classes.shape[0],-1)))
    boxes_list.append(box_preds)
    triplets_box_list.append(triplet_boxes) 

range_start = range_start + 1 #For MATLAB Saving 
range_end = range_end

#im_range = str(range_start) +'_' + str(range_end)
im_range = 'full'

save_file = 'data_rel_ground_' + method + setting + 'vg_' + im_range + '.mat'
sio.savemat(save_file, {'ground_rel_data':triplets_list,'ground_box_list':boxes_list,'ground_label_list':labels_list,'ground_triplets_box_list':triplets_box_list})
