import sys
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
results = torch.load('./checkpoints/trained_vctree_predcls/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch')
predictions = results['predictions']
M = len(predictions)
triplets_meas_list = []
triplets_infer_list = []
triplets_scores_list = []
labels_list = []
boxes_list = []
score_list = []
triplets_box_list =[]
relations_list = []
obj_logits_list = []
predicate_logits_list = []
range_start = 0
range_end = 26446
indices = np.arange(range_start,range_end)
method = 'vctree_'
setting = 'predcls_'
for i in indices:
    print(i)
    prediction = predictions[i]
    #prediction_pred = prediction_preds[i]
    image_width = prediction.size[0]
    image_height = prediction.size[1]
    obj_rois = prediction.bbox
    obj_scores = prediction.extra_fields['pred_scores'] 
    obj_labels = prediction.extra_fields['pred_labels']
    obj_logits = prediction.extra_fields['pred_obj_prob']
    obj_logits = np.float16(obj_logits.numpy()[:,1:])
    rel_inds = prediction.extra_fields['rel_pair_idxs']
    rel_scores = prediction.extra_fields['pred_rel_scores']
    obj_rois = obj_rois.cpu()
    obj_scores = obj_scores.cpu()
    obj_labels = obj_labels.cpu()
    rel_inds = rel_inds.cpu()
    rel_scores = rel_scores.cpu()
    box_preds = obj_rois.numpy()
    num_boxes = box_preds.shape[0]
    predicate_logits = rel_scores.numpy()
    predicate_logits = np.float16(predicate_logits[:, 1:])
    #Measured predicates is from the predicate logits
    predicates_meas = np.argmax(predicate_logits, 1).ravel() + 1
    #Normalize the softmax logits
    predicate_scores = predicate_logits.max(axis=1).ravel()
    relations = rel_inds.numpy()
    num_relations = relations.shape[0]
    classes = obj_labels.numpy()
    #classes = mylabels
    class_scores = obj_scores.numpy()
    subs = classes[relations[:,0]]
    objs = classes[relations[:,1]]
    boxes = box_preds
    assert(predicates_meas.shape[0] == relations.shape[0])
    triplets_meas = np.zeros([num_relations, 3]).astype(np.int32)
    triplets_infer = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations, 3]).astype(np.float32)
    triplet_scores_prod = np.zeros([num_relations]).astype(np.float32)
    for j in range(num_relations):
        triplets_meas[j, 1] = predicates_meas[j]
        sub_j, obj_j = relations[j,:2]
        triplets_meas[j, 0] = classes[sub_j]
        triplets_meas[j, 2] = classes[obj_j]
        triplet_boxes[j, :4] = boxes[sub_j, :]
        triplet_boxes[j, 4:] = boxes[obj_j, :]
        triplet_scores[j,0] =  class_scores[sub_j]
        triplet_scores[j,2] = class_scores[obj_j]
        triplet_scores[j,1] = predicate_scores[j]
        triplet_scores_prod[j] = class_scores[sub_j]*class_scores[obj_j]*predicate_scores[j]
    sorted_triplet_scores_inds = np.argsort(triplet_scores_prod)[::-1]
    sorted_triplet_scores_inds = sorted_triplet_scores_inds[triplet_scores_prod[sorted_triplet_scores_inds] > 0 ]
    #for j, index in enumerate(sorted_triplet_scores_inds):
        #print(triplets[index])
    print("Appending " + str(i))
    top_K = np.minimum(num_relations,992)
    sorted_triplet_scores_inds = sorted_triplet_scores_inds[:top_K]
    triplets_meas = triplets_meas[sorted_triplet_scores_inds,:]
    triplet_boxes  = triplet_boxes[sorted_triplet_scores_inds,:]
    triplet_scores = triplet_scores[sorted_triplet_scores_inds]
    relations = relations[sorted_triplet_scores_inds,:]
    predicate_logits = predicate_logits[sorted_triplet_scores_inds,:]

    triplets_meas_list.append(triplets_meas)
    triplets_scores_list.append(triplet_scores)
    labels_list.append(np.reshape(classes,(classes.shape[0],-1)))
    score_list.append(np.reshape(class_scores,(class_scores.shape[0],-1)))
    boxes_list.append(box_preds)
    triplets_box_list.append(triplet_boxes)
    #for compatibility with MATLAB the +1 comes
    relations_list.append(relations + 1)
    obj_logits_list.append(obj_logits)
    predicate_logits_list.append(predicate_logits)


range_start = range_start + 1 #For MATLAB Saving 
range_end = range_end


im_range = str(range_start) +'_' + str(range_end)
im_range = 'full'
save_file = 'data_rel_meas_infer_' + method + setting + 'vg_' + im_range + '.mat'


sio.savemat(save_file, {'measured_triplets':triplets_meas_list,'measured_box_list':boxes_list,'measured_label_list':labels_list,'measured_score_list':score_list,'measured_triplets_box_list':triplets_box_list,'measured_relations_list':relations_list,'predicate_logits_list':predicate_logits_list,'obj_logits_list':obj_logits_list})
