import h5py as h5 
import numpy as np 
import scipy.io as sio
data = h5.File("VG-SGG-with-attri.h5",'r')
fullM = len(data['img_to_first_rel'])
classes = data['labels']
boxes = data['boxes_1024']
triplets_list = []
boxes_list = []
labels_list = []
triplets_box_list = []
split = data['split'][:]
M = np.where(split == 0)[0][-1] + 1

for image_index in range(M):

    ind1 = data['img_to_first_rel'][image_index]
    if ind1 == -1:
        continue
    ind2 = data['img_to_last_rel'][image_index]
    ind1_box = data['img_to_first_box'][image_index]
    ind2_box = data['img_to_last_box'][image_index]
    num_boxes = ind2_box-ind1_box+1
    num_relations = ind2-ind1+1
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    predicates = data['predicates'][ind1:ind2+1]
    relations = data['relationships'][ind1:ind2+1]
    boxes_list.append(data['boxes_1024'][ind1_box:ind2_box+1])
    labels_list.append(data['labels'][ind1_box:ind2_box+1])
    for j in range(num_relations):
        triplets[j, 1] = predicates[j]
        sub_j, obj_j = relations[j,:2]
        triplets[j, 0] = classes[sub_j]
        triplets[j, 2] = classes[obj_j]
        triplet_boxes[j, :4] = boxes[sub_j, :]
        triplet_boxes[j, 4:] = boxes[obj_j, :]
    triplets = triplets[::-1]
    triplet_boxes = triplet_boxes[::-1]
    triplet_boxes, indices = np.unique(triplet_boxes,axis = 0, return_index = True)    
    triplets_list.append(triplets[indices,:])
    triplets_box_list.append(triplet_boxes)
    #print("Appending " + str(image_index))
    #print(triplets[indices,:])
    #print("Without Sorting")
    #print(triplets)
    print(image_index)
print(len(triplets_list))
sio.savemat('data_rel_ground_training_vg.mat', {'ground_rel_data':triplets_list,'ground_box_list':boxes_list,'ground_label_list':labels_list,'ground_triplets_box_list':triplets_box_list})
