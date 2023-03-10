clear all
close all
clc

dataset = "GQA"; %or GQA
setting = 'predcls_';
model = '_vctree_';
top = 100;
if dataset == "VG"
    dict = jsondecode(fileread('VG-SGG-dicts.json'));
    range_start = 1;
    range_end = 26446;
    range = '_full';
    meas_path = '..\Gen_data_evidence_prior\Gen_data_evidence\VG\';
    meas_file = strcat(meas_path,'data_rel_meas_infer',model,setting,'vg',range,'.mat');
    gt_file = strcat(meas_path,'data_rel_ground',model,setting,'vg',range,'.mat');
    load(meas_file)
    load(gt_file)

elseif dataset == "GQA"
    dict = jsondecode(fileread('GQA-SGG-dicts.json'));
    range_start = 1;
    range_end = 7227;
    range = '_full';
    meas_path = '..\Gen_data_evidence_prior\Gen_data_evidence\GQA\';
    meas_file = strcat(meas_path,'data_rel_meas_infer',model,setting,'GQA',range,'.mat');
    gt_file = strcat(meas_path,'data_rel_ground',model,setting,'GQA',range,'.mat');
    load(meas_file)
    load(gt_file)
end

test_start = 1;
test_end = range_end - range_start + 1;

%Calculating measurement accuracy
title_part = 'measured';
top = 100;
[acc_rec_meas,mean_recall_meas,correct_relations_measured,correct_relations_measured_all,total_ground_truth_relations_all,recall_meas_per_rel,cat_IA_cat,measured_aligned_triplets] = compute_recalls(measured_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);

%For predcls setting, we run infer_triplets_conditional (S = S_g, O = O_g)
%For sgcls and sgdet, we run infer_triplets and conflict resolution
method = 'org';
top = 100;
[inferred_triplets_rel_only] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
title_part = 'inferred-org-rel-only';
top = 100;
[acc_rec_infer_rel_only,mean_recall_infer_rel_only,correct_relations_inferred_rel_only,correct_relations_inferred_rel_only_all,total_ground_truth_relations_all,recall_infer_per_rel_rel_only,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets_rel_only,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);

%For SGCLS and SGDET, we need to run the following codes
%[inferred_triplets] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
%[resoluted_inferred_triplets] = resolute_conflict(inferred_triplets,measured_relations_list,measured_label_list,test_start,test_end,top);
% 
%title_part = 'inferred-reg-rel-obj';
%[acc_rec_infer,mean_recall_infer,correct_relations_inferred,correct_relations_inferred_all,total_ground_truth_relations_all,recall_infer_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);

method = 'aug';
top = 100;
[inferred_triplets_emb_rel_only] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
title_part = 'inferred-aug-rel-only';
[acc_rec_infer_emb_rel_only,mean_recall_infer_emb_rel_only,correct_relations_inferred_emb_rel_only,correct_relations_inferred_all_emb_rel_only,total_ground_truth_relations_all,recall_infer_emb_rel_only_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets_emb_rel_only,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);

%For SGCLS and SGDET, we need to run the following codes
%[inferred_triplets_emb] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
%[resoluted_inferred_triplets_emb] = resolute_conflict(inferred_triplets_emb,measured_relations_list,measured_label_list,test_start,test_end,top);
%[acc_rec_infer_emb,mean_recall_infer_emb,correct_relations_inferred_emb,correct_relations_inferred_all_emb,total_ground_truth_relations_all,recall_infer_emb_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets_emb,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);

%%%This part is for visualization of the mean recall improvement
fig_title = 'Improvent of mean recall from measurement to inference (BN-org)';
visualize_improvement(recall_meas_per_rel,recall_infer_per_rel_rel_only,fig_title,dataset)

fig_title = 'Improvent of mean recall from measurement to inference (BN-aug)';
visualize_improvement(recall_meas_per_rel,recall_infer_emb_rel_only_per_rel,fig_title,dataset)

fig_title = 'Improvent of mean recall from inference(BN-org) to inference (BN-aug)';
visualize_improvement(recall_infer_per_rel_rel_only,recall_infer_emb_rel_only_per_rel,fig_title,dataset)
