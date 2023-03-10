clear all
close all
clc

%loading measurement and ground truth
dataset = 'VG';% 'VG' or 'GQA'
dict = jsondecode(fileread(strcat(dataset,'-SGG-dicts.json')));
meas_path = strcat(dataset,'_Data\');
model = 'VCTree';
setting = 'SGCls'; % 'PredCls', 'SGCls' 
meas_file = strcat(meas_path,model,'\',setting,'\','data_rel_meas_infer_',model,'_',setting,'_',dataset,'.mat');
load(meas_file)
test_start = 1;
test_end = length(ground_box_list);
title_part = 'measured';
top = 100;

%Evaluating measurement results
[recall_meas,mean_recall_meas,correct_relations_measured,correct_relations_measured_all,total_ground_truth_relations_all,recall_meas_per_rel,cat_IA_cat,measured_aligned_triplets] = compute_recalls(measured_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);



% In this stage, we will perform within triplet inference with prior
% learnt from origial samples
top = 100;
method = 'reg';
title_part = 'inferred-org';
if strcmp(setting,'SGCls') %|| strcmp(setting,'SGDet')
    [inferred_triplets] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
    [resoluted_inferred_triplets] = resolute_conflict(inferred_triplets,measured_triplets,measured_relations_list,predicate_logits_list,obj_logits_list,measured_label_list,test_start,test_end,top,method,dataset);
    [recall_infer,mean_recall_infer,correct_relations_inferred,correct_relations_inferred_all,total_ground_truth_relations_all,recall_infer_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);
elseif strcmp(setting,'PredCls')
    [inferred_triplets] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
    [recall_infer,mean_recall_infer,correct_relations_inferred,correct_relations_inferred_all,total_ground_truth_relations_all,recall_infer_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);
end

% In this stage, we will perform within triplet inference with prior
% learnt from augmented samples
method = 'emb';
title_part = 'inferred-emb';
top =100;
if strcmp(setting,'SGCls') %|| strcmp(setting,'SGDet')
    [inferred_triplets_emb] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
    [resoluted_inferred_triplets_emb] = resolute_conflict(inferred_triplets_emb,measured_triplets,measured_relations_list,predicate_logits_list,obj_logits_list,measured_label_list,test_start,test_end,top,method,dataset);
    [recall_infer_emb,mean_recall_infer_emb,correct_relations_inferred_emb,correct_relations_inferred_all_emb,total_ground_truth_relations_all,recall_infer_emb_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets_emb,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);    
elseif strcmp(setting,'PredCls')
    [inferred_triplets_emb] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
    [recall_infer_emb,mean_recall_infer_emb,correct_relations_inferred_emb,correct_relations_inferred_all_emb,total_ground_truth_relations_all,recall_infer_emb_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets_emb,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);    
end

%Display Results
disp(strcat('Recall with measurements:',num2str(recall_meas)))
disp(strcat('Recall with Inf (org):',num2str(recall_infer)))
disp(strcat('Recall with Inf (aug):',num2str(recall_infer_emb)))

disp(strcat('Mean Recall with measurements:',num2str(mean_recall_meas)))
disp(strcat('Mean Recall with Inf (org):',num2str(mean_recall_infer)))
disp(strcat('Mean Recall with Inf (aug):',num2str(mean_recall_infer_emb)))

