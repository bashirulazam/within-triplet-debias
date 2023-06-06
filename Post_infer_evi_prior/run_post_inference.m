clear all
close all
clc

model = 'vctree';
dataset = 'vg';
setting = "predcls"; % or "predcls", "sgcls", "sgdet"
top = 100;
suffix = strcat(model,'_',setting,'_',dataset);
datapath = strcat('..\Data\',dataset,'\',model,'\',setting,'\');


%loading measurement results from baseline model
load(strcat(datapath,'data_rel_meas_infer_',suffix,'.mat'))
%loading ground truth results
load(strcat(datapath,'data_rel_ground_',suffix,'.mat'))

if strcmp(setting,'sgdet') && strcmp(dataset,'vg') 
    [predicate_logits_list, obj_logits_list] = combine_sgdet_probs(datapath,suffix);
end

%loading dictionary file for VG or GQA
if strcmp(dataset, 'vg')
    dict = jsondecode(fileread('VG-SGG-dicts.json'));
end
if strcmp(dataset, 'gqa')
   dict = jsondecode(fileread('GQA-SGG-dicts.json'));
end 



%You can perform testing with full dataset or partial dataset
test_start = 1;
test_end = length(ground_box_list);

    
    
%Evaluating measurement results
title_part = 'measured';
[recall_meas,mean_recall_meas,correct_relations_measured,correct_relations_measured_all,total_ground_truth_relations_all,recall_meas_per_rel,cat_IA_cat,measured_aligned_triplets] = compute_recalls(measured_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);



% In this stage, we will perform within triplet inference with prior
% learnt from origial samples
method = 'org';
title_part = 'inferred-org';
if strcmp(setting,'sgcls') || strcmp(setting,'sgdet')
    [inferred_triplets] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
    [resoluted_inferred_triplets] = resolute_conflict(inferred_triplets,measured_triplets,measured_relations_list,predicate_logits_list,obj_logits_list,measured_label_list,test_start,test_end,top,method,dataset);
    [recall_infer,mean_recall_infer,correct_relations_inferred,correct_relations_inferred_all,total_ground_truth_relations_all,recall_infer_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);
elseif strcmp(setting,'predcls')
    [inferred_triplets] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
    [recall_infer,mean_recall_infer,correct_relations_inferred,correct_relations_inferred_all,total_ground_truth_relations_all,recall_infer_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);
end

% In this stage, we will perform within triplet inference with prior
% learnt from augmented samples
method = 'aug';
title_part = 'inferred-aug';
if strcmp(setting,'sgcls') || strcmp(setting,'sgdet')
    [inferred_triplets_emb] = infer_triplets(measured_triplets,measured_relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset);
    [resoluted_inferred_triplets_emb] = resolute_conflict(inferred_triplets_emb,measured_triplets,measured_relations_list,predicate_logits_list,obj_logits_list,measured_label_list,test_start,test_end,top,method,dataset);
    [recall_infer_emb,mean_recall_infer_emb,correct_relations_inferred_emb,correct_relations_inferred_all_emb,total_ground_truth_relations_all,recall_infer_emb_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(resoluted_inferred_triplets_emb,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);    
elseif strcmp(setting,'predcls')
    [inferred_triplets_emb] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset);
    [recall_infer_emb,mean_recall_infer_emb,correct_relations_inferred_emb,correct_relations_inferred_all_emb,total_ground_truth_relations_all,recall_infer_emb_per_rel,cat_IA_cat,inferred_aligned_triplets] = compute_recalls(inferred_triplets_emb,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part);    
end
        
        
       

%Display Results
disp(strcat("Recall with measurements: ",num2str(recall_meas*100)))
disp(strcat("Recall with Inf (org): ",num2str(recall_infer*100)))
disp(strcat("Recall with Inf (aug): ",num2str(recall_infer_emb*100)))

disp(strcat("Mean Recall with measurements: ",num2str(mean_recall_meas*100)))
disp(strcat("Mean Recall with Inf (org): ",num2str(mean_recall_infer*100)))
disp(strcat("Mean Recall with Inf (aug): ",num2str(mean_recall_infer_emb*100)))