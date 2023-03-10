clear all
close all
clc

%loading measurement and ground truth
dataset = 'VG';% 'VG' or 'GQA'
dict = jsondecode(fileread(strcat(dataset,'-SGG-dicts.json')));
meas_inpath = strcat('..\..\..\Release_Data\',dataset,'_Data\');
model = 'VCTree';
setting = 'SGCls'; % 'PredCls', 'SGCls' or 'SGDet'
meas_outpath = strcat(dataset,'_Data\');
meas_outfile = strcat(meas_outpath,model,'\',setting,'\','data_rel_meas_infer_',model,'_',setting,'_',dataset,'.mat');
meas_file = strcat(meas_inpath,model,'\',setting,'\','data_rel_meas_infer_',model,'_',setting,'_',dataset,'.mat');
gt_file = strcat(meas_inpath,model,'\',setting,'\','data_rel_ground_',model,'_',setting,'_',dataset,'.mat');
load(meas_file)
load(gt_file)

test_start = 1;
test_end = length(ground_box_list);
top = 100;


for i = 1:1000
    sampled_ground_box_list{1,i} = ground_box_list{1,i};
    sampled_ground_label_list{1,i} = ground_label_list{1,i};
    sampled_ground_rel_data{1,i} = ground_rel_data{1,i};
    sampled_ground_triplets_box_list{1,i} = ground_triplets_box_list{1,i};
    sampled_measured_box_list{1,i} = measured_box_list{1,i};
    sampled_measured_label_list{1,i} = measured_label_list{1,i};
    sampled_measured_relations_list{1,i} = measured_relations_list{1,i};
    sampled_measured_score_list{1,i} = measured_score_list{1,i};
    sampled_measured_triplets{1,i} = measured_triplets{1,i};
    sampled_measured_triplets_box_list{1,i} = measured_triplets_box_list{1,i};
    sampled_predicate_logits_list{1,i} = predicate_logits_list{1,i};
    sampled_obj_logits_list{1,i} = obj_logits_list{1,i};
    
end
clearvars -except sampled_* meas_outfile

ground_box_list = sampled_ground_box_list;
ground_label_list = sampled_ground_label_list;
ground_rel_data = sampled_ground_rel_data;
ground_triplets_box_list = sampled_ground_triplets_box_list;
measured_box_list = sampled_measured_box_list;
measured_label_list = sampled_measured_label_list;
measured_relations_list = sampled_measured_relations_list;
measured_score_list = sampled_measured_score_list;
measured_triplets = sampled_measured_triplets;
measured_triplets_box_list = sampled_measured_triplets_box_list;
predicate_logits_list = sampled_predicate_logits_list;
obj_logits_list = sampled_obj_logits_list;
clearvars sampled_*
save(meas_outfile)

