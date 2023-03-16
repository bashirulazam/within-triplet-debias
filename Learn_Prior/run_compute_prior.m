clear all
close all
clc

dataset = "vg"; % or "gqa";
method = "aug";% or "aug";
savefile = strcat('..\Post_infer_evi_prior\Prior\BN_priors_',method,'_',dataset,'.mat');

if dataset == "vg"
    if method == "org"
        %load the original training samples 
        load('..\Gen_data_evidence_prior\Gen_data_prior\VG\data_rel_ground_training_vg.mat')
    elseif method == "aug"
        %load the augmented training samples 
        load('..\Gen_data_evidence_prior\Gen_data_prior\VG\training_data_vg_emb.mat')
        ground_rel_data = training_triplets;
    end
elseif dataset == "gqa"
    if method == "org"
       %load the original training samples 
       load('..\Gen_data_evidence_prior\Gen_data_prior\GQA\data_rel_ground_training_gqa.mat')
    elseif method == "aug"
       %load the augmented training samples
       load('..\Gen_data_evidence_prior\Gen_data_prior\GQA\training_data_gqa_emb.mat')
       ground_rel_data = training_triplets;
    end
end


all_triplets = zeros([1,3]);

for i = 1:length(ground_rel_data)
    i
    all_triplets = [all_triplets; ground_rel_data{1,i}];
end

all_triplets = all_triplets(2:end,:);
[pr_rso,pr_r] = compute_prior(double(all_triplets));
pr_r  = pr_r/sum(pr_r);
pr_r = pr_r';

pr_r_so = pr_rso;

save(savefile,'pr_r_so','pr_r')
