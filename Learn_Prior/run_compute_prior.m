clear all
close all
clc

dataset = "VG"; % or "GQA";
method = "aug";% or "aug";

if dataset == "VG"
    if method == "org"
        %load the original training samples 
        load('..\Gen_data_evidence_prior\Gen_data_prior\VG\data_rel_ground_training_vg.mat')
        savefile = 'VG\BN_priors.mat';
    elseif method == "aug"
        %load the augmented training samples 
        load('..\Gen_data_evidence_prior\Gen_data_prior\VG\training_data_vg_emb.mat')
        ground_rel_data = training_triplets;
        savefile = 'VG\BN_priors_emb.mat';
    end
elseif dataset == "GQA"
    if method == "org"
       %load the original training samples 
       load('..\Gen_data_evidence_prior\Gen_data_prior\GQA\data_rel_ground_training_GQA.mat')
       savefile = 'GQA\BN_priors_GQA.mat';
    elseif method == "aug"
       %load the augmented training samples
       load('..\Gen_data_evidence_prior\Gen_data_prior\GQA\training_data_GQA_emb.mat')
       ground_rel_data = training_triplets;
       savefile = 'GQA\BN_priors_emb_GQA.mat';
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
