# Introduction

This is the github repo of our recently published work [Probabilistic Debiasing of Scene Graphs](https://openaccess.thecvf.com/content/CVPR2023/html/Biswas_Probabilistic_Debiasing_of_Scene_Graphs_CVPR_2023_paper.html) at CVPR 2023. We address the long-tailed distibution of scene graphs through within-triplet debiasing of measurement triplets. We perform experiments on Visual Genome and GQA dataset and achieve state-of-the-art debiased scene graphs. The codebase is primarily developed in MATLAB. The measurement results of the baseline models are extracted using python code and stored as MAT file. Afterwards, we perform the inference, evaluate the scene graphs, and plot the recall of each relationship with MATLAB. 

# Training and Testing

We generate measurements from the publicly available baselines released by Kang et al. (Unbiased scene graph generation from biased training). The github link for baseline training and testing is https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

# Uncertain evidence from baseline model

You can download our extracted evidence data for VCTree model [here](https://drive.google.com/drive/folders/1-gIr7jz2Jf65mc1gJEcRh3iA0z4xr6TD?usp=share_link). You can also use the following steps for extracting evidence data from the baseline data by yourself.  

## For Visual Genome 
1. Go to "Gen_data_evidence/" 
2. Copy "generate_meas_infer_triplet.py" for VG in the run directory of the baseline (see github directory above). It will generate measurement results of testing images with associated measurement probabilities for specified baseline (imp, motif, vctree, ...), setting (predcls, sgcls, sgdet), dataset (vg) . The MATLAB data file will have the name "data_rel_meas_infer_*baseline*\_*setting*\_*dataset*.mat". An example is "data_rel_meas_infer_vctree_sgcls_vg.mat"
3. Copy and run "generate_tripelts_ground.py" for VG in the same dir. It will generate ground truth annotations for testing data in VG ("data_rel_ground_vctree_sgcls_vg.mat") 

## For GQA
The baseline github repo by Kang et al. does not cover GQA dataset. We will release our baseline with GQA dataset very soon!
<!---
1. Go to "Gen_data_evidence/Gen_data_evidence/GQA"
2. Copy "generate_meas_infer_triplet_GQA.py" for GQA in the run directory of the baseline (see github directory above). It will generate measurement results with associated measurement probabilities for VCTree PredCls in GQA. ("data_rel_meas_infer_vctree_predcls_GQA_full.mat")
3. Copy and run "generate_tripelts_ground_GQA.py" for GQA in the same directory. It will generate ground truth annotations for testing data in GQA. ("data_rel_ground_vctree_predcls_GQA_full") -->


<!---# Collecting original samples for learning BN  
## For Visual Genome 
1. Go inside folder "Gen_data_evidence_prior/Gen_data_prior/VG"
2. Run "generate_triplets_training_gt_org.py". It will generate MAT file ("data_rel_ground_training_vg.mat") of GT annotations for VG with the original training samples.

## For GQA
1. Go inside folder "Gen_data_evidence_prior/Gen_data_prior/GQA"
2. Run "generate_triplets_training_gt_org.py". It will generate MAT file ("data_rel_ground_training_GQA.mat") of GT annotations for GQA with the original training samples. 

# Collecting augmented samples for learning BN  
## For Visual Genome
1. Go inside folder "Gen_data_evidence_prior/Gen_data_prior/VG"
2. Run "generate_embeddings_for_triplets_vg.py' to generate embeddings of VG triplets and save as "embeddings_rel_val_vg.mat". 
3. Run "generate_triplets_training_gt_aug_vg.py". It will generate MAT file ("training_data_vg_emb.mat") of GT annotations for VG with the augmented training samples.

## For GQA
1. Go inside folder "Gen_data_evidence_prior/Gen_data_prior/GQA"
2. Run "generate_embeddings_for_triplets_GQA.py' to generate embeddings of GQA triplets and save as ""embeddings_rel_val_GQA.mat" 
3. Run "generate_triplets_training_gt_aug_GQA.py". It will generate MAT file ("training_data_GQA_emb.mat") of GT annotations for GQA with the augmented training samples.


# Learning BN with both original and augmentated samples
1. Go insider folder "Learn_Prior/"
2. Run "run_compute_prior.m" with  _dataset_ = {'vg', 'gqa'} and _method_ = {'org', 'aug'}. This will create four MAT files: (1) BN_priors_org_vg.mat, (2) BN_priors_aug_vg.mat, (3) BN_priors_org_gqa.mat, and (4) BN_priors_aug_gqa.mat. -->

# Performing Inference
1. Go insider folder "Post_infer_evi_prior/"
2. You can select the model, setting, and dataset in the "run_post_inference.m". 
3. Run "run_post_inference.m". It will take the measurements as input and produce inferred triplets with BN learnt from original and augmented samples. 
	It will also calculate and plot the mean recall improvement. 
