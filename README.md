# Training and Testing

We implemented the publicly available baseline Kang et al. (Unbiased scene graph generation from biased training). The github link for baseline training and testing is https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

# Generating uncertain evidence/measurement results using baseline model. 
## For Visual Genome 
1. Go to "Gen_data_evidence_prior\Gen_data_evidence\VG" 
2. Copy "generate_meas_infer_triplet.py" for VG in the run directory of the baseline (see github directory above). It will generate measurement results with associated measurement probabilities for VCTree PredCls of VG testing images ("data_rel_meas_infer_vctree_predcls_vg_full.mat")
3. Copy and run "generate_tripelts_ground.py" for VG in the same dir. It will generate ground truth annotations for testing data in VG ("data_rel_ground_vctree_predcls_vg_full") 

## For GQA
1. Go to "Gen_data_evidence_prior\Gen_data_evidence\GQA"
2. Copy "generate_meas_infer_triplet_GQA.py" for GQA in the run directory of the baseline (see github directory above). It will generate measurement results with associated measurement probabilities for VCTree PredCls in GQA. ("data_rel_meas_infer_vctree_predcls_GQA_full.mat")
3. Copy and run "generate_tripelts_ground_GQA.py" for GQA in the same directory. It will generate ground truth annotations for testing data in GQA. ("data_rel_ground_vctree_predcls_GQA_full")


# Collecting original samples for learning BN  
## For Visual Genome 
1. Go inside folder "Gen_data_evidence_prior\Gen_data_prior\VG"
2. Run "generate_triplets_training_gt_org.py". It will generate MAT file ("data_rel_ground_training_vg.mat") of GT annotations for VG with the original training samples.

## For GQA
1. Go inside folder "Gen_data_evidence_prior\Gen_data_prior\GQA"
2. Run "generate_triplets_training_gt_org.py". It will generate MAT file ("data_rel_ground_training_GQA.mat") of GT annotations for GQA with the original training samples. 


# Collecting augmented samples for learning BN  
## For Visual Genome
1. Go inside folder "Gen_data_evidence_prior\Gen_data_prior\VG"
2. Run "generate_embeddings_for_triplets_vg.py' to generate embeddings of VG triplets and save as "embeddings_rel_val_vg.mat". 
3. Run "generate_triplets_training_gt_aug_vg.py". It will generate MAT file ("training_data_vg_emb.mat") of GT annotations for VG with the augmented training samples.

## For GQA
1. Go inside folder "Gen_data_evidence_prior\Gen_data_prior\GQA"
2. Run "generate_embeddings_for_triplets_GQA.py' to generate embeddings of GQA triplets and save as ""embeddings_rel_val_GQA.mat" 
3. Run "generate_triplets_training_gt_aug_GQA.py". It will generate MAT file ("training_data_GQA_emb.mat") of GT annotations for GQA with the augmented training samples.


# Learning Within-triplet prior with both original and augmentated samples
1. Go insider folder "Learn_Prior\"
2. Run "run_compute_prior.m". It will either generate "BN_prior.mat" (VG) or "BN_prior_GQA.mat" (GQA) with original training samples 
	or "BN_prior_emb.mat" (VG) or "BN_priors_GQA_emb.mat" (GQA)  with augmented training samples. 

# Performing Inference
1. Go insider folder "Post_infer_evi_prior/"
2. You can select the dataset and setting in the "run_post_inference.m". 
3. Run "run_post_inference.m". It will take the measurements as input and produce inferred triplets with BN learnt from original and augmented samples. 
	It will also calculate and plot the mean recall improvement. 
