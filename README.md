# Introduction

This is the github repo of our recently published work [Probabilistic Debiasing of Scene Graphs](https://openaccess.thecvf.com/content/CVPR2023/html/Biswas_Probabilistic_Debiasing_of_Scene_Graphs_CVPR_2023_paper.html) at CVPR 2023. We address the long-tailed distibution of scene graphs through within-triplet debiasing of measurement triplets. We perform experiments on Visual Genome and GQA dataset and achieve state-of-the-art debiased scene graphs. The codebase is primarily developed in MATLAB. The measurement results of the baseline models are extracted using python code and stored as MAT file. Afterwards, we perform the inference, evaluate the scene graphs, and plot the recall of each relationship with MATLAB. 

# Baseline Models

We generate measurements from the following publicly available baselines.
1. IMP, MOTIF, VCTree, Causal-MOTIF (TDE-MOTIF) -- Released by Kang et al. (Unbiased scene graph generation from biased training). The github link is https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
2. DLFE_MOTIF -- Released by Chiou et al. (Recovering the Unbiased Scene Graphs from the Biased Ones). The github link is https://github.com/coldmanck/recovering-unbiased-scene-graphs
3. BGNN -- Released by Li et al. (Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation). The github link is https://github.com/SHTUPLUS/PySGG


## Uncertain evidence from baseline models

You can download our extracted evidence data for the above-mentioned baselines [here](https://rpi.box.com/s/r0uyi8eyyyj1334dubvfm905mcb92h7p). You can also use the following steps for extracting evidence data from the baseline data by yourself.  

1. Go to "Gen_data_evidence/" 
2. Copy "generate_meas_infer_triplet.py" in the run directory of the baseline (see github directories above). It will generate measurement results of testing images with associated measurement probabilities for specified baseline (imp, motif, vctree, ...), setting (predcls, sgcls, sgdet), and dataset (vg,gqa) . The MATLAB data file will have the name "data_rel_meas_infer_*baseline*\_*setting*\_*dataset*.mat". An example is "data_rel_meas_infer_vctree_sgcls_vg.mat"
3. Copy and run "generate_tripelts_ground.py" in the same dir. It will generate ground truth annotations for testing data in VG. An example is "data_rel_ground_vctree_sgcls_vg.mat" 

# Datasets (Visual Genome and GQA) 
The baseline github repos already cover Visual Genome. However, they do not utilize GQA dataset. You can follow our [DATASET.MD](https://github.com/bashirulazam/within-triplet-debias/blob/main/DATASET.md) for creating GQA dataset. 


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
3. Run "run_post_inference.m". It will take the measurements as input and produce inferred triplets with BN learnt from original and augmented samples. It will also calculate and plot the mean recall improvement. 

# Citation
If you find our work useful, please cite our paper

```
@InProceedings{Biswas_2023_CVPR,
    author    = {Biswas, Bashirul Azam and Ji, Qiang},
    title     = {Probabilistic Debiasing of Scene Graphs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10429-10438}
}

```
