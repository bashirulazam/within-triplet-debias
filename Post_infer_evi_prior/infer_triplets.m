function [inferred_triplets] = infer_triplets(measured_triplets,relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset)

%load prior with original samples
if strcmp(method,'reg')
    if strcmp(dataset,'VG')
        load('./Prior/VG/BN_priors.mat');
        dict = jsondecode(fileread('VG-SGG-dicts.json'));
    elseif strcmp(dataset,'GQA')
        load('./Prior/GQA/BN_priors_GQA.mat')
        dict = jsondecode(fileread('GQA-SGG-dicts.json'));

    else
        disp('wrong dataset')
    end
%load prior with augmented samples
elseif strcmp(method,'emb') 
    if strcmp(dataset,'VG')
        load('./Prior/VG/BN_priors_emb.mat');
        dict = jsondecode(fileread('VG-SGG-dicts.json'));
    elseif strcmp(dataset,'GQA')
        load('./Prior/GQA/BN_priors_emb_GQA.mat')
        dict = jsondecode(fileread('GQA-SGG-dicts.json'));
    else
        disp('wrong dataset')
    end
else
    disp('wrong name in the method')
end

NS = 150;
NO  = 150;
NR = 50;

%Perform inference for each testing image
for i = test_start:test_end
    i
    
    L = size(measured_triplets{1,i}(:,:),1);
    K = min(top,L);
    %WTI for each triplet in Top@K
    for k = 1:K
            relk = k;
            subi = relations_list{1,i}(k,1);
            obji = relations_list{1,i}(k,2);
            sub_dist = obj_logits_list{1,i}(subi,:);
            obj_dist = obj_logits_list{1,i}(obji,:);
            rel_dist = predicate_logits_list{1,i}(relk,:); 
            r_sm = sub_dist./sum(sub_dist);
            r_om = obj_dist./sum(obj_dist);
            r_rm = rel_dist./sum(rel_dist);

            sm = measured_triplets{1,i}(k,1);
            rm = measured_triplets{1,i}(k,2);
            om = measured_triplets{1,i}(k,3);
            
            r_sm = fix_obj_dist(r_sm,sm);
            r_om = fix_obj_dist(r_om,om);
  
            r_rm = r_rm./sum(r_rm);
           [s,r,o] = VE_infer_exp1_v3(pr_r,pr_r_so,r_sm,r_om,r_rm,sm,rm,om);
            inferred_triplets{1,i}(k,1:3) = [s, r, o];
            
    end
    
    
end

