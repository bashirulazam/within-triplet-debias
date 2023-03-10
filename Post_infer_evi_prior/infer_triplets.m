function [inferred_triplets] = infer_triplets(measured_triplets,relations_list,obj_logits_list,predicate_logits_list,top,test_start,test_end,method,dataset)


if strcmp(method,'reg')
    if strcmp(dataset,'VG')
        load('BN_priors.mat');
    elseif strcmp(dataset,'GQA')
        load('BN_priors_GQA.mat')
    else
        disp('wrong dataset')
    end
elseif strcmp(method,'emb') 
    if strcmp(dataset,'VG')
%         load('Sentence_Embedding/VG_data/prior_and_emb_prior_val_vg.mat');
%         pr_r_so = pr_r_so_emb;
%         pr_r = marginal_pr_emb'/sum(marginal_pr_emb);
          load('Sentence_Embedding/VG_data/BN_priors_emb.mat');
    elseif strcmp(dataset,'GQA')
        load('Sentence_Embedding/GQA_data/prior_and_emb_prior_val_gqa.mat')
        pr_r_so = pr_r_so_emb;
        pr_r = marginal_pr_emb'/sum(marginal_pr_emb);
    else
        disp('wrong dataset')
    end
else
    disp('wrong name in the method')
end
%This is for learnt prior from categorical space
% pr_r_so = pr_r_so_cat;
% pr_r = marginal_pr_cat'/sum(marginal_pr_cat);

%This is for learnt prior from embedding space
 
% load('BN_priors_mod.mat');
% pr_r_so = pr_r_so_mod;
%load('HEX_graphs.mat')
%load('BN_priors_typical.mat');

NS = 150;
NO  = 150;
NR = 50;

for i = test_start:test_end
    i
    L = size(measured_triplets{1,i}(:,:),1);
    K = min(top,L);
    
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
               
%             s = sm;
%             o = om;
%             [r,posterior,ent,score] = MAP_conditional_scaled(pr_r_so(:,s,o),pr_r,r_rm',rm);
%             
                  
           
            inferred_triplets{1,i}(k,1:3) = [s, r, o];
%             inferred_prob{1,i}(k,:) = posterior;
%             entropy_list{1,i}(k) = ent;
%             post_score{1,i}(k) = score;
        
    end
end

