function [inferred_triplets] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset)

if strcmp(method,'org')
    if strcmp(dataset,'VG')
        load('BN_priors.mat');
    elseif strcmp(dataset,'GQA')
        load('BN_priors_GQA.mat')
    else
        disp('wrong dataset')
    end
elseif strcmp(method,'aug') 
    if strcmp(dataset,'VG')
       load('BN_priors_emb.mat');
    elseif strcmp(dataset,'GQA')
        load('BN_priors_emb_GQA.mat')
    else
        disp('wrong dataset')
    end
else
    disp('wrong name in the method')
end

NS = 150;
NO  = 150;
NR = 50;

for i = test_start:test_end
    i
    L = size(predicate_logits_list{1,i}(:,:),1);
    K = min(top,L);
    
    for k = 1:K
           
            relk = k;
%             subi = relations_list{1,i}(k,1);
%             obji = relations_list{1,i}(k,2);
%             sub_dist = obj_logits_list{1,i}(subi,:);
%             obj_dist = obj_logits_list{1,i}(obji,:);
            rel_dist = predicate_logits_list{1,i}(relk,:); 
%             r_sm = sub_dist./sum(sub_dist);
%             r_om = obj_dist./sum(obj_dist);
            r_rm = rel_dist./sum(rel_dist);

            sm = measured_triplets{1,i}(k,1);
            rm = measured_triplets{1,i}(k,2);
            om = measured_triplets{1,i}(k,3);
            
%             r_sm = fix_obj_dist(r_sm,sm);
%             r_om = fix_obj_dist(r_om,om);
%   
            r_rm = r_rm./sum(r_rm);



%           [s,r,o] = VE_infer_exp1_v3(pr_r,pr_r_so,r_sm,r_om,r_rm,sm,rm,om);
               
            s = sm;
            o = om;
            %Prior + Uncertain Evidence
            [r,posterior,ent,score] = MAP_conditional_scaled(pr_r_so(:,s,o),pr_r,r_rm',rm);
%            [~,r] = max(pr_r_so(:,s,o));
                  
           
            inferred_triplets{1,i}(k,1:3) = [s, r, o];
%             inferred_prob{1,i}(k,:) = posterior;
%             entropy_list{1,i}(k) = ent;
%             post_score{1,i}(k) = score;
        
    end
end

