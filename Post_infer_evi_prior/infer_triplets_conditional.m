function [inferred_triplets] = infer_triplets_conditional(measured_triplets,predicate_logits_list,top,test_start,test_end,method,dataset)


%getting the right prior
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

for i = test_start:test_end
    i
    L = size(predicate_logits_list{1,i}(:,:),1);
    K = min(top,L);
    
    for k = 1:K
            relk = k;
            rel_dist = predicate_logits_list{1,i}(relk,:); 
            r_rm = rel_dist./sum(rel_dist);

            sm = measured_triplets{1,i}(k,1);
            rm = measured_triplets{1,i}(k,2);
            om = measured_triplets{1,i}(k,3);
            
            r_rm = r_rm./sum(r_rm);
            s = sm;
            o = om;
            [r,posterior,ent,score] = MAP_conditional_scaled(pr_r_so(:,s,o),pr_r,r_rm',rm);

            inferred_triplets{1,i}(k,1:3) = [s, r, o];

        
    end
end

