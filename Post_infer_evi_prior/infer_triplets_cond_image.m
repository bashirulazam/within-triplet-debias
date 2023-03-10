function [inferred_triplets] = infer_triplets_cond_image(measured_triplets,predicate_logits,top,pr_r_so,pr_r)

NS = 150;
NO  = 150;
NR = 50;

L = size(predicate_logits,1);
K = min(top,L);
inferred_triplets = zeros(size(measured_triplets));    
for k = 1:K

    relk = k;
    rel_dist = predicate_logits(relk,:); 
    r_rm = rel_dist./sum(rel_dist);

    sm = measured_triplets(k,1);
    rm = measured_triplets(k,2);
    om = measured_triplets(k,3);

    r_rm = r_rm./sum(r_rm);

    s = sm;
    o = om;
    %Prior + Uncertain Evidence
    [r,posterior,ent,score] = MAP_conditional_scaled(pr_r_so(:,s,o),pr_r,r_rm',rm);

    inferred_triplets(k,1:3) = [s, r, o];


end


