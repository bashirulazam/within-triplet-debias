function [pr_rso,pr_r] = compute_prior(all_triplets)


NS = 150;
NR = 50;
NO = 150;
sub_obj_set = all_triplets(:,[1 3]);

for s = 1:NS 
    for o = 1:NO 
        sub_obj = [s, o];
        count_parents(s,o) = length(find(sum(abs(sub_obj_set - sub_obj),2) == 0));
        if count_parents(s,o) ~= 0
           for r = 1:NR 
               triplet = [s, r, o];
               count_relations(r,s,o) = length(find(sum(abs(all_triplets - triplet),2) == 0));
               pr_rso(r,s,o) = count_relations(r,s,o)/count_parents(s,o);
           end
        else 
            pr_rso(:,s,o) = (1/NR);
        end
    end
end

rels = all_triplets(:,2);
for r = 1:NR 
    pr_r(r) = sum(rels == r);
end