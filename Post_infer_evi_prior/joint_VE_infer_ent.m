function [e,pr_e] = joint_VE_infer_ent(triplets_ind_as_subject,triplets_ind_as_object,measured_triplets,obj_logits,pr_r_so,ent_ind,ent_label)
NE = 150;

sub_dist = obj_logits(ent_ind,:);
r_sm = sub_dist./sum(sub_dist);
sm = ent_label;
r_sm = fix_obj_dist(r_sm,sm);
r_om = r_sm;

 %Inference from the BN 
cnd_pr = zeros(NE,1);

for e = 1:NE
     for p = 1:length(triplets_ind_as_subject)
        trip_ind = triplets_ind_as_subject(p);
        rp = measured_triplets(trip_ind,2);
        op = measured_triplets(trip_ind,3);
        cnd_pr(e) = cnd_pr(e) + (pr_r_so(rp,e,op)*r_sm(e)); %e is subject here
     end
     for q = 1:length(triplets_ind_as_object)
        trip_ind = triplets_ind_as_object(q);
        rq = measured_triplets(trip_ind,2);
        sq = measured_triplets(trip_ind,1);
        cnd_pr(e) = cnd_pr(e) + (pr_r_so(rq,sq,e)*r_om(e)); % e is object here
     end
     
end
cnd_pr = cnd_pr./sum(cnd_pr(:));
[~,I] = sort(cnd_pr(:),'descend');
% flatten_cnd_pr = cnd_pr(:);
% entropy = -flatten_cnd_pr'*log(flatten_cnd_pr+eps);
% index = find_index(cnd_pr,I(1));

if  sum(cnd_pr(:)) ~= 0
    e = I(1);
else
    e = sm;
end
pr_e = cnd_pr;    
    