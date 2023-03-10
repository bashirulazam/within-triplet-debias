function [s,r,o] = VE_infer_exp1_v3(pr_r,pr_r_so,r_sm,r_om,r_rm,sm,rm,om)
NS = 150;
NO = 150;
NR = 50;
 
[~,sorted_sub_inds] = sort(r_sm,'descend');
[~,sorted_obj_inds] = sort(r_om,'descend');
[~,sorted_rel_inds] = sort(r_rm,'descend'); 
 
 sm_ind = sorted_sub_inds(1:2);
 om_ind = sorted_obj_inds(1:2);
 rm_ind = sorted_rel_inds(1:20);
% [~,sm_ind] = find(r_sm(1,:) ~= 0);
% [~,om_ind] = find(r_om(1,:) ~= 0);
%  
 

new_r_sm = r_sm(1,sm_ind);
new_r_sm = new_r_sm/sum(new_r_sm);

new_r_om = r_om(1,om_ind);
new_r_om = new_r_om/sum(new_r_om);

new_r_rm= r_rm(1,rm_ind);
new_r_rm = new_r_rm/sum(new_r_rm);

r_sm = new_r_sm;
r_om = new_r_om;
r_rm = new_r_rm;
 %Inference from the BN 
cnd_pr = zeros(2,20,2);

 for si = 1:length(sm_ind)
     s = sm_ind(si);
   for oi = 1:length(om_ind)
       o = om_ind(oi);
       if sum(pr_r_so(:,s,o) == 0.02) ~= NR
        for ri = 1:length(rm_ind)
            r = rm_ind(ri);
            if pr_r_so(r,s,o) ~= 0 
      
               cnd_pr(si,ri,oi) = cnd_pr(si,ri,oi) + (pr_r_so(r,s,o)/pr_r(r))*r_sm(si)*r_om(oi)*r_rm(ri);
                
            end
        end
       end
   end
 end
cnd_pr = cnd_pr./sum(cnd_pr(:));
[~,I] = sort(cnd_pr(:),'descend');
flatten_cnd_pr = cnd_pr(:);
entropy = -flatten_cnd_pr'*log(flatten_cnd_pr+eps);
index = find_index(cnd_pr,I(1));

if  sum(cnd_pr(:)) ~= 0
    s = sm_ind(index(1));
    o = om_ind(index(3));
    r = rm_ind(index(2));
else
    s = sm;
    r = rm;
    o = om;
end
%     
    