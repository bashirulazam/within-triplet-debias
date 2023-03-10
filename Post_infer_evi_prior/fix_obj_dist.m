function pr_om = fix_obj_dist(pr_om,o_m)

[~,o_max_ind] = max(pr_om);

while o_max_ind ~= o_m
    pr_om(o_max_ind) = 0;
    pr_om = pr_om./sum(pr_om);
    [~,o_max_ind] = max(pr_om);
end

pr_om = pr_om/sum(pr_om);