function is_converged = check_diff(triplets,prev_triplets,ent_labels,relations)

check_flag = 1;
diff_triplets = abs(triplets - prev_triplets);
is_converged = sum(sum(diff_triplets,2))==0;

if is_converged
    for e = 1:length(ent_labels)
            ent_label = ent_labels(e);
            triplets_ind_as_subject = find(relations(:,1)==e);
            triplets_ind_as_object = find(relations(:,2)==e);
            labels_in_triplet = [triplets(triplets_ind_as_subject,1); triplets(triplets_ind_as_object,3)];
            if ~isempty(labels_in_triplet) && ~all(labels_in_triplet == labels_in_triplet(1))
                check_flag = 0;
                break;
            end
    end
end
is_converged = is_converged*check_flag;