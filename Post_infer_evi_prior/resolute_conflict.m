function [resoluted_inferred_triplets,inferred_label_list] = resolute_conflict(inferred_triplets,measured_triplets,measured_relations_list,predicate_logits_list,obj_logits_list,measured_label_list,test_start,test_ind,top,method,dataset)
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


for i = test_start:test_ind
    i
    is_converged = 0;
    L =  size(measured_relations_list{1,i}(:,:),1);
    K = min(top,L);
    triplets = inferred_triplets{1,i}(1:K,:);
    ent_labels = measured_label_list{1,i};
    inferred_label_list{1,i} = measured_label_list{1,i};
    relations = measured_relations_list{1,i}(1:K,:);
    prev_triplets = triplets;
    predicate_logits = predicate_logits_list{1,i};
    obj_logits = obj_logits_list{1,i};
    while ~is_converged   
        %Update object labels with consistency
        for e = 1:length(ent_labels)
            ent_label = ent_labels(e);
            triplets_ind_as_subject = find(relations(:,1)==e);
            triplets_ind_as_object = find(relations(:,2)==e);
            labels_in_triplet = [triplets(triplets_ind_as_subject,1); triplets(triplets_ind_as_object,3)];
            if ~isempty(labels_in_triplet) 
                inferred_label = joint_VE_infer_ent(triplets_ind_as_subject,triplets_ind_as_object,triplets,obj_logits,pr_r_so,e,ent_label);
                triplets(triplets_ind_as_subject,1) = inferred_label;
                triplets(triplets_ind_as_object,3) = inferred_label;
                labels_in_triplet_after = [triplets(triplets_ind_as_subject,1); triplets(triplets_ind_as_object,3)];  
                ent_labels(e) = inferred_label;
            end
    
        end   
        %Update the relationship labels based on the updated object labels
        [triplets] = infer_triplets_cond_image(triplets,predicate_logits_list{1,i},top,pr_r_so,pr_r);
        is_converged = check_diff(triplets, prev_triplets,ent_labels,relations);
        if ~is_converged
            prev_triplets = triplets; 
        end      
    end
    %extracting the updated object labels
    resoluted_inferred_triplets{1,i} = triplets;
    for e = 1:length(ent_labels)
        triplets_ind_as_subject = find(relations(:,1)==e);
        triplets_ind_as_object = find(relations(:,2)==e);
        labels_in_triplet = [triplets(triplets_ind_as_subject,1); triplets(triplets_ind_as_object,3)];
        if ~isempty(labels_in_triplet)
            inferred_label_list{1,i}(e) = labels_in_triplet(1);%Every label should be same
        else
            inferred_label_list{1,i}(e) = ent_labels(e); %Put the measured label here
        end
    end
        
end
