function [acc_rec_meas,mean_recall_meas,correct_relations_measured,correct_relations_measured_all,total_ground_truth_relations_all,recall_meas_per_rel,cat_IA_cat,aligned_triplets_first_one] = compute_recalls(measured_triplets,measured_triplets_box_list,ground_rel_data,ground_triplets_box_list,test_start,test_end,top,dict,title_part)
%Remember this is applicable for unbiased baseline
%Change the xyxy_to_matlab for Graph-RCNN baseline
iou_thresh = 0.5;
NR = 50;
for i = test_start:test_end
    i
   
    L = size(measured_triplets{1,i}(:,:),1);
    Lg = size(ground_rel_data{1,i}(:,:),1);
    correct_relations_measured(i) = 0;
    correct_relations_measured_all(:,i) = zeros(NR,1);
    total_ground_truth_relations(i) = Lg;
    total_ground_truth_relations_all(:,i) = zeros(NR,1);
    mis_tri_gt(1,i)  = 0;
    mis_tri_gt(2,i)  = 0;
    mis_tri_gt(3,i)  = Lg;
    %ground_truth_relations{1,i} = trip_to_rel(ground_rel_data{1,i},dict);
    for g = 1:Lg
        
   
        gt_tri = ground_rel_data{1,i}(g,:);
        total_ground_truth_relations_all(gt_tri(2),i) = total_ground_truth_relations_all(gt_tri(2),i) + 1;
        %Calculating Accuracy for Measured Relations
        
        measured_subj_box_list = double((xyxy_to_matlab(measured_triplets_box_list{1,i}(1:min(L,top),1:4) )));
        measured_obj_box_list = double((xyxy_to_matlab(measured_triplets_box_list{1,i}(1:min(L,top),5:8) )));
        ground_subj_box_list = xyxy_to_matlab(double(ground_triplets_box_list{1,i}(g,1:4) ));
        ground_obj_box_list = xyxy_to_matlab(double(ground_triplets_box_list{1,i}(g,5:8) ));
        subj_width = ground_subj_box_list(3);
        subj_height = ground_subj_box_list(4);
        obj_width = ground_obj_box_list(3);
        obj_height = ground_obj_box_list(4);
        if subj_width == 0 || subj_height == 0 || obj_width == 0 || obj_height == 0 
            disp(strcat('Error in GT annotiation ', num2str(i)))
            continue
        end
        subj_iou = bboxOverlapRatio(ground_subj_box_list,measured_subj_box_list);
        obj_iou = bboxOverlapRatio(ground_obj_box_list,measured_obj_box_list);
        aligned_indices = find((subj_iou>=iou_thresh).*(obj_iou>=iou_thresh) == 1);
        if isempty(aligned_indices)
            mis_tri_gt(1,i) =  mis_tri_gt(1,i) + 1;
        else 
            mis_tri_gt(2,i) = mis_tri_gt(2,i) + 1;
        end
        pred_tri = measured_triplets{1,i}(aligned_indices,:);
        IA2 = find(sum(abs(double(gt_tri) - double(pred_tri)),2) == 0);
        correct_relations_measured(i) = correct_relations_measured(i) + any(IA2);
        correct_relations_measured_all(gt_tri(2),i) = correct_relations_measured_all(gt_tri(2),i) + any(IA2);
        measured_aligned_triplets{1,i}{1,g} =  pred_tri;
        measured_aligned_indices{1,i}{1,g} = aligned_indices;
        La = size(measured_aligned_triplets{1,i}{1,g},1);
        
        %There might be many matching predicted triplets, we only want
        %to see the first one
        if ~isempty(aligned_indices)
            aligned_triplets_first_one{1,i}(g,:) = pred_tri(1,:);
        end
        
        if La ~= 0 
            measured_aligned_relations{1,i}{1,g} = trip_to_rel(measured_aligned_triplets{1,i}{1,g},dict);
        end
    end
end


total_mis_gt = sum(mis_tri_gt(1,test_start:test_end)); 
total_det_gt = sum(mis_tri_gt(2,test_start:test_end));
total_gt = sum(mis_tri_gt(3,test_start:test_end));


%Mean Recall & Recall 
val_image_no = sum(total_ground_truth_relations(test_start:end)~=0);
acc_rec_meas = sum(correct_relations_measured(test_start:end)./(total_ground_truth_relations(test_start:end)+eps))/val_image_no;
%Finding the row where at least one ground truth relation exist
IA = find(sum(total_ground_truth_relations_all(:,test_start:end),2) ~= 0);
for i = 1:length(IA)
    cat_IA{1,i} = getfield(dict.idx_to_predicate,strcat('x',num2str(IA(i))));
end
cat_IA_cat = categorical(cat_IA);
dummy = correct_relations_measured_all(IA,test_start:end)./total_ground_truth_relations_all(IA,test_start:end);
nonandummy = dummy;
nonandummy(isnan(dummy)) = 0;
%finding where ground truth is not zero
valid_sum = sum(total_ground_truth_relations_all(IA,:)~=0,2);
recall_meas_per_rel = sum(nonandummy,2)./valid_sum;
mean_recall_meas = mean(recall_meas_per_rel);
figure
bar(cat_IA_cat,recall_meas_per_rel)
title(strcat("Mean Recall for ",title_part," Result"))
ylim([0 1])

