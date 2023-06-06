function [predicate_logits_list, obj_logits_list] = combine_sgdet_probs(datapath,suffix)
range_start = 1;
range_end = 10000;
range_string = strcat(num2str(range_start),'_',num2str(range_end));
load(strcat(datapath,'data_rel_meas_infer_prob_',suffix,'_',range_string,'.mat'))
offset = 0;
for i = range_start:range_end
    new_predicate_logits_list{1,i} = predicate_logits_list{1,i-offset};
    new_object_logits_list{1,i} = obj_logits_list{1,i-offset};
end
range_start = 10001;
range_end = 20000;
range_string = strcat(num2str(range_start),'_',num2str(range_end));
load(strcat(datapath,'data_rel_meas_infer_prob_',suffix,'_',range_string,'.mat'))
offset = 10000;
for i = range_start:range_end
    new_predicate_logits_list{1,i} = predicate_logits_list{1,i-offset};
    new_object_logits_list{1,i} = obj_logits_list{1,i-offset};
end
range_start = 20001;
range_end = 26446;
range_string = strcat(num2str(range_start),'_',num2str(range_end));
load(strcat(datapath,'data_rel_meas_infer_prob_',suffix,'_',range_string,'.mat'))
offset = 20000;
for i = range_start:range_end
    new_predicate_logits_list{1,i} = predicate_logits_list{1,i-offset};
    new_object_logits_list{1,i} = obj_logits_list{1,i-offset};
end
clear predicate_logits_list obj_logits_list
predicate_logits_list = new_predicate_logits_list;
obj_logits_list = new_object_logits_list;