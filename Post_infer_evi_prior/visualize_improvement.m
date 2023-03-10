function [] = visualize_improvement(recall_before_per_rel,recall_after_per_rel,fig_title,dataset)

if strcmp(dataset,'VG')
    load('BN_priors.mat');
elseif strcmp(dataset,'GQA')
    load('BN_priors_GQA.mat')
else
    disp('wrong dataset')
end




cat_IA = form_cat('VG');
cat_IA_cat = categorical(cat_IA);
[sorted,ord_ind] = sort(pr_r','descend');
reorder_cat = reordercats(cat_IA_cat,string(cat_IA_cat(ord_ind)));
improve_per_rel =  (recall_after_per_rel - recall_before_per_rel)*100;
figure
bar(reorder_cat,improve_per_rel)
ylim([-40,100])
Y = improve_per_rel(ord_ind,1);
text(1:length(Y),Y',strcat(num2str(Y,'%0.2f'),'%'),'horiz','left','rotation',90);
xlabel('Relationships')
ylabel('Improvemnt of Recall @100')
title(fig_title)