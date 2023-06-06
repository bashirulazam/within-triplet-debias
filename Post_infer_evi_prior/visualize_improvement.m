function [] = visualize_improvement(recall_before_per_rel,recall_after_per_rel)

load('Prior/BN_priors_org_vg.mat')

cat_IA = form_cat('VG');
cat_IA_cat = categorical(cat_IA);
[sorted,ord_ind] = sort(pr_r','descend');
reorder_cat = reordercats(cat_IA_cat,string(cat_IA_cat(ord_ind)));
improve_per_rel =  (recall_after_per_rel - recall_before_per_rel)*100;
figure
for r = 1:50
    h = bar(reorder_cat(r),improve_per_rel(r));
    hold on
    if improve_per_rel(r) > 0
       set(h,'FaceColor','g');
    else
       set(h,'FaceColor','c');
    end
    
end

set(gca,'FontSize',9,'FontName','Calibri')
ylim([-60,40])
Y = improve_per_rel(ord_ind,1);
text(1:length(Y),Y',strcat(num2str(Y,'%0.1f'),'%'),'horiz','right','rotation',90,'FontSize',9,'FontName','Calibri');
xlabel('Relationships','FontSize',16,'FontName','Calibri')
ylabel('(%) Increase of Recalls  ','FontSize',16,'FontName','Calibri')