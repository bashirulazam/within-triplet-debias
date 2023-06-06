function [] = bar_plot_mine(probs)

load('Prior/BN_priors_org_vg.mat')

cat_IA = form_cat('VG');
cat_IA_cat = categorical(cat_IA);
[sorted,ord_ind] = sort(pr_r','descend');
reorder_cat = reordercats(cat_IA_cat,string(cat_IA_cat(ord_ind)));

bar(reorder_cat,probs)
set(gca,'FontSize',12,'FontName','Calibri')
set(gca, 'XTickLabel',{})