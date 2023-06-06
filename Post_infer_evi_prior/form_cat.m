function cat_IA_cat = form_cat(dataset)

if dataset == 'VG'
    dict = jsondecode(fileread('VG-SGG-dicts.json'));
elseif dataset == 'GQA'
    dict = jsondecode(fileread('GQA-SGG-dicts.json'));
end
for i = 1:50
    cat_IA{1,i} = getfield(dict.idx_to_predicate,strcat('x',num2str(i)));
end

cat_IA_cat = categorical(cat_IA);