function relation = trip_to_rel(triplets,dict)

N = size(triplets,1);

for i = 1:N
    if triplets(i,1) == 0 
        relation{i,1} = [];
    else
        subject = getfield(dict.idx_to_label,strcat('x',num2str(triplets(i,1))));
        relation{i,1} = subject ;
    end
    
    if triplets(i,2) == 0 
        relation{i,2} = [];
    else
        predicate = getfield(dict.idx_to_predicate,strcat('x',num2str(triplets(i,2))));
        relation{i,2} = predicate ;
    end

    if triplets(i,3) == 0 
        relation{i,3} = [];
    else
        object = getfield(dict.idx_to_label,strcat('x',num2str(triplets(i,3))));
        relation{i,3} = object ;
    end

  
end