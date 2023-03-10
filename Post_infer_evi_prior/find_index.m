function index = find_index(array,I)

neldim      = size(array);              % Length of each dimension
ndim        = length(neldim);           % Number of dimensions      
remaining = 1;                          % Counter to evaluate the end of dimensions
index = [];                             % Initialize index
    while remaining~=ndim+1                 % Break after the loop for the last dimension has been evaluated
       % Divide the integer with the the value of each dimension --> Identify at which group the integer belongs    
       r       = rem(I,neldim(remaining)); % The remainder identifies the index for the dimension under evaluation
       int     = fix(I/neldim(remaining)); % The integer is the number that has to be used for the next iteration
       if r == 0                           % Compensate for being the last element of a "group" --> It index is equal to the dimension under evaluation
           new_index   = neldim(remaining);
       else                                % Compensate for the number of group --> Increase by 1 (e.g if remainder 8/3 = 2 and integer = 2, it means that you are at the 2+1 group in the 2nd position)
           int         = int+1;
           new_index   = r;                    
       end
       I     = int;                        % Adjust the new number for the division. This is the group th
       index = [index new_index];          % Append the current index at the end       
       remaining = remaining + 1;
    end
