function [r,posterior,ent,score] = MAP_conditional_scaled(prior_conditional,prior_marginal,likelihood,rm)

NR = 50;
ent = -sum(prior_conditional.*log(prior_conditional+eps));
tol = 2e-08;
posterior = prior_conditional.*(likelihood.*(1./prior_marginal));
posterior = posterior./sum(posterior);
[score,r] = max(posterior);


if sum(posterior) == 0
    r = rm;
    posterior = likelihood;
    posterior = posterior./sum(posterior);
end
    
