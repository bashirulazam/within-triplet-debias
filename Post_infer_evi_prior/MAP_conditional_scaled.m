function [r,posterior,ent,score] = MAP_conditional_scaled(prior_conditional,prior_marginal,likelihood,rm)

NR = 50;
ent = -sum(prior_conditional.*log(prior_conditional+eps));
tol = 2e-08;
%if abs(ent - log(NR)) > tol
    %This is for combining the posterior with the scaled likelihood
    posterior = prior_conditional.*(likelihood.*(1./prior_marginal));
    %This is only for prior network
    %posterior = prior_conditional;
    %This is combining posterior with likelihood
    %posterior = prior_conditional.*likelihood;
    %This is only the debiasing of the measurement
    %posterior = likelihood.*(1./prior_marginal);
    posterior = posterior./sum(posterior);
    [score,r] = max(posterior);
    
%else
    %[score,r] = max(likelihood);
%end
if sum(posterior) == 0
    r = rm;
    posterior = likelihood;
end
    
