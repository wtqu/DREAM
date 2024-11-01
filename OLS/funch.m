function [h_loss, h_grad] = funch(Y,X,Lambda,num_block,beta,mu)
% Calculate the gradient and function values of h

[~,p] = size(Y);
Ip = eye(p);
gh_grad  = 0; h_loss_2 = 0;

for i = 1:num_block                                                               % Calculate the gradient and function values of h
    X_local  = X{i};     Lambda_local = Lambda{i};
    gh_grad  = gh_grad - beta*(X_local-Y-Lambda_local/beta);
    h_loss_2 = h_loss_2 + beta/2*(norm(X_local-Y-Lambda_local/beta,'fro')^2);
end
h_grad = mu*Y*((Y')*Y-Ip) + gh_grad;                                              % gradient of h
h_loss = mu/4*(norm(Y'*Y-Ip,'fro'))^2 + h_loss_2/2;                               % function values of h
end