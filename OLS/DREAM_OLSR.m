function [out] = DREAM_OLSR(data, options)
% This code aims at solving the DREAM for OLSR
%     \min_{Y,X_i} \sum_{i=1}^d ||A_i^T*X_i-B_i^T||_F^2 + mu/4*||Y'Y-I_p||_F^2
%     s.t. ||Y||_{2,0} <=s,
% where Y, X_i\in\R^{n\times p}, s<<n is an given integer
%       ||Y||_{2,0} is the number of nonzero rows of Pv
%
% Inputs: 
%        A\in R^{n \times m}: m and n are the number of samples and features, respectively
%        B\in R^{p \times m}  
%        options  -- a structure
%                 options.s         -- the sparsity level, an integer in (0,n)      (required)
%                 options.maxiter   -- the maximum number of iterations
%                 options.mu        -- the tuning parameters
%                 options.num_block -- the number of agents
%                 options.beta      -- the penalty parameter
%                 options.tol       -- the default stopping error
%
% Outputs: 
%       out.Y           -- the sparse solution
%       out.X           -- the auxiliary variables
%       out.idx_block   -- the label for sample segmentation
%       out.obj         -- objective function values
%       out.iter        -- number of iterations
%       out.Error_Y     -- error1
%       out.Error_obj   -- error2
%       out.time        -- runing time

%% get data sequence
A = data.A;
B = data.B;

%% Initialization
if isfield(options,'maxiter');   maxiter   = options.maxiter;   else; maxiter   = 1e3;  end
if isfield(options,'beta');      beta      = options.beta;      else; beta      = 1e3;  end
if isfield(options,'mu');        mu        = options.mu;        else; mu        = 2^5;  end
if isfield(options,'num_block'); num_block = options.num_block; else; num_block = 10;   end
if isfield(options,'s');         s         = options.s;         else; s         = 10;  end
if isfield(options,'tol');       tol       = options.tol;       else; tol       = 1e-4; end

[n,m] = size(A);
[p,~] = size(B);
Ip = eye(p); In = eye(n); Nmaxit=4;

X0      = normrnd(0,1,[n,p]);
Y       = normrnd(0,1,[n,p]);
Lambda0 = zeros(n,p);

%% Defining Functions
f_loss = @(X,A,B) (norm(A'*X-B','fro'))^2/(2*n);
g_loss = @(X) (mu/4*(norm(X'*X-Ip,'fro'))^2);
Fnorm   = @(var)norm(var,'fro')^2;

%% assign data
idx_data_shuffled = randperm(m); % randomly permute the data for assignment to blocks
size_data_block = floor(m/num_block); X = {}; Lambda = {};
for i = 1:num_block
    idx_block{i} = idx_data_shuffled(((i-1)*size_data_block+1):(i*size_data_block));
    A_block{i} = A(:,idx_block{i});         B_block{i} = B(:,idx_block{i});    
    X{i} = X0;                              Lambda{i} = Lambda0;
end

fprintf(' Run solver DSMNAL---------------------------------%5d\n',num_block);
fprintf('Iter\t  ObjVal\t error_obj\t  error_Y\n');

tic;
for iter = 1:maxiter
    %% update Y by NHTP
    Y_old        = Y;
    pars.maxit   = Nmaxit;
    pars.Y0      = Y;
    pars.mu      = mu;
    pars.beta    = beta;
    pars.X0      = X;
    pars.Lambda0 = Lambda;

    Y  = NHTP(n,p,s,num_block,pars); 

    %% update X and Lambda
    f_cost = 0;
    for j = 1:num_block
        A_local = A_block{j};	B_local = B_block{j};	Lambda_local = Lambda{j};
        X{j} = pinv(A_local*(A_local')+beta* In)*(A_local*(B_local') + beta*Y + Lambda_local);      % update X
        Lambda{j} = Lambda_local - beta*(X{j}-Y);                                                   % update Lambda
        f_cost = f_cost+ f_loss(X{j},A_local,B_local)/(2*num_block);   
    end
    obj(iter) = f_cost + g_loss(Y);                                                                 % calculate the loss function
    
    %%  check the stop criteria
    if iter>1
        error_obj(iter) = Fnorm(obj(iter)-obj(iter-1))/Fnorm(1+abs(obj(iter-1)));      % error_obj
        error_Y(iter)   = Fnorm(Y-Y_old)/(1+Fnorm(Y_old));                      % error_Y
        fprintf('%5d\t  %6.2e\t %6.2e\t  %6.2e\n',iter, obj(iter), error_obj(iter), error_Y(iter));
        if (max([error_obj(iter),error_Y(iter)]) <tol); break; end
        if iter == maxiter; fprintf('The number of iterations reaches maxiter.\n'); end
    end
end
time = toc;
out.Y           = Y;
out.X           = X;
out.idx_block   = idx_block;
out.obj         = obj;
out.iter        = iter;
out.Error_Y     = error_Y;
out.Error_obj   = error_obj;
out.time        = time;
end