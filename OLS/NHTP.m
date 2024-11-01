function [out_Y,out] = NHTP(n,p,s,num_block,pars)
%This code aims at solving the sparsity constrained optimization
%     min_Y mu/4*||Y'Y-I_p||_F^2 + \sum_{i=1}^d beta_i/2*||X_i-Y-Lambda_i/beta_i||_F^2
%     s.t. ||Y||_{2,0}<=s,
% where Y\in\R^{n x p}, 
%       s <<n is an given integer
%       ||Y||_{2,0} is the number of nonzero rows of Y
%       X_i\in R^{n x d}  
%       Lambda_i\in R^{n x d}
% Inputs: 
%        funch -- a function handle computing                  (required)
%                   (objective, gradient, sub-hessian)
%       

%        s     -- the sparsity level, an integer in (0,n)      (required)
%        pars  -- a structure
%                 pars.maxit     = 100         (default)
%                 pars.Y         = zeros(n,p)  (default)
%                 pars.mu        = 2^5         (default)
%                 pars.beta      = 1e3         (default)
%                 pars.tol       = 1e-5        (default)

% Outputs: 
%       Y         -- the sparse solution
%       out.time  -- computational time  
%       out.iter  -- number of iterations

t0    = tic;
if nargin < 5; pars = []; end
if isfield(pars,'maxit');     maxit     = pars.maxit;     else; maxit     = 1e2;        end
if isfield(pars,'Y0');        Y0        = pars.Y0;        else; Y0        = zeros(n,p); end
if isfield(pars,'mu');        mu        = pars.mu;        else; mu        = 2^5;         end
if isfield(pars,'beta');      beta      = pars.beta;      else; beta      = 1e3;        end
if isfield(pars,'tol');       tol       = pars.tol;       else; tol       = 1e-5;       end

eta    = 1e-4;
sigma  = 1e-8;
gamma  = 1e-10;
Y      = Y0;
X      = pars.X0;
Lambda = pars.Lambda0;
Ip = eye(p); Is = eye(s);

Fnorm  = @(var)norm(var,'fro')^2;

[h0, g] = funch(Y,X,Lambda,num_block,beta,mu);                         % Calculate the gradient and function values of g
[~,Tu]  = maxk(sum((Y - eta * g).^2,2),s,'ComparisonMethod','abs');    % Find the support indices Tu
YT      = Y(Tu,:);     

% fprintf('\n Start to run the solver -- NHTP\n')
% fprintf(' ----------------------------------------------------\n')
% fprintf(' Iter         Error          Objective         Time\n')
% fprintf(' ----------------------------------------------------\n')

%% main body
for iter = 1 : maxit
        Y_old     = Y;

        h_grad_YT  = g(Tu,:);                                                           % Compute (\nabla h(Y))_Tu
        err_newton = Fnorm(h_grad_YT);
        if err_newton < tol; break; end                                                 % check the stop criteria of NHTP

        P = transpose_derivative(YT);
        Kron  = kron(Ip,YT* (YT')); 
        Kron2 = kron((YT')*YT,Is); 
        Kron3 = kron((YT'),YT)*P; 
        nb_beta_mu = num_block * beta- mu;
        H_grad_1   = mu * (Kron+Kron2+ Kron3)+ nb_beta_mu* eye(s*p);                    % calculate the hessian of h(Y)
        
        vec_h_grad_YT = reshape(h_grad_YT,[],1);      
        vec_D_Tu = (H_grad_1)\(-vec_h_grad_YT );                                        % solve the Newton equation   

        D_Tu   = reshape(vec_D_Tu,[s,p]);                                               % update the search direction D
        temp1  = max(0,Fnorm(Y)-Fnorm(YT)); 
        marker = (trace(D_Tu'*h_grad_YT) >= -gamma*(Fnorm(D_Tu)) + temp1/4/eta );
        if Fnorm(D_Tu) > 1e16 || marker
            D_Tu = - h_grad_YT;
        end
        D         = -Y;
        D(Tu,:)   = D_Tu;

        % Armijio line search
        alpha  = 1;
        Y      = zeros(n,p);
        temp2  = sigma * trace(g'*D);  
        for mm = 1:8
            Y(Tu,:)         = Y_old(Tu,:) + alpha * D_Tu;
            [h_loss_new, g] = funch(Y,X,Lambda,num_block,beta,mu);
            if h_loss_new   <= (h0+alpha*temp2);  break; end
            alpha           = alpha/2;
        end

         
        [~,Tu] = maxk( sum((Y - eta * g).^2,2),s,'ComparisonMethod','abs'); % find the support indices Tu
        YT     = Y(Tu,:);

        if mod(iter,5)==0; eta = max(eta/2,1e-5); end
    end

% fprintf(' ----------------------------------------------------\n')

out.iter = iter;
out.time = toc(t0);
out_Y       =  Y;
end