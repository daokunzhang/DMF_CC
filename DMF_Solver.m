function [ W, H, eta, iter ] = DMF_Solver( M, T, k, labels, trainId, mu_, lambda1, lambda2 )
% Decompose matrix M as M = W^{T}HT
% M random walk matrix size |V|X|V|
% T text feature f_{t} X |V|
% W low-rank matrix k X |V|
% H low-raml matrix k X f_{t}
ft = size(T,1);
n = size(T,2);
W = rand(k,n);
H = rand(k,ft);
eta = rand(k+1,1);
max_iter = 50;
eps_ = 0.01;
mu_0 = 1.0e-5;
gamma = lambda1/mu_;
for iter = 1:max_iter
    iter
    eta0 = eta;
    H0 = H;
    W0 = W;
    W2 = [W;ones(1,n)];
    
    eta = Update_For_eta( W2(:,trainId)', labels(trainId,1), eta, gamma );
    H = Update_For_H( M, W, H, T, lambda1 );
    W = Update_For_W( M, W, H, T, eta, labels, trainId, mu_0, lambda2 );
    mu_0
    if mu_0>=mu_
        Delta_eta = eta-eta0;
        Delta_W = W-W0;
        Delta_H = H-H0;
        val = Delta_eta'*Delta_eta+trace(Delta_W*Delta_W')+trace(Delta_H*Delta_H');
        disp('val:');
        disp(val);
        if val < eps_
            break;
        end
    end
    mu_0 = min(mu_,mu_0*1.5);
end
end

