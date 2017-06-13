function [ res ] = Update_For_eta( X, Y, eta, lambda )

k=size(eta,1);
X2 = X'*X;
r = -(lambda*eye(k)+X2)*eta+X'*Y;
d = r;
max_iter = 1000;
eps_ = 1.0e-12;
H = lambda*eye(k)+X2;
for iter = 1:max_iter
    Hd = H*d;
    r2 = r'*r;
    alpha = r2/(d'*Hd);
    eta = eta + alpha*d;
    r = r - alpha*Hd;
    beta = r'*r/r2;
    d = r + beta*d;
    if r'*r<eps_
        break;
    end
end
res = eta;
end

