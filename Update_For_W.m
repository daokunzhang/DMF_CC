function [ res ] = Update_For_W( M, W, H, T, eta, labels, trainId, mu_, lambda )

k = size(W,1);
n = size(W,2);
W = W';
w = W(:);
HT = H*T;
HT2 = HT*HT';
G = W*HT2-M*HT'+lambda*W;
E = speye(n);
G = G+mu_*(E(trainId,:))'*((W(trainId,:)*eta(1:k,1)-labels(trainId,1))+eta(k+1,1))*eta(1:k,1)';
r = - G(:);
d = r;
max_iter = 1000;
eps_ = 1.0e-12;
for iter = 1:max_iter
    D = reshape(d,n,k);
    HD = D*HT2+mu_*(E(trainId,:)'*(D(trainId,:)*eta(1:k,1))*eta(1:k,1)')+lambda*D;
    Hd = HD(:);
    r2 = r'*r;
    alpha = r2/(d'*Hd);
    w = w + alpha*d;
    r = r - alpha*Hd;
    beta = (r'*r)/r2;
    d = r + beta*d;
    if r'*r<eps_
        break;
    end
end
res = reshape(w,n,k);
res = res';
end

