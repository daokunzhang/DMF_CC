function [ res ] = Update_For_H( M, W, H, T, lambda )

k = size(H,1);
ft = size(H,2);
h = H(:);
T2 = T*T';
G = W*(W'*(H*T2))-W*(M*T')+lambda*H;
r = - G(:);
d = r;
eps_ = 1.0e-12;
maxIter = 1000;
for iter = 1:maxIter
    D = reshape(d,k,ft);
    HD = W*(W'*(D*T2))+lambda*D;
    Hd = HD(:);
    r2 = r'*r;
    alpha = r2/(d'*Hd);
    h = h + alpha*d;
    r = r - alpha*Hd;
    beta = (r'*r)/r2;
    d = r + beta*d;
    if r'*r<eps_
        break;
    end
end
res = reshape(h,k,ft);
end