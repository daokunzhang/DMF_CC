function [W, H, time] = IMC(R, X, Y, k, lambda, maxiter, WInit, HInit)
% 
% Inductive Matrix Completion (IMC) using squared loss: 
%
%		min_{W,H} \| R - X * W' * H * Y' \|_F^2 + \lambda * (\|W\|_F^2 + \|H\|_F^2 )
%
% !!! Note that this version treats the unknown/missing entries in R as zeros.  !!!
% !!! For other settings/loss functions, please use the Matlab wrapper from the !!!
% !!! C/C++ version of IMC.                                                     !!!
%
%	Usage:
%		[W, H, time] = IMC(R, X, Y, k, lambda, maxiter, WInit, HInit)
%
%	Input:
%		R:		 user-item matrix (ratings/associations). m x n.
%		X:		 user (row) features. m x d1.
%		Y:		 item (column) features. n x d2.
%		k:		 rank of the linear model Z = WH^T.
%		lambda:	 regularization parameter in the objective.
%		maxiter: maximum number of iterations to run Alternating Minimization.
%		WInit:	 W factor matrix initialization. d1 x k.
%		HInit:	 H factor matrix initialization. d2 x k.
%
%	Output:
%		Factor matrices W (k x d1) and H (k x d2).
%
%
% For any questions and comments, please send your email to
% Nagarajan Natarajan <naga86@cs.utexas.edu> or Donghyuk Shin <dshin@cs.utexas.edu>.
%
	
	[m, d1] = size(X);
	[n, d2] = size(Y);
	assert(size(R,1) == m & size(R,2) == n);
	W = []; H = [];
	if (~exist('WInit') | isempty(WInit))
		W = rand(d1, k);	
	else
		assert(size(WInit,1) == d1 & size(WInit,2) == k);
		W = WInit;
	end
	if (~exist('HInit') | isempty(HInit))
		H = rand(k, d2);	
	else
		assert(size(HInit,1) == d2 & size(HInit,2) == k);
		H = HInit'; %% Transposing is required. See below.
	end
	hessparams.lambda = lambda;
	time = cputime();
    %obj = sum(sum((R-X*W*H*Y').*(R-X*W*H*Y')))+lambda*(sum(sum(W.*W))+sum(sum(H.*H)));
    %disp('obj:');
    %disp(obj);
    eps_ = 0.01;
	for i=1:maxiter
        W0 = W;
        H0 = H;
		%% Fix H and update W.
		fprintf('Iter %d. Updating W. \n', i);
		H2 = Y * H';
		M = H2' * H2;
		Gradw = X'*( (X*W)*M - R*H2 ) + lambda * W;
		hessparams.X = X;
		hessparams.M = M;
		hessparams.d = d1;
        hessparams.k = k;
        w = CGD(W(:), Gradw(:), hessparams, maxiter);
        W = reshape(w, d1, k);
        
        %obj = sum(sum((R-X*W*H*Y').*(R-X*W*H*Y')))+lambda*(sum(sum(W.*W))+sum(sum(H.*H)));
        %disp('obj:');
        %disp(obj);
        %% Fix W and update H.
        fprintf('Updating H.\n');
        H2 = X * W;
        M = Y' * Y;
        Gradh = H2'*((H2*H)*M - R*Y) + lambda * H;
        hessparams.X = H2;
        hessparams.M = M;
        hessparams.d = k;
        hessparams.k = d2;
        h = CGD(H(:), Gradh(:), hessparams, maxiter);
        H = reshape(h, k, d2);
        %obj = sum(sum((R-X*W*H*Y').*(R-X*W*H*Y')))+lambda*(sum(sum(W.*W))+sum(sum(H.*H)));
        %disp('obj:');
        %disp(obj);
        
        Delta_W = W-W0;
        Delta_H = H-H0;
        val = trace(Delta_W'*Delta_W)+trace(Delta_H*Delta_H');
        disp('val:');
        disp(val);
        if  val < eps_
            break;
        end
    end
	time = cputime() - time;
	W = W';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [w] = CGD(winit, gradw, hessparams, maxiter)
%
%	Input:
%		winit: starting value of w.
%		gradw: gradient of the objective at winit.
%		hessparams: Params required for constructing Hessian matrix.
%		maxiter: Max. number of CG iterations.
%
	r = -gradw;
	d = r;
	%tol = 1e-6;
	w = winit;
    eps_ = 1.0e-12;
    for t=1:maxiter
        %if norm(r) <= tol, break; end
        r2 = r' * r;
        A = hessparams.X * reshape(d, hessparams.d, hessparams.k);
        XAM = hessparams.X' * (A * hessparams.M);
        hd = XAM(:) + hessparams.lambda * d;
        alpha = r2/(d'*hd);
        w = w + alpha * d;
        r = r - alpha * hd;
        beta = r'*r / r2;
        d = r + beta * d;
        if r'*r<eps_
            break;
        end
    end
end

