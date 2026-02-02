function T2 = moment(seq,Pattern)
n = length(seq);
ngrp = numel(Pattern);

T2 = zeros(1,ngrp);
% D = zeros(1,ngrp);
N=zeros(1,ngrp);
Mu=zeros(1,ngrp);
W = zeros(ngrp,n);

for i = 1 : n
    for j = 1 : ngrp
        if any(ismember(Pattern{j},seq(i)))
           W(j,i) = 1;
           break;
        end
    end   
end
i = 1 : n;
for r = 1 : ngrp
    N(r) = sum(W(r,i));
    Mu(r) = sum(i.*W(r,i)/N(r));
    D(r) = sum((i-Mu(r)).^2.*W(r,i)./(N(r)*n));
end
T2 = D;
T2(isnan(D))=0;
end