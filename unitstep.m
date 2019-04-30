function ut=Unitstep(t)
    ut = zeros(size(t));
    ut(t>=0) = 1;
return;

