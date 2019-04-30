function rt = rect(t,A,a,b)
t1 = t-a;
ut1 = unitstep(t1);

t2 = t-b;
ut2 = unitstep(t2);

rt = A*(ut1-ut2);
return;