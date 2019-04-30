x = @(t) [1 .* (t<5 & t>3) + 0 .*(t>=5 & t<=3)];
h = @(t) [exp(-3 .* t).*(t>=0)];

t = -10:0.001:20;
tou = -10:0.01:20;

for tx = 1:length(t)
    tt = t(tx);
    y(tx) = trapz(tou , x(tou) .* h(tt - tou));
end
y_th = zeros(size(t));
y_th(t <= 5 & t > 3) = (1-exp(9-3*(t(t<= 5 & t > 3))))/3;
y_th(t > 5) = ((1-exp(-6))*exp(15-3*(t(t > 5))))/3;

plot(t,y,'o');
hold on;
plot(t,y_th,'y','LineWidth',2);