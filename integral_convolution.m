t = -10:0.001:20;
tou = -10:0.01:20;

x_tau = unitstep(tou-3) - unitstep(tou-5);

for tx = 1:length(t)
    tt = t(tx);
    h_tou = my_ht(tt-tou);
    y(tx) = trapz(tou , x_tau .* h_tou);
end

y_th = zeros(size(t));
y_th(t <= 5 & t > 3) = (1-exp(9-3*(t(t<= 5 & t > 3))))/3;
y_th(t > 5) = ((1-exp(-6))*exp(15-3*(t(t > 5))))/3;

plot(t,y,'o');
hold on;
plot(t,y_th,'y','LineWidth',2);
