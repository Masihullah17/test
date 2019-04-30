Transformation  ---------------------------------------------------------------------
%%
nstart = -10;
nend = 10;
n = nstart:nend;
%%
%Plotting u[n] using unitstep function
u1 = unitstep(n);
subplot(221);
stem(n,u1,'filled','LineWidth',2);
xlabel('n'); ylabel('u[n]');
%%
%Plotting u[n-5]-u[n-10]
n21 = n-5;
u21 = unitstep(n21);
n22 = n-10;
u22 = unitstep(n22);
r1 = u21-u22;
subplot(222);
stem(n,r1,'filled','LineWidth',2);
xlabel('n'); ylabel('u[n-5]-u[n-10]');
%%
%Plotting u[6-n]-u[3-n]
n31 = 6-n;
u31 = unitstep(n31);
n32 = 3-n;
u32 = unitstep(n32);
r2 = u31 - u32;
subplot(223);
stem(n,r2,'filled','LineWidth',2);
xlabel('n'); ylabel('u[6-n]-u[3-n]');
%%
%Plotting u[8-n]
n41 = 8-n;
u41 = unitstep(n41);
subplot(224);
stem(n,u41,'filled','LineWidth',2);
xlabel('n'); ylabel('u[8-n]');




Discrete---------------------------------------------------------------------------
%%
%Initializing
n = [0 , 1 , 2 , 3 , 4];
x = [-1, -2, -3, 4, -2];
%%
%Plotting x[n]
subplot(321);
stem(n,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[n]');
%%
%Plotting x[n+1]
n1 = n-1;
subplot(322);
stem(n1,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[n+1]');
%%
%Plotting x[n-2]
n2 = n+2;
subplot(323);
stem(n2,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[n-2]');
%%
%Plotting x[3-n]
n3 = -n+3;
subplot(324);
stem(n3,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[3-n]');
%%
%Plotting x[3-2n]
n4 = -(n+3)/2;
subplot(325);
stem(n4,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[3-2n]');
%%
%Plotting x[4n+5]
n5 = (n-5)/4;
subplot(326);
stem(n5,x,'filled','LineWidth',2);
xlabel('n'); ylabel('x[4n+5]');


Duplicating the signals--------------------------------------------------------------------------
%%
nstart = 0;
nend = 3;
n = nstart:delta:nend;
subplot(212);
for i = 1:5
plot(n,x(t),'b','LineWidth',2);
n = n+3;
hold on
end
xlabel('t'); ylabel('y(t)');


Integral Convolution Function----------------------------------------------------------------------------------------
function y = convolution(x,h)
M = length(x);
N = length(h);
L = M+N-1;
Xe = zeros(1,L);
He = zeros(1,L);
Xe(1:M) = x;
He(1:N) = h;
X = zeros(L,L);
for n = 1:L
Xtemp = Xe(n:-1:1);
X(n,1:n) = Xtemp;
end
y = He * transpose(X);
return;




Integral Convolution----------------------------------------------------------------------------------
%%
%Defining Signals
x1 = [ 3, 1, 4, 16, 2];
x2 = [ 3, -1, 3, -1];
h = [ 2, -1, -4, 1, -3];
%%
%Calculating lengths of signals
M = max(length(x1),length(x2));
N = length(h);
L = M+N-1;
%%
%Calling convolution fucntion seperately to both x signals
y_1 = convolution(x1,h);
y_2 = convolution(x2,h);
%Adding seperately convoluted signals (adding 0 to make both of same length)
y1 = y_1 + [y_2,0];
%Convoluting using inbuilt function to verify
y_1_conv = conv(x1,h);
y_2_conv = conv(x2,h);
y1_conv = y_1_conv + [y_2_conv,0];
%%
%Adding both x signals to convolute after adding
X = x1 + [x2,0];
%Calling convolution function
y2 = convolution(X,h);
%Convoluting using inbuilt function to verify
y2_conv = conv(X,h);
%%
%Verifying my convolution function with inbuilt convolution function
nvec = 0:L-1;
figure();
plot(nvec,y1,'+','MarkerSize',10);
hold on;
plot(nvec,y1_conv,'s','MarkerSize',10);
xlabel('n');
ylabel('y1 & y1-conv');
legend('My-Function','Inbuilt-Function');
figure();
plot(nvec,y2,'+','MarkerSize',10);
hold on;
plot(nvec,y2_conv,'s','MarkerSize',10);
xlabel('n');
ylabel('y2 & y2-conv');
legend('My-Function','Inbuilt-Function');
%%
%Verifying both y1 and y2 are same
figure();
plot(nvec,y1,'+','MarkerSize',10);
hold on;
plot(nvec,y2,'s','MarkerSize',10);
xlabel('n');
ylabel('y1 & y2');
legend('y1[n] = (x1[n] + x2[n])*h[n]','y2[n] = x1[n]*h[n] + x2[n]*h[n]');
%%
%Plotting x and h signals
nx1 = 0:M-1;
nx2 = 0:length(x2)-1;
figure();
subplot(231);
stem(nx1,x1,'filled','LineWidth',2);
xlabel('n');
ylabel('x1[n]');
subplot(232);
stem(nx2,x2,'filled','LineWidth',2);
xlabel('n');
ylabel('x2[n]');
n2 = 0:N-1;
subplot(233);
stem(n2,h,'filled','LineWidth',2);
xlabel('n');
ylabel('h[n]');
%Plotting y signals
figure();
subplot(221);
stem(nvec,y1,'filled','LineWidth',2);
xlabel('n');
ylabel('y1[n]');
subplot(222);
stem(nvec,y2,'filled','LineWidth',2);
xlabel('n');
ylabel('y2[n]');


Shifted Discrete Convolution------------------------------------------------------------------------
%%
%Defining Signals
x = [ -3, -2, 0, 1, 2, 3];
h = [ 3, 1, 1, 3, 1, 1];
%Calculating lengths of signals
M = length(x);
N = length(h);
L = M+N-1;
%%
%Convoluting the signals
y = convolution(x,h)
%Using Inbuilt function
y_conv = conv(x,h)
%%
%Defining ranges
n = -3:M-4;
n1 = 0:N-1;
nvec = 0:L-1;
%%
%Verifying that both gives the same result
figure();
plot(nvec,y,'+','MarkerSize',10);
hold on;
plot(nvec,y_conv,'s','MarkerSize',10);
xlabel('n');
ylabel('y & y-conv');
legend('My-Function','Inbuilt-Function');
%%
%Plotting x,h signals
figure();
subplot(121);
stem(n,x,'filled','LineWidth',2);
xlabel('n');
ylabel('x[n-3]');
subplot(122);
stem(n1,h,'filled','LineWidth',2);
xlabel('n');
ylabel('h[n]');
%Plotting y signal
figure();
stem(nvec,y,'filled','LineWidth',2);
xlabel('n');
ylabel('y[n]');



Fourier Coefficients-------------------------------------------------------------------------------------
clear
%%
% Defining the signal
x = @(t) [t.^2 .* (t<1 & t>-1)];
%%
% Defining parameters
To = 3;
Wo = 2*pi/To;
tvec = -To/2:0.001:To/2;
%%
% Taking signal in the given time period
x_t = x(tvec);
%%
% Taking 100 samples
M=100;
for mx=1:M
% Calculating Fourier Series
kvec = -mx:mx;
for kx=1:length(kvec)
k=kvec(kx);
basis=exp(-1i*k*Wo*tvec);
avec(kx) = 1/To*trapz(tvec,x_t.*basis);
end
% Reconstructing the original signal
recon=zeros(size(tvec));
for kx=1:length(kvec)
k=kvec(kx);
basis=exp(1i*k*Wo*tvec);
recon=recon+avec(kx)*basis;
end
% Calculating the convergence
recon_err(mx) = mean((abs(recon-x_t)).^2);
end
%%
% Theoretical equation
a_th = 1/3*((2*sin(kvec*Wo)./(kvec*Wo)) ...
+ (4*cos(kvec*Wo)./(kvec.^2 * Wo.^2)) ...
- (4*sin(kvec*Wo)./(kvec.^3*Wo.^3)));
a_th(kvec == 0) = 0.22;
%%
% Plotting the coefficients and comparing with the theoretical values
figure();
subplot(211); stem(kvec,real(avec));
title('Real Component');
xlabel('k');
ylabel('real(a(k))');
hold on;
stem(kvec,real(a_th));
legend('Numerical','Theoretical');
subplot(212); stem(kvec,imag(avec));
title('Imaginary Component');
xlabel('k');
ylabel('imag(a(k))');
ylim([-0.1 0.5]);
hold on;
stem(kvec,imag(a_th));
legend('Numerical','Theoretical');
% Plotting the original vs constructed signals
figure();
plot(tvec,x_t);
xlabel('t');
ylabel('x(t)');
hold on;
plot(tvec,recon);
legend('Original','Reconstructed');
% Plotting the convergence
figure();
stem(1:M, recon_err);
xlabel('M');
ylabel('Error');
legend('Error');


Fourier Transform-----------------------------------------------------------------------------------------
clear;
%Defining parameters
tvec = -1:0.01:1;
W = -100:100;
%Defining Signal
x_t = tvec.^3 ;
%Fourier Transform
for i = 1:length(W)
basis = exp(-1i*W(i)*tvec);
X(i) = trapz(tvec,x_t.*basis);
end
%Inverse Fourier Transform
for i = 1:length(tvec)
basis1 = exp(1i*W*tvec(i));
Rec(i) = (1/(2*pi))*trapz(W,basis1.*X);
end
%Theoretical approach
X_th = 2*1i*cos(W)./(W) ...
- 6*1i*sin(W)./(W).^2 ...
-12*1i*cos(W)./(W).^3 ...
+12*1i*sin(W)./(W).^4;
X_th(W == 0) = 0;
%Plotting the outputs
figure();
subplot(211);
stem(W,real(X),'o');
hold on;
stem(W,real(X_th),'+');
ylim([-1 1]);
xlabel('\omega');
ylabel('Real');
legend('Numerical','Theoretical');
subplot(212);
stem(W,imag(X),'o');
hold on;
stem(W,imag(X_th),'+');
xlabel('\omega');
ylabel('Imaginary');
legend('Numerical','Theorotical');
figure();
plot(tvec,x_t);
hold on;
plot(tvec,Rec);
xlabel('t');
ylabel('x(t)');
legend('Original','Reconstructed');

SINC Fourier Transform-------------------------------------------------------------------------
clear;
%Defining parameters
tvec = -2*pi:pi/100:2*pi;
W = -200:200;
%Defining Signal
x_t = sinc(tvec) ;
%Fourier Transform
X = fftshift(fft(x_t));
%Inverse Fourier Transform
Rec = ifft(ifftshift(X));
%Plotting the outputs
figure();
subplot(211);
plot(W,abs(real(X)));
xlabel('\omega');
ylabel('Real');
legend('Numerical');
subplot(212);
plot(W,abs(imag(X)));
xlabel('\omega');
ylabel('Imaginary');
legend('Numerical');
figure();
plot(tvec,x_t);
hold on;
plot(tvec,Rec);
xlabel('t');
ylabel('x(t)');
legend('Original','Reconstructed');


SAMPLING-------------------------------------------------------------------------------------------
clear;close all;
%Initializing the parameters
deltaT = 100;
t = 0:1/deltaT:2;
W = -5:0.01:5;
%Initializing the signal
x_t = (t-1).^2;
L = length(t); %Finding the length of time domain
%Calculating the fourier transform of the signal
Xw = zeros(size(W));
for iw = 1:length(W)
basis = exp(-1i*W(iw)*t);
Xw(iw) = trapz(t,x_t.*basis);
end
%Finding the -3db frequency value
mx = max(abs(Xw))/(sqrt(2));
mxi = find((abs(Xw) >= mx), 1 , 'last');
f1 = W./(2*pi);
%Plotting
figure();
plot(f1,Xw);
hold on;
stem(abs(f1(mxi)),Xw(mxi));
title('Approximate Bandwidth');
xlabel('W');
ylabel('X[W]');
legend('Signal','Wm');
%Calcuating the approximate bandwidth
Bandwidth = abs(f1(mxi));
Bw = sprintf('\n Wm = %.2fHz\n\n',Bandwidth);
text(f1(mxi),Xw(mxi),Bw);
%Calculating the Nyquist Rate
Nr = 2*Bandwidth;
fprintf('\n Wm = %.2fHz\n\n',Bandwidth);
fprintf(' Nyquist Rate is %.2fHz\n\n',Nr);
%Sampling at Nyquist Rate
figure();
plot(t,x_t,'linewidth',2);
set(gca,'Box','on',....
'FontSize',12,....
'FontWeight','bold',....
'LineWidth',1.5,....
'FontName','Helveltica',....
'Color',[0.95,0.95,0.95],....
'XGrid','off',....
'YGrid','off');
Ns = round(1/Nr*deltaT); %Nyquist rate
sampling = 1:Ns:L;
x_s = zeros(size(x_t));
x_s(x_s==0)=NaN;
x_s(sampling)= x_t(sampling);
hold on;
stem(t,x_s,'r','LineWidth',2);
title('Sampling');
xlabel('t');
ylabel('x(t)');
legend('Analog:Non Zero Phase','Sampling at Nyquist Rate');


SINC Sampling-------------------------------------------------------------------------------------------
clear; close all;
%Initializing the parameters
deltaT = 100;
t = -2*pi:pi/deltaT:2*pi;
W = -10:0.01:10;
%Initializing the signal
x_t = sinc(t/pi).^2;
L = length(t); %Finding the length of time domain
%Calculating the fourier transform of the signal
Xw = zeros(size(W));
for iw = 1:length(W)
basis = exp(-1i*W(iw)*t);
Xw(iw) = trapz(t,x_t.*basis);
end
f1 = W./(2*pi);
%Finding the -3db frequency value
mxi = find((abs(Xw(W >= 0)) <= 0.001),1, 'first') + length(Xw(W < 0));
%Plotting
figure();
plot(f1,Xw);
hold on;
stem(f1(mxi),Xw(mxi));
title('Approximate Bandwidth');
xlabel('W');
ylabel('X[W]');
legend('Signal','Wm');
%Calcuating the approximate bandwidth
Bandwidth = abs(f1(mxi));
Bw = sprintf('\n Wm = %.2fHz\n\n',Bandwidth);
text(f1(mxi),Xw(mxi),Bw);
%Calculating the Nyquist Rate
Nr = 2*Bandwidth;
fprintf('\n Wm = %.2fHz\n\n',Bandwidth);
fprintf(' Nyquist Rate is %.2fHz\n\n',Nr);
%Sampling at Nyquist Rate
figure();
plot(t,x_t,'linewidth',2);
set(gca,'Box','on',....
'FontSize',12,....
'FontWeight','bold',....
'LineWidth',1.5,....
'FontName','Helveltica',....
'Color',[0.95,0.95,0.95],....
'XGrid','off',....
'YGrid','off');
Ns = round(1/Nr*deltaT); %Nyquist rate
sampling = 1:Ns:L;
x_s = zeros(size(x_t));
x_s(x_s==0)=NaN;
x_s(sampling)= x_t(sampling);
hold on;
stem(t,x_s,'r','LineWidth',2);
title('Sampling');
xlabel('t');
ylabel('x(t)');
legend('Analog:Non Zero Phase','Sampling at Nyquist Rate');


CIRCULAR CONVOLUTION---------------------------------------------------------------------------------
clear;
%%
%Initialzing the signals
n = 0:7;
L = length(n);
x1_n = (1/4).^n;
x2_n = cos((3*pi*n)/8);
%%
%Circular Convolution
cir_conv = zeros(1,L);
for i = 1:L
for j = 1:L
cir_conv(i) = cir_conv(i) + x1_n(j).*x2_n(mod(i-j,L)+1);
end
end
%Plotting
figure();
stem(n,cir_conv);
title('Using Shifting method');
xlabel('n');
ylabel('x[n]');
fprintf('Using Shifting method : \n\n');
disp(cir_conv);
%%
%DFT and IDFT Method
lx1=length(x1_n);
lx2=length(x2_n);
N=max(lx1,lx2);
%DFT
x1=[x1_n zeros(N-lx1)];
x2=[x2_n zeros(N-lx2)];
W=zeros(N,N);
for m=0:N-1
for k=0:N-1
W(m+1,k+1)=exp(-1i*2*pi*m*k/N);
end;
end;
X1=W*x1.';
X2=W*x2.';
%IDFT
Y1=X1.*X2;
w=zeros(N,N);
for m=0:N-1
for k=0:N-1
w(m+1,k+1)=exp(1i*2*pi*m*k/N);
end;
end;
B=w*Y1;
Y=B/N;
%Plotting
figure();
stem(0:N-1,Y);
title('Using DFT & IDFT method');
xlabel('n');
ylabel('x[n]');
fprintf('Using DFT and IDFT :\n\n');
disp(real(Y.'));
%%
%Comparision
figure();
plot(n,cir_conv,'+','MarkerSize',10);
hold on;
plot(0:N-1,real(Y.'),'s','MarkerSize',10);
title('Comparing the both methods');
xlabel('n');
ylabel('x[n]');
legend('Using Shifting method','Using DFT & IDFT method');