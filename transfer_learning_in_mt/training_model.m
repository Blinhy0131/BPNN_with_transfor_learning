clc;
clear;
close all

fs=1/1e2;
x=0:fs:10;
y=sin(2*x);

%define the net
hiddenSize=[100 100 100 100] ;
trainFac='trainscg';
%name the net as net
net=feedforwardnet(hiddenSize,trainFac);
net=train(net,x,y,'useGPU','yes');
y_pred=net(x,'useGPU','yes');
%save the net name as "net.mat"
save net

figure(1)
hold on
plot(x,y)
plot(x,y_pred,'.')
hold off
grid on