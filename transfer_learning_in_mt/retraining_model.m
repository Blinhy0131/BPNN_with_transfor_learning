clc
clear
close all

%load the model had been train
load net

fs=1/1e2;
x=0:fs:10;
y=sin(2*x);

%use the model had been trained and train again
net=train(net,x,y,'useGPU','yes');
y_pred=net(x,'useGPU','yes');

plot(x,y_pred,'.')
