close all;
clear all;

%% source: https://github.com/biaslab/ACC2022-vmpNARMAX/blob/main/datasets/gen_input.m
%% Specify options

stdu = 0.1;
stde = .005;

options.na = 3; % # output delays
options.nb = 3; % # input delays
options.ne = 3; % # innovation delays
options.nd = 1; % # degree polynomial nonlinearity
%options.N = 2^16;
options.N = 200;
options.P = 1;
options.M = 1;
options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type =  'odd';

%% Generate

[uTrain, ~] = fMultiSinGen(options);
% uTrain

%% plot(uTrain)
csvwrite('data/inputs.csv', uTrain)
% uTrain