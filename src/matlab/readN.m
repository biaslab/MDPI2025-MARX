function [N, P, M, fMin, fMax, fs] = readN()

fileID = fopen('N.txt', 'r');
% Read 6 numbers from the file (N, P, M, fMin, fMax, fs)
data = textscan(fileID, '%f %f %f %f %f %f');

% Assign the values from the cell array to individual variables
N = data{1};
P = data{2};
M = data{3};
fMin = data{4};
fMax = data{5};
fs = data{6};

%fprintf('matlab: %d %d %d %d %d %d\n', N, P, M, fMin, fMax, fs);

fclose(fileID);

end
