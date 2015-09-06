function [ sign ] = mysign( vector )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

sign = vector;
sign(vector>=0) = 1;
sign(vector<0) = -1;

end

