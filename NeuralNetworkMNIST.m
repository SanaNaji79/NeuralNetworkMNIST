%% README
%here is a code for clustering MNIST dataset through a neural network
%% the main class
clear 
clc
load 'C:\Users\Sana\OneDrive\Desktop\semester\neuroscience_karbalaee\homework\HW3_Neuroscience\data.mat' ; % the MNIST data set
% separating training and test data
X_train = zeros(300*10 , 400) ;
X_test = zeros(200*10 , 400) ;
y_train = zeros(300*10 , 1) ;
y_test = zeros(200*10 , 1) ;
for i = 1:10
    X_train((i-1)*300 + 1 : i*300 , :) = X((i-1)*500 + 1 : (i-1)*500 + 300 , :) ;
    y_train((i-1)*300 + 1 : i*300) = y((i-1)*500 + 1 : (i-1)*500 + 300) ;
    X_test((i-1)*200 + 1 : i*200 , :) =  X((i-1)*500 + 301 : (i)*500 , :) ;
    y_test((i-1)*200 + 1 : i*200) = y((i-1)*500 + 301 : (i)*500) ;
end

% 
lambda = 1 ;
input_layer_size = 400 ;
hidden_layer_size = 25 ;
output_layer_size = 10 ;
initial_nn_params = initial(input_layer_size , hidden_layer_size , output_layer_size) ;
costFunction = @(p) CostFunction(p, input_layer_size , hidden_layer_size , X_train , y_train , lambda) ;
options = optimset('MaxIter', 50) ;
[nn_params , costt] = fmincg(costFunction , initial_nn_params , options) ;

% reshaping weigths
Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size + 1)) , [hidden_layer_size , (input_layer_size + 1)]);
Theta2 = reshape(nn_params((1 + (hidden_layer_size*(input_layer_size + 1))):end) , [output_layer_size , (hidden_layer_size + 1)]);

%% examination
X_test1 = [ones(2000 , 1) , X_test] ;
hiddenlayer_output = sigmoid(Theta1*transpose(X_test1(1 , :))) ;
alpha = [1 ; hiddenlayer_output] ;
alpha2 = sigmoid(Theta2*alpha) ;
y_result_1 = zeros(size(y_test)) ;
y_result_2 = zeros(size(y_test)) ;
errors = zeros(size(y_test)) ;
for i = 1:2000
    alpha = [1 ; sigmoid(Theta1*transpose(X_test1(i , :)))] ;
    alpha2 = sigmoid(Theta2*alpha) ;
    [y_result_1(i) , y_result_2(i)] = max(alpha2) ;
    errors(i) = y_result_2(i) - y_test(i) ;
end

accuracy = 0 ;

for i = 1:2000
    if errors(i) == 0
        accuracy = accuracy + 1 ;
    end
end
accuracy = 100*accuracy/2000 ;
figure ;
for i = 1:36
    j = randi(2000 , 1) ;
    b = y_result_2(j) ;
    if b == 10
        b = 0;
    end
    subplot(6 , 6 , i) ;
    p = reshape(X_test(j , :) , [20 , 20]) ;
    imshow(p) ;
    title(['detected: ' , num2str(b)]) ;
end

%% the random image

random = randperm(5000 , 100) ;
new_data = zeros(100 , 400) ;
number = zeros(100 , 1) ;
for i = 1:100
    new_data(i , :) = X(random(i) , :) ;
    number(i) = y(random(i)) ;
end
picture = zeros(200 , 200) ;
for i = 1:100
    a = mod(i , 10) ;
    if a == 0
        a = 10 ;
    end
    for j = 1:400
        b = mod(j , 20) ;
        if b == 0
           b = 20 ; 
        end
        picture(20*(a-1) + b , 20*fix((i-1)/10) + fix((j-1)/20)+1) = new_data(i , j) ;
    end
end
imshow(picture) ;
title('100 random data') ;
%%
function [out] = one_hot(y , k)
s1 = length(y) ;
out = zeros(k , s1) ;
for i = 1:s1
    out(y(i) , i) = 1 ;
end
end

function [out] = sigmoid(x)
out = 1 ./ (1 + exp(-x)) ;
end

function [out] = sigmoidGrad(x)
out = sigmoid(x).*(1 - sigmoid(x)) ;
end

function [parameter] = initial(input_layer , hidden_layer , k) 
e = 00.12 ;
theta1 = rand(hidden_layer , input_layer + 1)*2*e-e ;
theta2 = rand(k , hidden_layer + 1)*2*e-e ;
parameter = [reshape(theta1 , [(input_layer+1)*(hidden_layer) , 1]) ; reshape(theta2 , [(hidden_layer+1)*(k) , 1])] ;
end

function [cost , grad] = CostFunction(parameter , input_layer , hidden_layer , x , y , lambda)
k = 10 ;
m = length(y) ;
theta1 = reshape(parameter(1 : ((input_layer+1)*(hidden_layer))) , [hidden_layer , input_layer+1]) ;
theta2 = reshape(parameter((input_layer+1)*(hidden_layer)+1 : end) , [k , hidden_layer+1]) ;
a1 = [ones(m , 1) , x] ;
c1 = a1 ;
a2 = sigmoid(theta1*transpose(a1)) ;
a2 = [ones(m , 1) , transpose(a2)] ;
h = sigmoid(theta2*transpose(a2)) ;
y_1 = one_hot(y , k) ;

j1 = (sum(sum((-y_1.*log(h))-((1-y_1).*log(1-h)))))/m ;
j2 = sum(sum((theta1(: , 2:end)).^2))/(2*m) ;
j3 = sum(sum((theta2(: , 2:end)).^2))/(2*m) ;

cost = j1 + lambda*(j2 + j3) ;
theta1_grad = zeros(size(theta1)) ; % 25*401
theta2_grad = zeros(size(theta2)) ; % 10*26

for i = 1:m
    a1 = c1(i , :) ; % 1*401
    z2 = theta1*transpose(a1) ; % (25*401)(401*1) = 25*1
    a2 = sigmoid(z2) ; % 25*1
    a2 = [1 ; a2] ; % 26*1
    z3 = theta2*a2 ; % (10*26)(26*1) = 10*1
    a3 = sigmoid(z3) ; % 10*1
    
    z2 = [1 ; z2] ; % 26*1
    d3 = a3 - y_1(: , i) ; % 10*1
    d2 = (transpose(theta2)*d3).*(sigmoidGrad(z2)) ; % (26*10)(10*1)=26*1 , (26*1)
    d2 = d2(2:end) ; % 25*1
    
    theta1_grad = theta1_grad + (d2*a1) ; % (25*401) , (25*1)(1*401)=25*401
    theta2_grad = theta2_grad + (d3*(transpose(a2))) ; % (10*26) , (10*1)(1*26)=10*26
end

theta1_grad = theta1_grad/m ;
theta2_grad = theta2_grad/m ;

theta1_grad(: , 2:end) = theta1_grad(: , 2:end) + theta1(: , 2:end)*lambda/m ;
theta2_grad(: , 2:end) = theta2_grad(: , 2:end) + theta2(: , 2:end)*lambda/m ;

grad = [reshape(theta1_grad , [(input_layer+1)*(hidden_layer) , 1]) ; reshape(theta2_grad , [(hidden_layer+1)*(k) , 1])] ;

end

function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates
% within a few iterations, it could be an indication that the function value
% and derivatives are not consistent (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad 
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
% 
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
% 
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
  argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!
  
  X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  X = X + z1*s;                                             % begin line search
  [f2 df2] = eval(argstr);
  i = i + (length<0);                                          % count epochs?!
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
  if length>0, M = MAX; else M = min(MAX, -length-i); end
  success = 0; limit = -1;                     % initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
      limit = z1;                                         % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
      end
      if isnan(z2) || isinf(z2)
        z2 = z3/2;                  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
      z1 = z1 + z2;                                           % update the step
      X = X + z2*s;
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                           % count epochs?!
      d2 = df2'*s;
      z3 = z3-z2;                    % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
      break;                                                % this is a failure
    elseif d2 > SIG*d1
      success = 1; break;                                             % success
    elseif M == 0
      break;                                                          % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
      if limit < -0.5                               % if we have no upper limit
        z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2;                                   % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
      z2 = (limit-z1)/2;                                               % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
      z2 = z1*(EXT-1.0);                           % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
    [f2 df2] = eval(argstr);
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d2 = df2'*s;
  end                                                      % end of line search

  if success                                         % if line search succeeded
    f1 = f2; fX = [fX' f1]';
    fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    d2 = df1'*s;
    if d2 > 0                                      % new slope must be negative
      s = -df1;                              % otherwise use steepest direction
      d2 = -s'*s;    
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed || i > abs(length)          % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    s = -df1;                                                    % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);                     
    ls_failed = 1;                                    % this line search failed
  end
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
end
fprintf('\n');
end