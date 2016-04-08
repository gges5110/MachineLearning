clear; home;

nSamples = 10000;
%% INITIALIZE THE GIBBS SAMPLER
% First bivariate normal parameters
muX = 2;   varX = 1;  corrXY = .2;
muY = 2;   varY = 2;  covXY = sqrt(varX)*sqrt(varY)*corrXY; 
mu_dist1 = [muX muY]; cov_dist1 = [varX covXY; covXY varY];     % Means and Covariance Matrix for target "PDF"
% Second bivariate normal parameters
muX = -2;   varX = 0.49;  corrXY = -0.5;
muY = -2;   varY = 0.80;  covXY = sqrt(varX)*sqrt(varY)*corrXY; 
mu_dist2 = [muX muY]; cov_dist2 = [varX covXY; covXY varY];     % Means and Covariance Matrix for target "PDF"
% Target "PDF"
p = @(iD, nIx, x_t) ...
    normrnd(mu_dist1(iD) + cov_dist1(iD, iD) / cov_dist1(nIx, nIx) * cov_dist1(1, 2) * (x_t - mu_dist1(iD)), ...
    abs(sqrt(1 - cov_dist1(1, 2)^2)) * cov_dist1(iD, iD)) + ...
    normrnd(mu_dist2(iD) + cov_dist1(iD, iD) / cov_dist1(nIx, nIx) * cov_dist2(1, 2) * (x_t - mu_dist2(iD)), ...
    abs(sqrt(1 - cov_dist2(1, 2)^2)) * cov_dist2(iD, iD));  % Gaussian mixture

% p1 = @(iD, nIx, x_t) ...
%     normrnd(mu_dist1(iD) + cov_dist1(iD, iD) / cov_dist1(nIx, nIx) * cov_dist1(1, 2) * (x_t - mu_dist1(iD)), ...
%     abs(sqrt(1 - cov_dist1(1, 2)^2)) * cov_dist1(iD, iD));  % Gaussian mixture
% 
% p2 = @(iD, nIx, x_t) ...
%     normrnd(mu_dist2(iD) + cov_dist1(iD, iD) / cov_dist1(nIx, nIx) * cov_dist2(1, 2) * (x_t - mu_dist2(iD)), ...
%     abs(sqrt(1 - cov_dist2(1, 2)^2)) * cov_dist2(iD, iD));  % Gaussian mixture


%% Marginals
X = (-8:0.05:8)';   nx = length(X);
Y = (-8:0.05:8)';   ny = length(Y);
[XX,YY] = meshgrid(X,Y);
pXY = @(x,y) mvnpdf([x' y'],mu_dist2,cov_dist2) + mvnpdf([x' y'],mu_dist2,cov_dist2); 
pX = zeros(nx,1);
for i = 1:nx
   pX(i) = quad(@(y) pXY(repmat(X(i),1,length(y)),y), -10, 10);  % Marginal X
end
pY = zeros(ny,1);
for i = 1:ny
   pY(i) = quad(@(x) pXY(x,repmat(Y(i),1,length(x))), -10, 10);  % Marginal Y
end
 
%% RUN GIBBS SAMPLER
% INITIALIZE SAMPLES
x = zeros(2, nSamples);
x(:,1) = [unifrnd(X(1), X(end)), unifrnd(Y(1), Y(end))];
dims = 1:2; % INDEX INTO EACH DIMENSION

for t = 2 : nSamples
    x(1,t) = p(1, 2, x(2,t - 1));
    x(2,t) = p(2, 1, x(1,t)); 
end
 
%% DISPLAY SAMPLING DYNAMICS
figure;
plot(x(1,:),x(2,:),'r.');
xlabel('x_1'); ylabel('x_2');
axis square

figure;
% Marginal Y
subplot(4,4,[1 5 9]); 
hold on;
[n2, x2] = hist(x(2,:), ceil(sqrt(nSamples))); 
barh(x2, n2/(nSamples*(x2(2)-x2(1))));                
set(gca,'XDir','reverse','YAxisLocation','right', 'Box','off'); 
plot(pY/trapz(Y,pY),Y,'r-','LineWidth',2);
axis([0, 1, Y(1), Y(end)]); 


% Marginal X
subplot(4,4,[14 15 16]);
[n1, x1] = hist(x(1,:), ceil(sqrt(nSamples))); 
bar(x1, n1/(nSamples*(x1(2)-x1(1))));  hold on; 
set(gca,'YDir','reverse','XAxisLocation','top','Box','off'); 
plot(X,pX/trapz(X,pX),'r-','LineWidth',2);
axis([X(1), X(end), 0, 1]); 

% Distribution of samples
subplot(4,4,[2 3 4 6 7 8 10 11 12]);  
plot(x(1,:),x(2,:),'b.','LineWidth',1); 
axis([X(1), X(end), Y(1), Y(end)]); 


