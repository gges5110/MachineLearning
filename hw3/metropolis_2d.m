clear; home;

%% Target "PDF"
% First bivariate normal parameters
muX = 2;   varX = 1;  corrXY = .2;
muY = 2;   varY = 2;  covXY = sqrt(varX)*sqrt(varY)*corrXY; 
mu_dist1 = [muX muY]; cov_dist1 = [varX covXY; covXY varY];                 % Means and Covariance Matrix for target "PDF"
% Second bivariate normal parameters
muX = -2;   varX = 0.49;  corrXY = -0.5;
muY = -2;   varY = 0.80;  covXY = sqrt(varX)*sqrt(varY)*corrXY; 
mu_dist2 = [muX muY]; cov_dist2 = [varX covXY; covXY varY];                 % Means and Covariance Matrix for target "PDF"
% Target "PDF"
p = @(X) mvnpdf(X,mu_dist1,cov_dist1) + mvnpdf(X,mu_dist2,cov_dist2);       % Gaussian mixture

%% Proposal PDF
% Bivariate normal parameters
varX_proposal  = 1;    varY_proposal = 1;   corrXY_proposal = -0.5;         % Var(X), var(Y) and CorCoef(X,Y) for proposal, assumed
covXY_proposal = sqrt(varX_proposal)*sqrt(varY_proposal)*corrXY_proposal;   % Cov(XY)for proposal
cov_proposal_PDF = [varX_proposal   covXY_proposal;...
                    covXY_proposal  varY_proposal];                         % Covariance Matrix for proposal
% Proposal PDF
proppdf = @(x, mu) mvnpdf(x, mu, cov_proposal_PDF);                         % Proposal PDF
proprnd = @(mu) mvnrnd(mu, cov_proposal_PDF);                               % Random distribution from proposal PDF


%% Sampling using Metropolis-Hasting sampling
sampleSize = 30000;
theta = zeros(2, sampleSize);
theta(:, 1) = [5 1];

for i = 1:sampleSize
    theta_ast = proprnd(theta(:, i));    % Sampling from proposal PDF
    r = p(theta_ast)/p(theta(:, i)');
    if rand <= min(1, r)
        theta(:, i + 1) = theta_ast;
    else 
        theta(:, i + 1) = theta(:, i);
    end    
end

%% Marginals
X = (-6:0.05:6)';   nx = length(X);
Y = (-6:0.05:6)';   ny = length(Y);
[XX,YY] = meshgrid(X,Y);
pXY = @(x,y) mvnpdf([x' y'],mu_dist1,cov_dist1) + mvnpdf([x' y'],mu_dist2,cov_dist2); 
pX = zeros(nx,1);
for i = 1:nx
   pX(i) = quad(@(y) pXY(repmat(X(i),1,length(y)),y), -10, 10);  % Marginal X
end
pY = zeros(ny,1);
for i = 1:ny
   pY(i) = quad(@(x) pXY(x,repmat(Y(i),1,length(x))), -10, 10);  % Marginal Y
end
Z = p([XX(:) YY(:)]);  Z = reshape(Z,length(YY),length(XX));

%% Plot Figures
% Target function and samples 
figure;
subplot(2,1,1);                                     % Target "PDF"
surf(X,Y,Z); grid on; shading interp;
title('f_{XY}(x,y)', 'FontSize', 12);
subplot(2,1,2);                                     % Distribution of samples
plot(theta(1,:),theta(2,:),'k.','LineWidth',1); hold on; 

figure;
% Marginal Y
subplot(4,4,[1 5 9]);
[n2, x2] = hist(theta(2,:), ceil(sqrt(sampleSize))); 
barh(x2, n2/(sampleSize*(x2(2)-x2(1))));   hold on;
set(gca,'XDir','reverse','YAxisLocation','right', 'Box','off'); 
plot(pY/trapz(Y,pY),Y,'r-','LineWidth',2); 

% Marginal X
subplot(4,4,[14 15 16]);
[n1, x1] = hist(theta(1,:), ceil(sqrt(sampleSize))); 
bar(x1, n1/(sampleSize*(x1(2)-x1(1))));  hold on;
set(gca,'YDir','reverse','XAxisLocation','top','Box','off'); 
plot(X,pX/trapz(X,pX),'r-','LineWidth',2);

% Distribution of samples
subplot(4,4,[2 3 4 6 7 8 10 11 12]);  
plot(theta(1,:),theta(2,:),'b.','LineWidth',1); axis tight;

