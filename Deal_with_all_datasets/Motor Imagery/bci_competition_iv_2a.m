% Just use the data of T, no E introduced
function data = csp(subject_index)

% subject_index = 2; %1-9

session_type = 'T';
dir = ['.\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf'];
[s, HDR] = sload(dir);

% Label 
% label = HDR.Classlabel;
labeldir = ['.\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir);
label_1 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;


% E data csp
session_type = 'E';
dir = ['.\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf'];
% dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
[s, HDR] = sload(dir);

% Label 
% label = HDR.Classlabel;
labeldir = ['.\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir);
label_2 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_2 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_2(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_2(isnan(data_2)) = 0;

%% shuffle the training data 
% num = size(data, 3);
t_data = data_1;
t_label = label_1 - 1;
e_data = data_2;
e_label = label_2 - 1;

%% calculate the csp of training data
index_0 = find(t_label==0);
index_1 = find(t_label==1);
index_2 = find(t_label==2);
index_3 = find(t_label==3);

%% CSP 
% obtain W of each band
fc = 250;
fb_data = zeros(1000,22,288);

Wl = 4; Wh = 40; % 通带范围
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:288
    fb_data(:,:,j) = filtfilt(b,a,t_data(:,:,j));
end


eeg_mean = mean(fb_data,3);
eeg_std = std(fb_data,1,3); 
fb_data = (fb_data-eeg_mean)./eeg_std;

W = zeros(22,22,4);
Cov = zeros(22,22,4);
Dis_mean = zeros(1,4);
Dis_std = zeros(1,4);
PP = zeros(22,22,4);
BB = zeros(22,22,4);
for nclass = 0:3 % let L represent one class, R represent other three classes

    if nclass == 0 
        index_L = index_0;
        index_R = [index_1;index_2;index_3];
    elseif nclass == 1
        index_L = index_1;
        index_R = [index_0;index_2;index_3];
    elseif nclass == 2
        index_L = index_2;
        index_R = [index_0;index_1;index_3];
    elseif nclass == 3
        index_L = index_3;
        index_R = [index_0;index_1;index_2];
    end   
    index_R = sort(index_R);
     
    Cov_L = zeros(22,22,length(index_L));
    Cov_R = zeros(22,22,length(index_R));
    for nL = 1:length(index_L)
        E = fb_data(:,:,index_L(nL));
        E = E'; % channel*sample point, don't mind
        EE = E*E';
        Cov_L(:,:,nL) = EE./trace(EE);
     end
     for nR = 1:length(index_R)
     E = fb_data(:,:,index_R(nR));
     E = E';
     EE = E*E';
     Cov_R(:,:,nR) = EE./trace(EE);
     end
     CovL = mean(Cov_L,3);
     CovR = mean(Cov_R,3);
     CovTotal = CovL + CovR;
     Cov(:,:,nclass+1) = CovL;
     
     % Cal the difference of Cov within the same class and the different
     % class
     Edt1 = (Cov_L - CovL).^2;
     Edt2 = (Cov_R - CovL).^2;
     Ed1 = squeeze(sqrt(sum(sum(Edt1))));
     Ed2 = squeeze(sqrt(sum(sum(Edt2))));
%      Ed1_max = max(Ed1);
%      Ed1_min = min(Ed1);
     Ed1_mean = mean(Ed1);
     Ed1_std = std(Ed1);
     Dis_mean(:,nclass+1) = Ed1_mean;
     Dis_std(:,nclass+1) = Ed1_std;
%      Ed1_sum = sum(Ed1);
%      Ed2_max = max(Ed2);
%      Ed2_min = min(Ed2);
     Ed2_mean = mean(Ed2);
     Ed2_std = std(Ed2);
%      Ed2_sum = sum(Ed2)/length(Ed2)*length(Ed1);
    % calculate the Euclidean Distance to condition probability
    % B = sum(exp(-Ed1^2));
    
        
     [Uc,lambda] = eig(CovTotal); % Uc is the eigenvector matrix, Dt is the diagonal matrix of eigenvalue
     eigenvalues = diag(lambda);
     [eigenvalues,egIndex] = sort(eigenvalues, 'descend');
     Ut = Uc(:,egIndex); % sort as the descend order
        
     P = sqrt(diag(eigenvalues)^-1)*Ut';
     PP(:,:,nclass+1) = P;
        
     SL = P*CovL*P';
     SR = P*CovR*P';   
                
     % [BL,lambda_L] = eig(SL);
     % evL = diag(lambda_L);
     % [evL,egI] = sort(evL, 'descend');
     % B = BL(:,egI);
     [BR,lambda_R] = eig(SR);
     evR = diag(lambda_R);
     [evR,egI] = sort(evR);
     B = BR(:,egI);
     BB(:,:,nclass+1) = B;
     w = P'*B;
     W(:,:,nclass+1) = w;
end
% Use the first four
% W1 = W(:,[1:2 (end-1):end],1);
% W2 = W(:,[1:2 (end-1):end],2);
% W3 = W(:,[1:2 (end-1):end],3);
% W4 = W(:,[1:2 (end-1):end],4);

% use the first four          
W1 = W(:,1:4,1);
W2 = W(:,1:4,2);
W3 = W(:,1:4,3);
W4 = W(:,1:4,4);

Wb = [W1,W2,W3,W4]; % Z = W' * X



%% Training data csp filtered
num_t = size(t_data, 3);
csp_data = zeros(1000,16,num_t);
for ntrial = 1:num_t
    tdata = fb_data(:,:,ntrial);
    tdata = tdata';
    tdata = Wb'*tdata;
    csp_data(:,:,ntrial) = tdata';
end
data = fb_data;
label = t_label + 1;
saveDir = ['.\strict_TE\A0',num2str(subject_index),'T.mat'];
save(saveDir,'data','label','csp_data','Wb','Cov','Dis_mean','Dis_std','PP','BB');


%% Test data csp filtered
fc = 250;
fb_data = zeros(1000,22,60);
Wl = 4; Wh = 40; % 通带范围
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:288
    fb_data(:,:,j) = filtfilt(b,a,e_data(:,:,j));
end

fb_data = (fb_data-eeg_mean)./eeg_std;

num_e = size(e_data, 3);
csp_data = zeros(1000,16,num_e);
for ntrial = 1:num_e
    edata = fb_data(:,:,ntrial);
    edata = edata';
    edata = Wb'*edata;
    csp_data(:,:,ntrial) = edata';
end

data = fb_data;
label = e_label+1;
saveDir = ['.\strict_TE\A0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label','csp_data');

end


        