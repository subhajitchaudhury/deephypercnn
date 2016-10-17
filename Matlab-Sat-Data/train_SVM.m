%train SVM

load('indian_pines_data.mat');

% [comp,recon]=pcExtract(permute(X,[2,1,3,4]),50);
% 
% X_r=permute(comp,[2,1,3,4]);

[num_pix,num_ch,h,w]=size(X);
train_num=round(0.8*num_pix);
test_num=num_pix-train_num;

train_X=reshape(X(1:train_num,:,:,:),[train_num,h*w*num_ch]);
train_y=labels(1:train_num);

test_X=reshape(X(train_num+1:end,:,:,:),[test_num,h*w*num_ch]);
test_y=labels(train_num+1:end);

disp('Linear');
model = svmtrain(double(train_y), double(train_X), '-s 0 -t 0 -b 1 -q 0');
[predict_label, accuracy, prob_values] = svmpredict...
    (double(test_y), double(test_X), model);
save('./results/svm_linear_raw.mat','model','predict_label', 'accuracy', 'prob_values');


disp('RBF');
model = svmtrain(double(train_y), double(train_X), '-s 0 -t 2 -b 1 -q 0');
[predict_label, accuracy, prob_values] = svmpredict...
    (double(test_y), double(test_X), model);
save('./results/svm_rbf_raw.mat','model','predict_label', 'accuracy', 'prob_values');
