clearvars
close all
clc

%% Core
%training
Train_inter_individuality = 0.800 + 0.1*(0:4);
Train_severity_level = (10 - (0:9))/10;
Train_sample_num = 25;
[tr_data, tr_label] = generate_massiave_data(Train_inter_individuality, Train_severity_level, Train_sample_num);
save('raw_data_train_small.mat', 'tr_data');
save('raw_label_train_small.mat', 'tr_label');

Test_inter_individuality = 0.800 + 0.1*(0:4);
Test_severity_level = (10 - (0:90)/10)/10;
Test_sample_num = 3;
[ts_data, ts_label] = generate_massiave_data(Test_inter_individuality, Test_severity_level, Test_sample_num);
save('raw_data_test_small.mat', 'ts_data');
save('raw_label_test_small.mat', 'ts_label');