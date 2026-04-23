%% BLE sim

clc
clear
close all

%% Define environment

room_x = 10;
room_y = 10;

anchors = [
0 0
10 0
5 9
];

num_anchors = size(anchors,1);

%% Path Loss Model

n = 3;
C = -40;
noise_std = 2;
shadow_std = 6;

%% Moving Tag Path

waypoints = [
1 1
2 2
4 3
6 5
8 4
7 2
5 1
3 3
2 5
];

points_per_segment = 8;

true_path = [];

for i = 1:size(waypoints,1)-1

    p1 = waypoints(i,:);
    p2 = waypoints(i+1,:);

    xs = linspace(p1(1),p2(1),points_per_segment);
    ys = linspace(p1(2),p2(2),points_per_segment);

    true_path = [true_path; [xs' ys']];

end

num_steps = size(true_path,1);

%% Fingerprint data creation

grid_step = 0.5;

[xg,yg] = meshgrid(0:grid_step:room_x,0:grid_step:room_y);
grid_points = [xg(:) yg(:)];

num_points = size(grid_points,1);

samples_per_point = 50;

total_samples = num_points * samples_per_point;

fingerprints = zeros(total_samples,num_anchors);
positions = zeros(total_samples,2);

idx = 1;

for p = 1:num_points

    pos = grid_points(p,:);

    for s = 1:samples_per_point

        for a = 1:num_anchors

            d = sqrt((pos(1)-anchors(a,1))^2 + ...
                     (pos(2)-anchors(a,2))^2);

            d = max(d,0.5);

            shadow = shadow_std*randn;

            fingerprints(idx,a) = -10*n*log10(d) + C + shadow + noise_std*randn;

        end

        positions(idx,:) = pos;

        idx = idx + 1;

    end

end

%% MLP Fingerprinting Model

mlp_net = fitnet([20 10]);
mlp_net.trainParam.epochs = 200;
mlp_net.trainParam.showWindow = false;

mlp_net = train(mlp_net,fingerprints',positions');

%% CNN Fingerprinting Model

X_cnn = reshape(fingerprints',[num_anchors 1 1 total_samples]);
Y_cnn = positions;

layers = [

imageInputLayer([num_anchors 1 1])

convolution2dLayer([2 1],16,"Padding","same")
batchNormalizationLayer
reluLayer

convolution2dLayer([2 1],32,"Padding","same")
batchNormalizationLayer
reluLayer

fullyConnectedLayer(32)
reluLayer

fullyConnectedLayer(2)

regressionLayer
];

options = trainingOptions("adam", ...
"MaxEpochs",40, ...
"MiniBatchSize",16, ...
"Verbose",false);

cnn_net = trainNetwork(X_cnn,Y_cnn,layers,options);

%% Storage

pos_trilat = zeros(num_steps,2);
pos_weighted = zeros(num_steps,2);
pos_kf = zeros(num_steps,2);
pos_knn = zeros(num_steps,2);
pos_mlp = zeros(num_steps,2);
pos_cnn = zeros(num_steps,2);

%% Kalman Filter Initialization

dt = 1;

F = [1 0 dt 0
     0 1 0 dt
     0 0 1 0
     0 0 0 1];

H = [1 0 0 0
     0 1 0 0];

Q = 0.01*eye(4);
R = 0.5*eye(2);

P = eye(4);

x_state = [true_path(1,1); true_path(1,2); 0; 0];

%% RSSI Temporal Smoothing Initialization

alpha = 0.3;
rssi_smooth = zeros(num_anchors,1);

%% Simulation Loop

for k = 1:num_steps

true_pos = true_path(k,:);

true_d = zeros(num_anchors,1);

for i=1:num_anchors

true_d(i) = sqrt((true_pos(1)-anchors(i,1))^2 + ...
                 (true_pos(2)-anchors(i,2))^2);

end

%% Raw RSSI

rssi_raw = -10*n*log10(true_d) + C + noise_std*randn(num_anchors,1);

%% Temporal RSSI Smoothing

if k == 1
    rssi_smooth = rssi_raw;
else
    rssi_smooth = alpha*rssi_raw + (1-alpha)*rssi_smooth;
end

rssi = rssi_smooth;

%% Distance Estimation

est_d = 10.^((C - rssi)/(10*n));

%% Trilateration

x1 = anchors(1,1);
y1 = anchors(1,2);
d1 = est_d(1);

A = zeros(num_anchors-1,2);
b = zeros(num_anchors-1,1);

for i=2:num_anchors

xi = anchors(i,1);
yi = anchors(i,2);
di = est_d(i);

A(i-1,:) = [2*(xi-x1) 2*(yi-y1)];

b(i-1) = d1^2 - di^2 - x1^2 + xi^2 - y1^2 + yi^2;

end

pos_trilat(k,:) = (A\b)';

%% Weighted Trilateration

weights = 1./(est_d(2:end).^2);
W = diag(weights);

pos_weighted(k,:) = (inv(A'*W*A)*A'*W*b)';

%% Kalman Filter

x_pred = F*x_state;
P_pred = F*P*F' + Q;

z = pos_weighted(k,:)';

K = P_pred*H'/(H*P_pred*H' + R);

x_state = x_pred + K*(z - H*x_pred);

P = (eye(4)-K*H)*P_pred;

pos_kf(k,:) = x_state(1:2)';

%% kNN Fingerprinting

k_neighbors = 5;

dist = sqrt(sum((fingerprints - rssi').^2,2));

[~,idx] = sort(dist);

nearest = idx(1:k_neighbors);

weights = 1 ./ dist(nearest);

pos_knn(k,:) = sum(positions(nearest,:).*weights,1) / sum(weights);

%% MLP Prediction

pos_mlp(k,:) = mlp_net(rssi)';

%% CNN Prediction

rssi_input = reshape(rssi,[num_anchors 1 1]);
pos_cnn(k,:) = predict(cnn_net,rssi_input)';

end


%% Error Calculation

err_trilat = sqrt(sum((pos_trilat-true_path).^2,2));
err_weighted = sqrt(sum((pos_weighted-true_path).^2,2));
err_kf = sqrt(sum((pos_kf-true_path).^2,2));
err_knn = sqrt(sum((pos_knn-true_path).^2,2));
err_mlp = sqrt(sum((pos_mlp-true_path).^2,2));
err_cnn = sqrt(sum((pos_cnn-true_path).^2,2));

%% Results Table

methods = ["Trilateration","Weighted","Kalman","kNN","MLP","CNN"];

mean_err = [
mean(err_trilat)
mean(err_weighted)
mean(err_kf)
mean(err_knn)
mean(err_mlp)
mean(err_cnn)
];

median_err = [
median(err_trilat)
median(err_weighted)
median(err_kf)
median(err_knn)
median(err_mlp)
median(err_cnn)
];

rmse = [
sqrt(mean(err_trilat.^2))
sqrt(mean(err_weighted.^2))
sqrt(mean(err_kf.^2))
sqrt(mean(err_knn.^2))
sqrt(mean(err_mlp.^2))
sqrt(mean(err_cnn.^2))
];

std_err = [
std(err_trilat)
std(err_weighted)
std(err_kf)
std(err_knn)
std(err_mlp)
std(err_cnn)
];

results_table = table(methods',mean_err,median_err,rmse,std_err,...
'VariableNames',{'Method','MeanError','MedianError','RMSE','StdDev'});

disp(results_table)

%% Visualization

figure
hold on
grid on

scatter(anchors(:,1),anchors(:,2),120,'b','filled')

plot(true_path(:,1),true_path(:,2),'k','LineWidth',2)

plot(pos_trilat(:,1),pos_trilat(:,2),'r')
plot(pos_weighted(:,1),pos_weighted(:,2),'m')
plot(pos_kf(:,1),pos_kf(:,2),'g','LineWidth',2)
plot(pos_knn(:,1),pos_knn(:,2),'c')
plot(pos_mlp(:,1),pos_mlp(:,2),'y')
plot(pos_cnn(:,1),pos_cnn(:,2),'Color',[0.5 0 0.5])

legend("Anchors","True Path","Trilateration","Weighted","Kalman","kNN","MLP","CNN")

xlabel("X (meters)")
ylabel("Y (meters)")

title("Comparison of Indoor Localization Algorithms")

xlim([0 room_x])
ylim([0 room_y])

%% CDF Plot

figure
hold on
grid on

[f,x] = ecdf(err_trilat); plot(x,f)
[f,x] = ecdf(err_weighted); plot(x,f)
[f,x] = ecdf(err_kf); plot(x,f)
[f,x] = ecdf(err_knn); plot(x,f)
[f,x] = ecdf(err_mlp); plot(x,f)
[f,x] = ecdf(err_cnn); plot(x,f)

legend(methods)

xlabel("Localization Error (m)")
ylabel("CDF")

title("CDF of Localization Error")
