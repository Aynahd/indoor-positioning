%% UWB simulation

clc
clear
close all

%% Environment

room_x = 10;
room_y = 10;

anchors = [
0 0
10 0
5 9
];

num_anchors = size(anchors,1);

%% UWB Parameters

uwb_noise = 0.15;
nlos_prob = 0.2;

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

%% Fingerprinting Dataset Creation

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

            fingerprints(idx,a) = d + uwb_noise*randn;

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
R = 0.05*eye(2);

P = eye(4);

x_state = [true_path(1,1); true_path(1,2); 0; 0];

%% Temporal smoothing

alpha = 0.3;
uwb_smooth = zeros(num_anchors,1);

%% Simulation Loop

for k = 1:num_steps

true_pos = true_path(k,:);

%% True Distances

true_d = zeros(num_anchors,1);

for i=1:num_anchors

true_d(i) = sqrt((true_pos(1)-anchors(i,1))^2 + ...
                 (true_pos(2)-anchors(i,2))^2);

end

%% UWB Measurement with NLOS

uwb_raw = true_d + uwb_noise*randn(num_anchors,1);

for i=1:num_anchors

    if rand < nlos_prob
        
        uwb_raw(i) = uwb_raw(i) + 0.5 + 1.5*rand;
        
    end
    
end

%% Temporal Smoothing

if k == 1
    uwb_smooth = uwb_raw;
else
    uwb_smooth = alpha*uwb_raw + (1-alpha)*uwb_smooth;
end

uwb_d = uwb_smooth;

%% Trilateration

x1 = anchors(1,1);
y1 = anchors(1,2);
d1 = uwb_d(1);

A = zeros(num_anchors-1,2);
b = zeros(num_anchors-1,1);

for i=2:num_anchors

xi = anchors(i,1);
yi = anchors(i,2);
di = uwb_d(i);

A(i-1,:) = [2*(xi-x1) 2*(yi-y1)];

b(i-1) = d1^2 - di^2 - x1^2 + xi^2 - y1^2 + yi^2;

end

pos_trilat(k,:) = (A\b)';

%% Weighted Trilateration

weights = 1./(uwb_d(2:end).^2);
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

dist = sqrt(sum((fingerprints - uwb_d').^2,2));

[sorted_dist,idx] = sort(dist);

nearest = idx(1:k_neighbors);

weights = 1 ./ (sorted_dist(1:k_neighbors) + 1e-6);

pos_knn(k,:) = sum(positions(nearest,:).*weights,1) / sum(weights);

%% MLP Prediction

pos_mlp(k,:) = mlp_net(uwb_d)';

%% CNN Prediction

input = reshape(uwb_d,[num_anchors 1 1]);
pos_cnn(k,:) = predict(cnn_net,input)';

end

%% Error Calculation

err_trilat = sqrt(sum((pos_trilat-true_path).^2,2));
err_weighted = sqrt(sum((pos_weighted-true_path).^2,2));
err_kf = sqrt(sum((pos_kf-true_path).^2,2));
err_knn = sqrt(sum((pos_knn-true_path).^2,2));
err_mlp = sqrt(sum((pos_mlp-true_path).^2,2));
err_cnn = sqrt(sum((pos_cnn-true_path).^2,2));

%% Metrics

methods = ["Trilateration","Weighted","Kalman","kNN","MLP","CNN"];

mean_err = [
mean(err_trilat)
mean(err_weighted)
mean(err_kf)
mean(err_knn)
mean(err_mlp)
mean(err_cnn)
];

rmse = [
sqrt(mean(err_trilat.^2))
sqrt(mean(err_weighted.^2))
sqrt(mean(err_kf.^2))
sqrt(mean(err_knn.^2))
sqrt(mean(err_mlp.^2))
sqrt(mean(err_cnn.^2))
];

results_table = table(methods',mean_err,rmse,...
'VariableNames',{'Method','MeanError','RMSE'});

disp(results_table)

% Path Visualization
figure
hold on
grid on

plot(true_path(:,1),true_path(:,2),'k-','LineWidth',3)

plot(pos_trilat(:,1),pos_trilat(:,2),'r--','LineWidth',1.5)
plot(pos_weighted(:,1),pos_weighted(:,2),'g--','LineWidth',1.5)
plot(pos_kf(:,1),pos_kf(:,2),'b--','LineWidth',1.5)
plot(pos_knn(:,1),pos_knn(:,2),'c--','LineWidth',1.5)
plot(pos_mlp(:,1),pos_mlp(:,2),'m--','LineWidth',1.5)
plot(pos_cnn(:,1),pos_cnn(:,2),'y--','LineWidth',1.5)

scatter(anchors(:,1),anchors(:,2),120,'filled','k')

legend("True Path","Trilateration","Weighted","Kalman","kNN","MLP","CNN","Anchors")

xlabel("X Position (m)")
ylabel("Y Position (m)")

title("Localization Path Comparison")

axis equal
xlim([0 room_x])
ylim([0 room_y])

% Error CDF Comparison


figure
hold on
grid on

cdfplot(err_trilat)
cdfplot(err_weighted)
cdfplot(err_kf)
cdfplot(err_knn)
cdfplot(err_mlp)
cdfplot(err_cnn)

legend("Trilateration","Weighted","Kalman","kNN","MLP","CNN")

xlabel("Localization Error (m)")
ylabel("CDF")

title("Localization Error CDF Comparison")


