%% Train DDPG Agent to Control Flying Robot
clear;clc;close all

mdl = 'rlFlyingRobotEnv';
open_system(mdl)

theta0 = 0;
x0 = -15;
y0 = 0;
ymax = 60;
xmax = 60;

Ts = 0.4;
Tf = 40;


integratedMdl = 'IntegratedFlyingRobot';
[~,agentBlk,observationInfo,actionInfo] = createIntegratedEnv(mdl,integratedMdl);

numObs = prod(observationInfo.Dimension);
observationInfo.Name = 'observations';

numAct = prod(actionInfo.Dimension);
actionInfo.LowerLimit = -ones(numAct,1);
actionInfo.UpperLimit =  ones(numAct,1);
actionInfo.Name = 'thrusts';


env = rlSimulinkEnv(integratedMdl,agentBlk,observationInfo,actionInfo);


env.ResetFcn = @(in) flyingRobotResetFcn(in);


rng(0)


% Specify the number of outputs for the hidden layers.
hiddenLayerSize = 100; 

%% critic
observationPath = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

% Create dlnetwork from layer graph
criticNetwork = dlnetwork(observationPath);

criticOptions = rlOptimizerOptions('LearnRate',1e-03,'GradientThreshold',1);

critic = rlValueFunction(criticNetwork,observationInfo,...
    'ObservationInputNames','observation');
critic.UseDevice = 'gpu';

%% actor
% Define common input path layer
commonPath = [ 
    featureInputLayer(prod(observationInfo.Dimension),Name="comPathIn")
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(1,Name="comPathOut") 
    ];

% Define mean value path
meanPath = [
    fullyConnectedLayer(15,Name="meanPathIn")
    reluLayer
    fullyConnectedLayer(prod(actionInfo.Dimension));
    tanhLayer;
    scalingLayer(Name="meanPathOut",Scale=actionInfo.UpperLimit) 
    ];

% Define standard deviation path
sdevPath = [
    fullyConnectedLayer(15,"Name","stdPathIn")
    reluLayer
    fullyConnectedLayer(prod(actionInfo.Dimension));
    softplusLayer(Name="stdPathOut") 
    ];

% Add layers to layerGraph object
actorNetwork = layerGraph(commonPath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,sdevPath);

% Connect paths
actorNetwork = connectLayers(actorNetwork,"comPathOut","meanPathIn/in");
actorNetwork = connectLayers(actorNetwork,"comPathOut","stdPathIn/in");

% Plot network 
plot(actorNetwork)

% Convert to dlnetwork and display number of weights
actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork)

actorOptions = rlOptimizerOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlContinuousGaussianActor(actorNetwork, observationInfo, actionInfo, ...
    "ActionMeanOutputNames","meanPathOut",...
    "ActionStandardDeviationOutputNames","stdPathOut",...
    ObservationInputNames="comPathIn");
actor.UseDevice = 'gpu';

%% agent
agentOptions = rlPPOAgentOptions(...
    'SampleTime',Ts,...
    'ActorOptimizerOptions',actorOptions,...
    'CriticOptimizerOptions',criticOptions,...
    'MiniBatchSize',256);

agent = rlPPOAgent(actor,critic,agentOptions);

rewardsuccess = 700;
maxepisodes = 20000;
maxsteps = ceil(Tf/Ts);
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'StopOnError',"on",...
    'Verbose',false,...
    'Plots',"training-progress",...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',rewardsuccess,...
    'ScoreAveragingWindowLength',10,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',rewardsuccess); 

%% train
doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions);
    save(['training/training-', char(datetime('today'))], 'trainingStats', 'trainingOptions', 'agent', 'env')
else
    % Load the pretrained agent for the example.
    load('FlyingRobotDDPG.mat','agent')       
end

%%
Ts = 0.4;
Tf = 70;
maxsteps=ceil(Tf/Ts);
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);

%%

data = squeeze(experience.Observation.observations.Data);
time = experience.Observation.observations.Time;

xs = data(1, :);
ys = data(2, :);
thetas = atan2(data(5, :), data(6, :));

figure
subplot(3, 1, 1)
plot(time, xs); hold on; plot(time, zeros(1, length(time)), '--'); grid on; ylabel('X')
subplot(3, 1, 2)
plot(time, ys); hold on;  plot(time, zeros(1, length(time)), '--'); grid on; ylabel('Y')
subplot(3, 1, 3)
plot(time, thetas*180/pi); hold on;  plot(time, 90*ones(1, length(time)), '--'); grid on; ylabel('Theta')
xlabel('Time (s)')













