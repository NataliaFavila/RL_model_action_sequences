%% LR reinforced with two action states
clear all

tic
%Number of trials 
Ntrials = 2000;

%Number of rats
Nrats = 100; 

%Learning paramaters
alpha = 0.02;     % alpha; learn rate for state value estimation 
beta = 0.04;      % beta; learn rate for action probability update 
gamma = 0.8;       % discount factor in return 
lambda = 0.2;      % credit assignment parameter

%Rewards
SeqR = 1;       %reward for correct sequence
OtherR = -0.05;  %reward for any other action

%Actions 
acts = [1, 2];         %possible actions R = 1, L = 2
Nacts = length(acts);

%Storing results from all rats
PrLs1 = zeros(Nrats,Ntrials);
PrRs1 = zeros(Nrats,Ntrials);
PrLs2 = zeros(Nrats,Ntrials);
PrRs2 = zeros(Nrats,Ntrials);
QL1 = zeros(Nrats,Ntrials);
QR1 = zeros(Nrats,Ntrials);
QL2 = zeros(Nrats,Ntrials);
QR2 = zeros(Nrats,Ntrials);
Lfreq = zeros(Nrats,Ntrials);
Rfreq = zeros(Nrats,Ntrials);
RLfreq = zeros(Nrats,Ntrials);
LRfreq = zeros(Nrats,Ntrials);
DELTA0 = zeros(Nrats,Ntrials);
DELTAA1 = zeros(Nrats,Ntrials);
DELTAA2 = zeros(Nrats,Ntrials);
DELTAE = zeros(Nrats,Ntrials);
VS0All = zeros(Nrats,Ntrials);
VSa1All = zeros(Nrats,Ntrials);
VSa2All = zeros(Nrats,Ntrials);
VSeRAll = zeros(Nrats,Ntrials);
VSeNoRAll = zeros(Nrats,Ntrials);
NSel = zeros(Nrats,Ntrials);

%% 

for r = 1:Nrats 
    
    %counters
    i = 1;  %trial counter
    a = 0; %action counter per trial
    LR = 0; RL = 0; %count different  sequences within session
    Lcounter = zeros(1,Ntrials); %Left counter 
    Rcounter = zeros(1,Ntrials);  %Right counter
    RLcounter = zeros(1,Ntrials); 
    LRcounter = zeros(1,Ntrials);
    
    %Rewards
    currR = 0;      %Was the trial reinforced (1) or not (0)?
    
    %Actions selected
    selActs = [];                %selected acts in each trial
    nselActs = zeros(1,Ntrials); %number of actions selected in one trial
    Act1 = 0;                    %action 1
    Act2 = 0;                     %action 2
    
    %Value states
    Vs0 = 0;               %Before trial
    VsA1 = 0;               %During action 1
    VsA2 = 0;               %During action 2
    VsE = zeros(2,1);      %During evaluation of outcome (if correct seq was executed or not)
    VsR = zeros(2,1);      %During reward delivery
    
    %Store Value states per trial
    VS0 = zeros(1,Ntrials);    %Before trial
    VSA1 = zeros(1,Ntrials);   %During action 1
    VSA2 = zeros(1,Ntrials);   %During action 2
    VSE = zeros(2,Ntrials);    %During evaluation of outcome (if seq was executed or not)
    VSR = zeros(2,Ntrials);    %During reward delivery

    %Prediction errors for each state
    delta0 = 0;
    deltaA1 = 0;
    deltaA2 = 0;
    deltaE = 0;
    
    %Store deltas for all trials
    Delta0 = zeros(1,Ntrials);
    DeltaA1 = zeros(1,Ntrials);
    DeltaA2 = zeros(1,Ntrials);
    DeltaE = zeros(1,Ntrials);

    %action preferences
    qs = 5 .* ones(2,Nacts); 
    QsallAct1 = zeros(Nacts, Ntrials);
    QsallAct2 = zeros(Nacts, Ntrials);
    Psalls1 = zeros(Nacts, Ntrials);
    Psalls2 = zeros(Nacts, Ntrials);
 
   %Eligibility traces
    e1 = [0 0];
    e2 = [0 0];
   
    
%% All experiment
for j = 1:Ntrials

    %One trial
    while (currR~=1)
        
        %%%%%%%%%%%%%% BEFORE TRIAL %%%%%%%%%%%%%%%%
        
        %Calculate delta0 and update Vs0
        delta0(i) = gamma.* VsA1(i) - Vs0(i);
        Vs0(i) =  0;

        %%%%%%%%%%%%%% ACTION 1%%%%%%%%%%%%%%%%
        
        %Policy
        normf = sum(exp(qs(1,:)));
        ps1 = exp(qs(1,:))./normf;
        
        %Take an action 1
        a = a + 1;  
        p =rand;
        selActs(a) = acts(find(p<cumsum(ps1),1,'first'));
        Act1 = selActs(a);
        
        %Update action eligibility trace
        e1(Act1) =  1; 
        
        %Calculate deltaA1 and update VsA1
        deltaA1(i) = gamma.*VsA2(i) - VsA1(i);
        VsA1(i) = VsA1(i) + alpha*deltaA1(i);
        
        %Update preference of selected actions 
        qs(1,Act1) = qs(1,Act1) + beta.*deltaA1(i)*e1(Act1);

        %Update eligibility trace
        e1 = gamma.*lambda.*e1;
      

       %%%%%%%%%%%%%% ACTION 2 %%%%%%%%%%%%%%%%
       
       %Policy
        normf = sum(exp(qs(2,:)));
        ps2 = exp(qs(2,:))./normf;
       
        %Take an action (softmax policy)
        a = a + 1;  
        p =rand;
        selActs(a) = acts(find(p<cumsum(ps2),1,'first'));
        Act2 = selActs(a);
       
        %Update action eligibility trace
        e2(Act2) = 1;
       
        %Find evaluative reward (LR sequence is reinforced in 1st phase)
        if (Act1 == 2 && Act2 == 1) 
            LR = LR + 1;
            currR = SeqR;
            state_ind = 1;
        else
            currR = OtherR;
            state_ind = 2;
        end

        %Count RL actions
        if (Act1 == 1 && Act2 == 2) 
            RL = RL + 1;
        end 
       
        %Calculate deltaA2 and update VsA2 
        deltaA2(i) = gamma.*VsE(state_ind, i) - VsA2(i);
        VsA2(i) = VsA2(i) + alpha*deltaA2(i);
        
        %Update preference of selected actions 
        qs(2,Act2) = qs(2,Act2) + beta.*deltaA2(i)*e2(Act2);
        qs(1,Act1) = qs(1,Act1) + beta.*deltaA2(i)*e1(Act1);
        
        %Update eligibility trace
        e1 = gamma.*lambda.*e1;
        e2 = gamma.*lambda.*e2;
        
        %%%%%%%%%%%%%% EVALUATION OF OUTCOME %%%%%%%%%%%%%%%%
       
        %Calculae deltaE and update VsE
        deltaE(i) = currR + gamma.*VsR(state_ind,i) - VsE(state_ind,i);
        VsE(state_ind,i) = VsE(state_ind,i) + alpha.*deltaE(i);

        %Update preference of selected actions 
        qs(2,Act2) = qs(2,Act2) + beta.*deltaE(i).*e2(Act2);
        qs(1,Act1) = qs(1,Act1) + beta.*deltaE(i).*e1(Act1);
        
        %Update previous action and copy values to next time step
        Vs0(i+1) = Vs0(i);
        VsA1(i+1) = VsA1(i);
        VsA2(i+1) = VsA2(i);
        VsE(:, i+1) =VsE(:, i);
        VsR(:, i+1) = VsR(:, i);
        i = i+1;
    end
    
    %Reset eligibility traces after obtaining reinforcer
    e2 = [0,0];
    e1 = [0,0];

    %Store last action values and probabilities of the trial
     QsallAct1(:,j) = qs(1,:);
     QsallAct2(:,j) = qs(2,:);
     Psalls1(:,j) = ps1;
     Psalls2(:,j) = ps2;
     
     %Store mean state and delta values of the trial
     VS0(j) = mean(Vs0(i-a/2:i-1));
     VSA1(j) = mean(VsA1(i-a/2:i-1));
     VSA2(j) = mean(VsA2(i-a/2:i-1));
     VSE(1,j) = mean(VsE(1,i-a/2:i-1));
     VSE(2,j) = mean(VsE(2,i-a/2:i-1));
     VSR(1,j) = mean(VsR(1,i-a/2:i-1));
     VSR(2,j) = mean(VsR(2,i-a/2:i-1));
     Delta0(j) = mean(delta0(i-a/2:i-1));
     DeltaA1(j) = mean(deltaA1(i-a/2:i-1));
     DeltaA2(j) = mean(deltaA2(i-a/2:i-1));
     DeltaE(j) = mean(deltaE(i-a/2:i-1));
     
     %Store  number of R and L responses on jth trial
     Rcounter(j) = sum(selActs == 1);
     Lcounter(j) = sum(selActs == 2);
     RLcounter(j) = RL;
     LRcounter(j) = LR;
     nselActs(j) = a;
     
    %Reset values for next trial
    currR = 0;
    a = 0; 
    LR = 0; RL = 0;
    selActs = [];

end

PrRs1(r,:) = Psalls1(1,:);
PrLs1(r,:) = Psalls1(2,:);
PrRs2(r,:) = Psalls2(1,:);
PrLs2(r,:) = Psalls2(2,:);
QL1(r,:) = QsallAct1(2,:);
QR1(r,:) = QsallAct1(1,:);
QL2(r,:) = QsallAct2(2,:);
QR2(r,:) = QsallAct2(1,:);
Lfreq(r,:) =  Lcounter;
Rfreq(r,:) =  Rcounter;
RLfreq(r,:) = RLcounter;
LRfreq(r,:) = LRcounter;
DELTA0(r,:) = Delta0;
DELTAA1(r,:) = DeltaA1;
DELTAA2(r,:) = DeltaA2;
DELTAE(r,:) = DeltaE;
VS0All(r,:) = VS0;
VSa1All(r,:) = VSA1;
VSa2All(r,:)= VSA2;
VSeRAll(r,:) = VSE(1,:);
VSeNoRAll(r,:) = VSE(2,:);
NSel(r,:) = nselActs;

end
%% Performance measurements

%Distal/proximal ratio (session wise)
n = 50;
Ratio = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    Distal = arrayfun(@(i) sum(Lfreq(r,i:i+n-1)),1:n:length(Lfreq)-n+1);
    Prox = arrayfun(@(i) sum(Rfreq(r,i:i+n-1)),1:n:length(Rfreq)-n+1); 
    Ratio(r,:) = Distal./Prox;
end

DPRatio = mean(Ratio);

%Perfect trials (session wise)
NSel2 = NSel == 2;
PerfTrials = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    Nperf = NSel2(r,:);
    perf = arrayfun(@(i) sum(Nperf(i:i+n-1)),1:n:length(Nperf)-n+1)';
    PerfTrials(r,:) = perf/n;
end

PropPerf = mean(PerfTrials);

%Number of actions per trial (session wise)

ActsTrial = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    NacRat = NSel(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    ActsTrial(r,:) = Nac;
end

nTActs = mean(ActsTrial);

toc