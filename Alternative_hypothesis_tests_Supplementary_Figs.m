%% LR reinforced with two action states
clear all

%Number of trials 
Ntrials = 3000;

%Number of rats
Nrats = 100; 

%Learning paramaters
alpha = 0.1;     % alpha; learn rate for state value estimation 
beta = 0.1;      % beta; learn rate for action probability update 
gamma = 0.9;       % discount factor in return 
lambda = 0.1;      % credit assignment parameter
emax = 1;

%Rewards
SeqR = 1;       %reward for correct sequence
OtherR = -0.05;  %reward for any other action

%Actions 
acts = [1, 2];         %possible actions R = 1, L = 2
Nacts = length(acts);

%Storing results from all rats
% probabilities of responses
PrLs1 = zeros(Nrats,Ntrials);
PrRs1 = zeros(Nrats,Ntrials);
PrLs2 = zeros(Nrats,Ntrials);
PrRs2 = zeros(Nrats,Ntrials);
% Estimated values of responses
QL1 = zeros(Nrats,Ntrials);
QR1 = zeros(Nrats,Ntrials);
QL2 = zeros(Nrats,Ntrials);
QR2 = zeros(Nrats,Ntrials);
% frequencies of responses
Lfreq = zeros(Nrats,Ntrials);
Rfreq = zeros(Nrats,Ntrials);
RLfreq = zeros(Nrats,Ntrials);
LRfreq = zeros(Nrats,Ntrials);
% RPE 
DELTA0 = zeros(Nrats,Ntrials);
DELTAA1 = zeros(Nrats,Ntrials);
DELTAA2 = zeros(Nrats,Ntrials);
DELTAE = zeros(Nrats,Ntrials);
%state values
VS0All = zeros(Nrats,Ntrials);
VSa1All = zeros(Nrats,Ntrials);
VSa2All = zeros(Nrats,Ntrials);
VSeRAll = zeros(Nrats,Ntrials);
VSeNoRAll = zeros(Nrats,Ntrials);
VSR1All = zeros(Nrats,Ntrials);
VSR2All = zeros(Nrats,Ntrials);
NSel = zeros(Nrats,Ntrials);
%% 
tic
for r = 1:Nrats 
    
    %counters
    i = 1;  %trial counter
    a = 0; %action counter per trial
    LR = 0; RL = 0; %count different  sequences within session
    Lcounter = zeros(1,Ntrials); %Left counter 
    Rcounter = zeros(1,Ntrials);  %Right counter
    RLcounter = zeros(1,Ntrials); 
    LRcounter = zeros(1,Ntrials);
    
    %INITIAL VALUES
    %Rewards
    currR = 0;      %Was the trial reinforced (1) or not (0)?
    
    %action preferences and probabilities
    qs = 5 .* ones(2,Nacts); 
    QsallAct1 = zeros(Nacts, Ntrials);
    QsallAct2 = zeros(Nacts, Ntrials);
    Psalls1 = zeros(Nacts, Ntrials);
    Psalls2 = zeros(Nacts, Ntrials);
 
   %Eligibility traces
    e1 = [0 0];
    e2 = [0 0];
    
    %Actions selected
    selActs = [];                %selected acts in each trial
    nselActs = zeros(1,Ntrials); %number of actions selected in one trial
    Act1 = 0;                    %action 1
    Act2 = 0;                    %action 2
    
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
    lambdaALL = zeros(1,Ntrials);

%% All experiment
for j = 1:Ntrials
    if (j <= 1500)
        %Signaling phases
        phase1 = 1;
        gamma = 0.9;
    else
        phase1 = 0;
        %HYPOTHESIS TESTING
        %changing parameter in first trials (100 trials)
        if (j >1500) && (j<1601)
           gamma = 0.6;
        else 
           gamma = 0.9;
        end
    end
    
    lambdaALL(j) = gamma;

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
        e1(Act1) =  emax; 
        
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
        e2(Act2) = emax;
       
         %Find evaluative reward (LR sequence is reinforced in 1st phase)
         
         %phase 1
         if (phase1 == 1)
             %reward LR
            if (Act1 == 2 && Act2 == 1) 
                LR = LR + 1;
                currR = SeqR;
                state_ind = 1;
            %no reward
            else
                currR = OtherR;
                state_ind = 2;
                %Count other heterogeneous squence
                if (Act1 == 1 && Act2 == 2)    
                     RL = RL + 1;
                end
            end
            
        %phase 2
         else
            %reward RL
            if (Act1 == 1 && Act2 == 2) 
                RL = RL + 1;
                currR = SeqR;
                state_ind = 1;
            %no reward
            else
                currR = OtherR;
                state_ind = 2;
                %Count other heterogenous sequence
                if (Act1 == 2 && Act2 == 1)   
                    LR = LR + 1;
                end
            end
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
VSR1All(r,:) = VSR(1,:);
VSR2All(r,:) = VSR(2,:);
NSel(r,:) = nselActs;

end
toc
%% Performance measurements

%Distal/proximal ratio (session wise)
n = 50;
Ratio1 = zeros(Nrats, Ntrials/n);
Ratio2 = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    Distal = arrayfun(@(i) sum(Lfreq(r,i:i+n-1)),1:n:length(Lfreq)-n+1);
    Prox = arrayfun(@(i) sum(Rfreq(r,i:i+n-1)),1:n:length(Rfreq)-n+1); 
    Ratio1(r,:) = Distal./Prox;
    Ratio2(r,:) = Prox./Distal;
end

%All simulations data
Ratio = [Ratio1(:,1:30), Ratio2(:,31:60)];


%Save data to file 
%xlswrite('RatioSalH1.xlsx', Ratio) 


%Perfect trials (session wise)
NSel2 = NSel == 2;
PerfTrials = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    Nperf = NSel2(r,:);
    perf = arrayfun(@(i) sum(Nperf(i:i+n-1)),1:n:length(Nperf)-n+1)';
    PerfTrials(r,:) = perf/n;
end

%Save data to file 
%xlswrite('PerfTrialSalH1.xlsx', PerfTrials) 


%Number of actions per trial (session wise)

ActsTrial = zeros(Nrats, Ntrials/n);
for r = 1:Nrats
    NacRat = NSel(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    ActsTrial(r,:) = Nac;
end

%Save data to file 
%xlswrite('NactsSalH1.xlsx', ActsTrial) 

%% frequencies (L and R lever and LR and RL sequences)

%Individual lever press frequency
n= 50;
Lfsum = zeros(Nrats, Ntrials/n);
Rfsum = zeros(Nrats, Ntrials/n);
LRfsum = zeros(Nrats, Ntrials/n);
RLfsum = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    %left
    NacRat = Lfreq(r,:);
    Nac = arrayfun(@(i) sum(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    Lfsum(r,:) = Nac;
   
    %right
    NacRat = Rfreq(r,:);
    Nac = arrayfun(@(i) sum(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    Rfsum(r,:) = Nac;
    
    %LR sequence
    NacRat = LRfreq(r,:);
    Nac = arrayfun(@(i) sum(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    LRfsum(r,:) = Nac;
    
    %RL sequence
    NacRat = RLfreq(r,:);
    Nac = arrayfun(@(i) sum(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    RLfsum(r,:) = Nac;
    
end

%Save data to file 
%xlswrite('LfsumSalH1.xlsx',Lfsum) 
%xlswrite('RfsumSalH1.xlsx', Rfsum)
xlswrite('LRfsumSal.xlsx', LRfsum)
xlswrite('RLfsumSal.xlsx', RLfsum)


%%  Q values 
%State action 1
n= 50;
QL1mean = zeros(Nrats, Ntrials/n);
QR1mean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    NacRat = QL1(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    QL1mean(r,:) = Nac;
   
    NacRat = QR1(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    QR1mean(r,:) = Nac;
end


%Save data to file 
xlswrite('QR1meanSal.xlsx', QR1mean) 
xlswrite('QL1meanSal.xlsx', QL1mean) 

%State action 2
n= 50;
QL2mean = zeros(Nrats, Ntrials/n);
QR2mean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    NacRat = QL2(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    QL2mean(r,:) = Nac;
   
    NacRat = QR2(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    QR2mean(r,:) = Nac;
end

%Save data to file 
xlswrite('QR2meanSal.xlsx', QR2mean) 
xlswrite('QL2meanSal.xlsx', QL2mean) 

%% Probabilities of actions, and overall proportion of left and right lever presses
%State action 1
n= 50;
prL1mean = zeros(Nrats, Ntrials/n);
prR1mean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    NacRat = PrLs1(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    prL1mean(r,:) = Nac;
   
    NacRat = PrRs1(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    prR1mean(r,:) = Nac;
end


%Save data to file 
%xlswrite('prLs1SalH1.xlsx',pL1mean) 
%xlswrite('prRs1SalH1.xlsx', pR1mean) 

%State action 2
n= 50;
prL2mean = zeros(Nrats, Ntrials/n);
prR2mean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    NacRat = PrLs2(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    prL2mean(r,:) = Nac;
   
    NacRat = PrRs2(r,:);
    Nac = arrayfun(@(i) mean(NacRat(i:i+n-1)),1:n:length(NacRat)-n+1)';
    prR2mean(r,:) = Nac;
end


%Save data to file 
%xlswrite('prLs2SalH1.xlsx',pL2mean) 
%xlswrite('prRs2SalH1.xlsx', pR2mean) 

%% State values

n= 50;
Vs0mean = zeros(Nrats, Ntrials/n);
VsA1mean = zeros(Nrats, Ntrials/n);
VsA2mean = zeros(Nrats, Ntrials/n);
VsE1mean = zeros(Nrats, Ntrials/n);
VsE2mean = zeros(Nrats, Ntrials/n);
VsR1mean = zeros(Nrats, Ntrials/n);
VsR2mean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    %Vs0
    tempData = VS0All(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    Vs0mean(r,:) = Nac;
    %VsA1
    tempData = VSa1All(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsA1mean(r,:) = Nac;
    %VsA2
    tempData = VSa2All(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsA2mean(r,:) = Nac;
    %VsE_reward
    tempData = VSeRAll(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsE1mean(r,:) = Nac;
    %VsE_noReward
    tempData = VSeNoRAll(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsE2mean(r,:) = Nac;
    %VsR_reward
    tempData = VSR1All(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsR1mean(r,:) = Nac;
    %VsR_noReward
    tempData = VSR2All(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    VsR2mean(r,:) = Nac;

end

% %Save data to file 
 xlswrite('Vs0meanAplha.xlsx',Vs0mean) 
 xlswrite('VsA1meanAplha.xlsx', VsA1mean) 
 xlswrite('VsA2meanAplha.xlsx', VsA2mean) 
 xlswrite('VsE1meanAplha.xlsx', VsE1mean) 
  xlswrite('VsE2meanAplha.xlsx', VsE2mean) 
 xlswrite('VsR1meanAplha.xlsx', VsR1mean) 
 xlswrite('VsR2meanAplha.xlsx', VsR2mean) 

%% delta error 

n= 50;
delta0mean = zeros(Nrats, Ntrials/n);
deltaA1mean = zeros(Nrats, Ntrials/n);
deltaA2mean = zeros(Nrats, Ntrials/n);
deltaEmean = zeros(Nrats, Ntrials/n);

for r = 1:Nrats
    %delta 0
    tempData = DELTA0(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    delta0mean(r,:) = Nac;
    %delta A1
    tempData = DELTAA1(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    deltaA1mean(r,:) = Nac;
    %delta A2
    tempData = DELTAA2(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    deltaA2mean(r,:) = Nac;
    %delta E
    tempData = DELTAE(r,:);
    Nac = arrayfun(@(i) mean(tempData(i:i+n-1)),1:n:length(tempData)-n+1)';
    deltaEmean(r,:) = Nac;
end


% %Save data to file 
xlswrite('delta0meanSal.xlsx', delta0mean) 
xlswrite('deltaA1meanSal.xlsx', deltaA1mean) 
xlswrite('deltaA2meanSal.xlsx', deltaA2mean) 
xlswrite('deltaEmeanSal.xlsx', deltaEmean) 
xlswrite('DeltaA1Sal.xlsx', DELTAA1)
xlswrite('DeltaA2Sal.xlsx', DELTAA2)

%% plots with error bars 

%Check paramter moved as planned
clf
plot(lambdaALL, 'k')
ylim([0 1])
vline(1500,'r--')
vline(1600,'r--')

%Data
x = 1:60;

%Saline
yS = xlsread('PerfTrialSal.xlsx'); ySmean = mean(yS);
y2S = xlsread('DPRatioSal.xlsx');    y2Smean = mean(y2S);
y3S = xlsread('NactsSal.xlsx');    y3Smean = mean(y3S);

%Hypothesis data
yLD =PerfTrials ; yLDmean = mean(yLD);
y2LD = Ratio;    y2LDmean = mean(y2LD);
y3LD = ActsTrial;    y3LDmean = mean(y3LD);

%Perfect trials 
%PHASE 1
clf
figure(1)
plot(1:30, yLDmean(1:30), '-g','LineWidth',2)
shadedErrorBar(1:30,yLD(:,1:30),{@mean,@std},'lineprops','-g', 'patchSaturation',0.13)
ylabel('Proportion of perfect trials')
xlabel('Sessions')
set(gca,'fontsize',14)
yline(0.4, 'k--')
hold on 
plot(1:30, ySmean(1:30), '-k','LineWidth',2)
box off
shadedErrorBar(1:30,yS(:,1:30),{@mean,@std},'lineprops','-k', 'patchSaturation',0.13)
ylim([0 1])
xlim([0,30])
set(gcf,'Position',[100 100 530 450])
title('Phase 1: LR')

%PHASE 2
figure(2)
plot(1:30, ySmean(31:60), '-k','LineWidth',2)
ylabel('Proportion of perfect trials')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:30, yLDmean(31:60), '-g','LineWidth',2)
box off
yline(0.4, 'k--')
shadedErrorBar(1:30,yS(:,31:60),{@mean,@std},'lineprops','-k', 'patchSaturation',0.13)
ylabel('Proportion of perfect trials')
xlabel('Sessions')
hold on 
shadedErrorBar(1:30,yLD(:,31:60),{@mean,@std},'lineprops','-g', 'patchSaturation',0.13)
ylim([0 1])
set(gcf,'Position',[100 100 570 450])



%Distal/Proximal ratio PHASE 2
clf
figure(3)
plot(1:30, y2Smean(41:70), '-b','LineWidth',2)
ylabel('Distal/Prox ratio')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:30, y2LDmean(41:70), '-r','LineWidth',2)
yline(0.75, 'k--')
yline(1.25, 'k--')
ylim([0,1.5])
box off
set(gcf,'Position',[100 100 570 450])
shadedErrorBar(1:30,y2S(:,41:70),{@mean,@std},'lineprops','-b', 'patchSaturation',0.13)
shadedErrorBar(1:30,y2LD(:,41:70),{@mean,@std},'lineprops','-r', 'patchSaturation',0.13)


%Number of actions per trial PHASE 2
clf
figure(4)
plot(1:30, y3Smean(41:70), '-b','LineWidth',2)
ylabel('Mean actions per reward')
xlabel('Sessions')
ylim([0,10])
set(gca,'fontsize',14)
set(gcf,'Position',[100 100 570 450])
hold on 
plot(1:30, y3LDmean(41:70), '-r','LineWidth',2)
yline(2, 'k--')
yline(3, 'k--')
box off
shadedErrorBar(1:30,y3S(:,41:70),{@mean,@std},'lineprops','-b', 'patchSaturation',0.13)
shadedErrorBar(1:30,y3LD(:,41:70),{@mean,@std},'lineprops','-r', 'patchSaturation',0.13)


%% Ploting relative frequencies

%Action relative frequency
%Sal
RfmeanS = xlsread('RfsumSal.xlsx');
LfmeanS = xlsread('LfsumSal.xlsx');
TotalActSal = RfmeanS + LfmeanS;
LfmeanS = LfmeanS./TotalActSal;
RfmeanS = RfmeanS./TotalActSal;
%Drug
TotalActD = Lfsum + Rfsum;
LfmeanD = Lfsum./TotalActD;
RfmeanD =  Rfsum./TotalActD;


clf 
figure(1)
plot(41:80, mean(LfmeanS(:,41:80)), '--b','LineWidth',2)
ylabel('Relative frequency')
xlabel('Sessions')
ylim([0 1])
set(gca,'fontsize',18)
hold on 
plot(41:80, mean(RfmeanS(:,41:80)), '-b','LineWidth',2)
vline(40, 'k--')
box off
hold on 
plot(41:80, mean(LfmeanD(:,41:80)), '--r','LineWidth',2)
hold on
plot(41:80, mean(RfmeanD(:,41:80)), '-r','LineWidth',2)

%% Ploting relative sequence frequencies

%Action frequency
RLfmeanS = xlsread('RLfsumSal.xlsx');
LRfmeanS = xlsread('LRfsumSal.xlsx');
TotalSeqSal = RLfmeanS + LRfmeanS;
TotalSeqD = RLfsum + LRfsum;

RLfmeanS =  RLfmeanS./TotalSeqSal;
LRfmeanS =  LRfmeanS./TotalSeqSal;
RLfmeanL1 = RLfsum ./TotalSeqD;
LRfmeanL1 = LRfsum./TotalSeqD;

clf 
figure(1)
plot(41:60, mean(LRfmeanS(:,41:60)), '-b','LineWidth',2)
ylabel('Relative frequency')
xlabel('Sessions')
ylim([0 1])
set(gca,'fontsize',18)
box off
hold on 
plot(41:60, mean(LRfmeanL1(:,41:60)), '-r','LineWidth',2)

figure(2)
plot(41:60, mean(RLfmeanS(:,41:60)), '-b','LineWidth',2)
vline(40, 'k--')
hold on
plot(41:60, mean(RLfmeanL1(:,41:60)), '-r','LineWidth',2)


%% Ploting Q values

%State 1
QL1meanS = xlsread('QL1meanSalH1.xlsx');
QR1meanS = xlsread('QR1meanSalH1.xlsx');

clf
figure(1)
plot(1:70,QL1meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
hold on
plot(1:70, mean(QL1meanS(:,1:70)), '-b','LineWidth',2)
ylabel('Distal action preferences')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:70,QR1meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QR1meanS(:,1:70)), '-g','LineWidth',2)
ylim([-2,10])
xlim([0, 70])
box off
hold on
plot(1:70,QL1mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QL1mean(:,1:70)), '--b','LineWidth',2)
hold on
plot(1:70,QR1mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QR1mean(:,1:70)), '--g','LineWidth',2)
xline(40, 'k--')

%State 2
QL2meanS = xlsread('QL2meanSalH1.xlsx');
QR2meanS = xlsread('QR2meanSalH1.xlsx');

clf
figure(1)
plot(1:70,QL2meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
hold on
plot(1:70, mean(QL2meanS(:,1:70)), '-b','LineWidth',2)
ylabel('Proximal action preferences')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:70,QR2meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QR2meanS(:,1:70)), '-g','LineWidth',2)
ylim([-4,10])
xlim([0, 70])
box off
hold on
plot(1:70,QL2mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QL2mean(:,1:70)), '--b','LineWidth',2)
hold on
plot(1:70,QR2mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(QR2mean(:,1:70)), '--g','LineWidth',2)
xline(40, 'k--')

%% Ploting probabilities
%State 1
prL1meanS = xlsread('prLs1SalH1.xlsx');
prR1meanS = xlsread('prRs1SalH1.xlsx');

clf
figure(1)
plot(1:70,prL1meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
hold on
plot(1:70, mean(prL1meanS(:,1:70)), '-b','LineWidth',2)
ylabel('Probability of distal action')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:70,prR1meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prR1meanS(:,1:70)), '-g','LineWidth',2)
ylim([0,1])
xlim([0, 70])
box off
hold on
plot(1:70,prL1mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prL1mean(:,1:70)), '--b','LineWidth',2)
hold on
plot(1:70,prR1mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prR1mean(:,1:70)), '--g','LineWidth',2)
xline(40, 'k--')


%State 2
prL2meanS = xlsread('prLs2SalH1.xlsx');
prR2meanS = xlsread('prRs2SalH1.xlsx');

clf
figure(1)
plot(1:70,prL2meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
hold on
plot(1:70, mean(prL2meanS(:,1:70)), '-b','LineWidth',2)
ylabel('Probability of proximal action')
xlabel('Sessions')
set(gca,'fontsize',14)
hold on 
plot(1:70,prR2meanS(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prR2meanS(:,1:70)), '-g','LineWidth',2)
ylim([0,1])
xlim([0, 70])
box off
hold on
plot(1:70,prL2mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prL2mean(:,1:70)), '--b','LineWidth',2)
hold on
plot(1:70,prR2mean(:,1:70),'LineWidth',1, 'color', [0,0,0,0.01])
plot(1:70, mean(prR2mean(:,1:70)), '--g','LineWidth',2)
xline(40, 'k--')


%% state values

%State 1
Vs0meanSal= xlsread('Vs0meanSalH1.xlsx');
VsA1meanSal= xlsread('VsA1meanSalH1.xlsx');
VsA2meanSal= xlsread('VsA2meanSalH1.xlsx');
VsE1meanSal= xlsread('VsE1meanSalH1.xlsx');
VsE2meanSal= xlsread('VsE2meanSalH1.xlsx');
VsR1meanSal= xlsread('VsR1meanSalH1.xlsx');
VsR2meanSal= xlsread('VsR2meanSalH1.xlsx');

clf
subplot(2,2,1)
plot(1:80, mean(Vs0meanSal), '-b','LineWidth',2)
ylim ([0,1])
ylabel('State value')
xlabel('Sessions')
title('Vs0')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(Vs0mean), '--r','LineWidth',2)
box off

subplot(2,2,2)
plot(1:80, mean(VsA1meanSal), '-b','LineWidth',2)
ylim ([0,1])
ylabel('State value')
xlabel('Sessions')
title('VsA1')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(VsA1mean), '--r','LineWidth',2)
box off

subplot(2,2,3)
plot(1:80, mean(VsA2meanSal), '-b','LineWidth',2)
ylim ([0,1])
ylabel('State value')
xlabel('Sessions')
title('VsA2')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(VsA2mean), '--r','LineWidth',2)
box off

subplot(2,2,4)
plot(1:80, mean(VsE1meanSal), '-b','LineWidth',2)
ylim ([0,1])
ylabel('State value')
xlabel('Sessions')
title('VsE-Reinf')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(VsE1mean), '--r','LineWidth',2)
box off


%% delta values

%State 1
delta0meanSal= xlsread('delta0meanSalH1.xlsx');
deltaA1meanSal= xlsread('deltaA1meanSalH1.xlsx');
deltaA2meanSal= xlsread('deltaA2meanSalH1.xlsx');
deltaEmeanSal= xlsread('deltaEmeanSalH1.xlsx');

clf

subplot(2,2,1)
plot(1:80, mean(delta0meanSal), '-b','LineWidth',2)
ylim ([-.2,0.6])
xlabel('Sessions')
title('delta 0')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(delta0mean), '--r','LineWidth',2)
box off

subplot(2,2,2)
plot(1:80, mean(deltaA1meanSal), '-b','LineWidth',2)
ylim ([-.2,0.6])
xlabel('Sessions')
title('delta A1')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(deltaA1mean), '--r','LineWidth',2)
box off

subplot(2,2,3)
plot(1:80, mean(deltaA2meanSal), '-b','LineWidth',2)
ylim ([-.2,0.6])
xlabel('Sessions')
title('delta A2')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(deltaA2mean), '--r','LineWidth',2)
box off

subplot(2,2,4)
plot(1:80, mean(deltaEmeanSal), '-b','LineWidth',2)
ylim ([-.2,0.6])
xlabel('Sessions')
title('delta E')
set(gca,'fontsize',18)
hold on 
plot(1:80, mean(deltaEmean), '--r','LineWidth',2)
box off


%% DELTAS per Trial
%State 1
deltaA1Sal= xlsread('DeltaA1SalH1.xlsx');
deltaA2Sal= xlsread('DeltaA2SalH1.xlsx');


clf
subplot(1,4,1)
plot(1:3500, mean(DELTA0(:,1:3500)),'r', 'LineWidth',1.5)
xline(2000, '--k');
shadedErrorBar(1:3500, DELTA0(:,1:3500),{@mean,@std},'lineprops','r', 'patchSaturation',0.02)
ylim ([-0.4,0.75])
xlabel('Trials')
title('delta 0')
set(gca,'fontsize',12)
xlim([0 3500]);
box off

subplot(1,2,1)
plot(1:3500, mean(deltaA1Sal(:,1:3500)),'b', 'LineWidth',1.5)
shadedErrorBar(1:3500, deltaA1Sal(:,1:3500),{@mean,@std},'lineprops','b', 'patchSaturation',0.02)
xline(2000, '--k');
ylim ([-0.4,0.75])
xlabel('Trials')
title('delta distal action')
set(gca,'fontsize',12)
set(gcf,'Position',[100 100 900 450])
xlim([0 3500]);
box off

subplot(1,2,2)
plot(1:3500, mean(deltaA2Sal(:,1:3500)),'b', 'LineWidth',1.5)
shadedErrorBar(1:3500, deltaA2Sal(:,1:3500),{@mean,@std},'lineprops','b', 'patchSaturation',0.02)
xline(2000, '--k');
ylim ([-0.4,0.75])
xlabel('Trials')
title('delta proximal action')
set(gcf,'Position',[100 100 900 450])
set(gca,'fontsize',12)
xlim([0 3500]);
box off

subplot(1,4,4)
xline(2000, '--k');
plot(1:3500, mean(DELTAE(:,1:3500)),'b', 'LineWidth',1.5)
shadedErrorBar(1:3500, DELTAE(:,1:3500),{@mean,@std},'lineprops','b', 'patchSaturation',0.02)
ylim ([-0.4,0.75])
xlim([0 3500]);
xlabel('Trials')
title('delta evaluation')
set(gca,'fontsize',12)
box off