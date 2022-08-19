# RL_model_action_sequences
Model-free reinforcment learning alogrithm with an actor-critic paradigm as an overarching architecture developed to learn sequences of two actions.

The code present in this repository contains the model described in Favila, N., Gurney, K. and Overton, G. P. (2022). The role of NK1 receptors in sequence learning and performance. __in revision_.

## Model overview 
**Learning**. Reinforcement learning agents learn to estimate the value of actions and states through trial and error in order to maximize reward (Sutton and Barto, 1998). Thus, let V ̂(s_t ) be the estimated value of state s_t at time t; V ̂(s_(t+1) ) the estimated value obtained in the next state s_(t+1) reached after taking action a_i at time t;  r the reinforcer procured after taking action a_i, and γ the discount factor, accounting for the fact that future states are temporally distant and thus less valued. Then, the reward prediction error (RPE), δ_t, is calculated as follows:
δ_t=r+γV ̂(s_(t+1) )-V ̂(s_t)

This term is coding the difference between the expected value, V ̂(s_t ), and the discounted value of the state reached, γV ̂(s_(t+1) ), plus the reward obtained, r . 

Let α<1 be a learning rate for updating state values, and β<1 the learning rate for updating a set of action preferences, z(a_i,s_t ) , which are used to choose between actions according to the policy defined below. The RPE is used to update the estimated value of the states and action preferences in the following way:

V ̂(s_t )=V ̂(s_t )+αδ_t

z(a_i,s_t )=z(a_i,s_t )+βδ_t


**Choice**. Action selection was performed via a policy of softmax selection using the action preferences. In this policy, the probability of an action is given by:
π(a_i,s_t )=e^(z(a_i,s_t))/e^∑▒〖z(a_i,s_t)〗 
such that actions with higher values are more likely to be selected, but not in a deterministic way, so there is still a small probability that other actions will be picked, promoting exploration.


**Credit assignment**. To capture the credit assignment problem, in which rats initially assign credit to the proximal response performed right before the delivery of the reward rather than to the whole action sequence, we added eligibility traces to the action preferences (Sutton and Barto, 1998). Eligibility traces account for the fact that temporally distant actions from the reinforcer are less affected by the RPE than those closer to it. 
To implement them, we added a memory variable, e(a_i,s_t), associated with each action-state pair. Let λ be a decay parameter, controlling how much previous actions are affected by the current RPE, and γ the discount factor previously mentioned. Then, at each time step, if an action is performed, its eligibility trace increased to 1 and the eligibility trace of the other action decayed by a factor of γλ. That is:

e(a_i,s_t )= {■(γλe(a_i,s_t )  if a_i  was not perfomed@1             if a_i  was perfomed

If λ=1, all previously performed actions are remembered perfectly and all are given credit for the reward. If λ=0, then only the most recently performed action is given credit, and it is the only one affected by the RPE. 
The addition of the memory variable e(a_i,s_t) makes the update of the action preferences in the following way:
z(a_i,s_t )=z(a_i,s_t )+ αδ_t e(a_i,s_t)
Thus, eligibility traces modulate which actions performed are eligible to undergo learning changes produced by RPE δ_t. 

## Simulations 
We ran two groups of simulations to reproduce the structure of sequence learning experiments.In the first simulation, replicating a reversal learning experiment, simulated agents were trained to perform a two-action sequence for 30 sessions, and then, in a second phase, the learning contingency was reversed, such that the agents had to reverse the order of the actions to obtain the reward for another 30 sessions. In the second simulation, replicating a estable performance experiment, simulated agents were trained to perform a two-action sequence for 30 sessions, and then, in the second phase they were kept performing the same sequence for another 20 sessions. All sessions in both simulations lasted until 50 rewards were obtained. During the first 100 trials of the second phase of both simulations, different parameters of the model were modified to try to simulate the effects produced by the NK1 antagonist injection on the experimental animals’ performance (Favila, Gurney & Overton, __in revision__). 

## Using the code

The scripts are divided into the two simulations performed for the control group and the experimental group, and they allow the user to replicate the the figures in Favila, Gurney & Overton (__in revision__).