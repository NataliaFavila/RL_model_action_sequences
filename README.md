# RL_model_action_sequences
Model-free reinforcment learning alogrithm with an actor-critic paradigm as an overarching architecture developed to learn sequences of two actions.

The code present in this repository contains the model described in Favila, N., Gurney, K. and Overton, G. P. (2022). The role of NK1 receptors in sequence learning and performance. _Under revision_.

## Model overview 


**Learning**. Reinforcement learning agents learn to estimate the value of actions and states through trial and error in order to maximize reward (Sutton and Barto, 1998). Thus, let V ̂(s_t ) be the estimated value of state s_t at time t; V ̂(s_(t+1) ) the estimated value obtained in the next state s_(t+1) reached after taking action a_i at time t;  r the reinforcer procured after taking action a_i, and γ the discount factor, accounting for the fact that future states are temporally distant and thus less valued. Then, the reward prediction error (RPE), δ_t, is calculated as follows:
δ_t=r+γV ̂(s_(t+1) )-V ̂(s_t)

