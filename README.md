#ACCWN-DRL
==========

This directory and its subdirectories contain source code for the ACCWN-DRL work and support routines for the NS3Gym.

Base files and main files :
1. sim.cc contain ns3 scripts for creating network configuration 
2. tcp-rl.cc, tcp-rl.h, tcp-rl-env.cc, tcp-rl-env.h are the files for ns3gym environment
3. actor-critic_continuous.py contain modules for agent, n-step agent, basic RL implementation, deep neural network creation, etc.
4. tcp_base.py contain modules for environment interaction which can be time based and event based [refer to ns3-gym .cc and .h files to understand about different variables in the environment and it's interactions]
5. n_step_td_agent.py is the file contain n-step TD algorithm. Specify the port number, duration, number of iteractions, etc. The 'filename' variable in the python script must be initialised with proper path location. 

	
##Run the ACCWN-DRL
===================

-- Please copy the folder into scratch of the ns3-gym
-- In order to run it, please execute in two terminals:


#Terminal 1:
cd ./scratch/Actor-Critic-ML
./waf --run "Actor-Critic-ML"


#Terminal 2:
cd ./scratch/Actor-Critic-ML
python3 n_step_td_agent.py

-- After running the python files, .csv files consisting of network states and all observations are created in the folder. Use the observations for result analysis.

#NS3GYM
=======

#Basic Interface
1. The 'gym.make('ns3-v0')' starts ns-3 simulation script located in current working directory.

'''

import gym
import ns3gym
import MyAgent

env = gym.make('ns3-v0')
obs = env.reset()
agent = MyAgent.Agent()

while True:
	action = agent.get_action(obs)
	obs, reward, done, info = env.step(action)
	
	if done:
		break
env.close()

'''
2. Any ns-3 simulation script can be used as a Gym enviromnet. The generic ns3-gym interface allows to observe any variable or parameter in a simulation.

'''

Ptr<OpenGymSpace> GetObservationSpace();
Ptr<OpenGymSpace> GetActionSpace();
Ptr<Open GymDataContainer> GetObservation();
float getReward();
bool GetGameOver();
std::string GetExtraInfo();
bool ExecuteActions(Ptr<OpenGymDataContainer> action);

'''
