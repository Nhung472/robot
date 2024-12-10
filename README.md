# Design a algorithm to balance the two wheel robot using Deep Reinforcement Learning

The objective of this project is to train a Two-Wheel Robot (TWR) to balance, move and traverse slopes to reach a destination using Deep Reinforcement Learning in PyBullet physics engine. In this project, we have implemented State-Of-The-Art (SOTA) Deep Reinforcement Learning Algorithms (Deep Q-Network, Soft-Actor Critic, Actor Advantage Critic etc) to accomplish these tasks.

![slope_demo](https://user-images.githubusercontent.com/69728128/234353570-e9368a8a-5388-4a23-b4ae-fe567f65436a.gif)

# Use file check_requirements.py
1. Open terminal
2. Cd to folder have the file check_requirements.py
3. Run:
```
python check_requirements.py
```

## Setup & Installation of Repository
The code is written using Python 3.8 and used the following libraries:
```
matplotlib==3.7.1
numpy==1.23.5
opencv_python==4.7.0.72
pandas==2.0.0
pybullet==3.2.5
PyYAML==6.0
seaborn==0.12.2
tensorflow==2.12.0
tensorflow_probability==0.19.0
torch==1.7.1
```

1. Create conda environment
``` 
conda create -n two_wheel_robot python=3.8 
```
2. Activate conda environment
``` 
conda activate two_wheel_robot
```
3. Install Dependencies
``` 
pip install -r requirements.txt 
```

## Folders
### 1. Object Models
This folder stores the two-wheel robot xml file created in Gazebo. The object models for the slope is also stored here. 

### 2. Configs
This folder stores the configuration files (.yaml) for running the training and testing experiments. 

### 3. Results
This folder stores the training and testing results of the model runs and the saved models.

## Source Code

### 1. robot_environment.py
This file contains the robot's environment that interfaces with the PyBullet environment. It loads the Two-Wheel Robot model into the Pybullet environment and sets up the terrain. The code also retrieves the state observations of the Two-Wheel Robot and controls the robot based on the actions given by the model. This class also defines the reward functions to be given to the agent during training.

### 2. robot_move.py
This file contains the MoveRobot class for moving the robot during training and testing.
This is the main file where all classes in the source code interact with each other during training/testing. The class reads the configuration file parameters to initialise other classes. The Two-Wheel Robot is trained by running through episodes. All training and testing metrics are logged and the results are plotted in the results subdirectories.

### 3. robot_agent.py
This defines the operations and functions for the tensorflow models, the class can update the model weights via backpropagating. The class also enables saving of the tensorflow model weights.

### 4. robot_neural_network.py
This file contain both PyTorch and Tensorflow 2 model structures. It defines the Fully-Connected Networks of the Tensorflow models and the model classes of the PyTorch DQN and SAC. The select_action functions for the models are also defined here to select greedy actions or random actions for balancing the exploration and exploitation trade-off.

### 5. robot_train.py
The main code for training an agent to balance and move the Two-Wheel Robot. It takes in a training configuration file which stores all the training parameters. Users can also enable DIRECT mode (Not using GUI) to speed up training.

### 6. robot_test.py
The main code for testing the trained agent. The user can run this file to validate and test the model to visualise the performance of the trained agent.


## Training Robot Agent
1. Define your training config file by creating a new config.yaml under configs directory. Rename the run name for every new run so that a new subdirectory will be created under results/train/<run_name> for storing the training plots and model weights.

2. After editing the config file, run the following python code to train the agent.
```
python robot_train.py --config-file ./config/main.yaml
```

## Testing Robot Agent
1. After training a model, find the path tp the saved config file in the results/train/<run_name> directory after training to load the trained model weigths path.

2. Run the following code to test the model
```
python robot_test.py --config-file ./results/train/hyperparam_DQN_slope_40m/epsilon_decay_type_linear/hyperparam_DQN_slope_40m.yaml
```

## Running Hyperparameter Tuning
1. To run hyperparameter tuning, edit the hyperparameters in the configuration file

For example in main.yaml:
```
# ======= HYPERPARAMETER TUNING ======= #
hyperparameter_tuning:
  hyperparam_type: 'environment'
  hyperparameter_tuning_variable: 'x_distance_to_goal' 
  hyperparameter_value_list: [5, 10, 15]
```
2. Run the following code with the configuration file
```
python robot_train_hyperparameter_tuning.py --config-file ./config/main.yaml
```






 
