# Balancing, Moving, and Traversing Slopes with a Two-Wheel Robot using Deep Reinforcement Learning

The objective of this project is to train a Two-Wheel Robot (TWR) to balance, move and traverse slopes to reach a destination using Deep Reinforcement Learning in PyBullet physics engine. In this project, we have implemented State-Of-The-Art (SOTA) Deep Reinforcement Learning Algorithms (Deep Q-Network, Soft-Actor Critic, Actor Advantage Critic etc) to accomplish these tasks.
Project Report can be found [here](https://drive.google.com/file/d/1H8NA9CpchFa2E4XIwNDL3Hk-ycZIH1lS/view?usp=sharing).

This project was done by: [Ng Zhili](https://github.com/ngzhili), [Foo Jia Yuan](https://github.com/epsilonfunction), [Natasha Soon](https://github.com/natashasoon).

![slope_demo](https://user-images.githubusercontent.com/69728128/234353570-e9368a8a-5388-4a23-b4ae-fe567f65436a.gif)

# Cách sử dụng file check_requirements.py
1. Mở terminal
2. Di chuyển đến thư mục chứa file check_requirements.py
3. Chạy lệnh sau:
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

## Training Robot Agent
1. Define your training config file by creating a new config.yaml under configs directory. Rename the run name for every new run so that a new subdirectory will be created under results/train/<run_name> for storing the training plots and model weights.

For example in main.yaml:
```
# ======== CONFIGURATION FILE FOR TRAINING / TESTING ========
# RENAME THE RUN NAME FOR A NEW RUN
run: 
  name: DQN_Train


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

## Video Demo
https://user-images.githubusercontent.com/69728128/233765311-ea57ea41-e370-46c6-8069-491c71ef3d8d.mp4





