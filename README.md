# ppo-car-racing

Solving the Gym Car Racing with Proximal Policy Optimization (PPO)

## Environment Description

A description of the environment can be found in the [Gymnasium Documentation](https://gymnasium.farama.org/environments/box2d/car_racing/).

## Preprocessing the Observation Space

The default observation space is a top-down 96x96 RGB image of the car and race track, which includes some indicators at the bottom of the window along with the state RGB buffer. This bar includes information on the true speed, four ABS sensors, steering wheel position, and gyroscope. These indicators do not add meaningful information about the current state since it's so small and not very readable.

I employed some changes to the observation space:

- removed the bottom panel from observation
- resized observation space to 84x84 RGB image
- converted RGB image to grayscale
- stacked 4 consecutive frames together

The image below shows the original observation (left) and the resulting frame for training on the far right.
![preprocessing](https://github.com/aibenStunner/ppo-car-racing/assets/42221332/c96080f2-d169-4645-9845-0c02db363b72)

### Further modifications

Usually, **clipping the action** helps achieve a lower variance which is generally better for continuous control tasks like this one.

The **MaxAndSkipEnv** wrapper skips 4 frames by default, repeats the agent's last action on the skipped frames, and sums up the rewards in the skipped frames. Such a **frame-skipping technique** should considerably speed up the algorithm because the environment step is computationally cheaper than the agent's forward pass [[1]](#references).

**Normalizing the observation space and reward** is also a good idea. The observation space is used as input to the neural network of the agent, and normalizing the input is beneficial for many reasons (e.g. increases convergence speed, aids computer precision, prevents divergence of parameters, etc.) [[2]](#references).

Also, normalizing the returns improves stability. We can tell from backpropagation equations that the returns directly affect the gradients and thus, we would like to keep its values in a specific convenient range. This prevents backpropagation from leading the network weights to extreme values [[3]](#references).

## Why PPO?

PPO is one of the most popular and robust Deep Reinforcement Learning (DRL) algorithms. It can be adapted to be asynchronous, which gives us the ability to leverage vector (parallel) environments to improve convergence by reducing the correlation between samples and running reasonably fast.

## Installation

1. Download and install Python 3.9.5 using [pyenv](https://github.com/pyenv/pyenv):

```sh
    $ pyenv install 3.9.5
```

2. Create virtualized environment:

```sh
    $ pyenv virtualenv 3.9.5 ppo-car-racing
```

3. Use [poetry](https://github.com/python-poetry/poetry) as the package manager to install all dependencies:

```sh
    $ poetry install --no-root
```

4. Run the following to get list arguments for environment (algorithm specific (hyperparameters) and experiment arguments)

```sh
    $ python train.py --help
```

5. To train the agent with [wandb](https://wandb.ai/site) tracking and video captures, run:

```sh
    $ python train.py --track --capture-video
```

## Preprocessing vs No Preprocessing

In the graph below, we notice a great improvement in the episodic return when the above preprocessing techniques are applied to the environment.

<p align="center">
  <img width="600" height="400" src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/b50b9d86-44fd-4163-98ea-ff5cbf772dcd" />
</p>

## Results

Here are some intriguing behaviors I discovered in my trained model.

### Going Backwards

The agent recovers from slipping but ends up going backward in the wrong direction.

<p align="center">
  <img src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/24342d4a-2219-4e83-bb11-8d7dfb1000a5" />
</p>

### Recovery

Here, the agent recovers form slipping, and returns to right on track

<p align="center">
  <img src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/346db1e4-2896-4028-af59-1e9315d87a6c" />
</p>

### Double Recovery

This is rather interesting behavior exhibited by the agent. The agent first slips and spins but spins again and returns right on track.

<p align="center">
  <img src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/b0b1965f-3e45-42a6-8a1f-e0898964a4ec" />
</p>

### Safe behavior

This is a behavior exhibited by the agent when it does not want to go out of the track while learning. It is efficient if the agent wants to only over all the track and doesn't care about time or speed.

<p align="center">
  <img src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/91a76d85-92ff-458a-a3d7-1d13142b6cee" />
</p>

### The Racer!

Finally, the agent solves the track.

<p align="center">
  <img src="https://github.com/aibenStunner/ppo-car-racing/assets/42221332/fe1b9393-9cae-4496-a7ec-596cc9f924b5" />
</p>

## Training Graphs

![graphs](https://github.com/aibenStunner/ppo-car-racing/assets/42221332/0fb21815-2a55-4e9a-9bec-26d65bba8d7d)

## References

[1] [Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/) \
[2] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) \
[2] [Learning values across many orders of magnitude](https://arxiv.org/pdf/1602.07714.pdf) \
[CleanRL](https://docs.cleanrl.dev/)
