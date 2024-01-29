import gym
from gym.spaces import Box

import cv2
import numpy as np


cv2.ocl.setUseOpenCL(False)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_skip: int = 8):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.frame_skip = frame_skip

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)
        self.obs_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`. Repeat action, sum reward, and max over last observations."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.game_over = terminated

            if terminated or truncated:
                break
            if t == self.frame_skip - 2:
                self.obs_buffer[1] = obs
            elif t == self.frame_skip - 1:
                self.obs_buffer[0] = obs

            # Note that the observation on the
            # terminated=True or truncated=True frame doesn't matter

        return self._get_obs(), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = self.obs_buffer[0]

        return obs


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        ).astype(np.uint8)


class CropObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_top, crop_bottom, crop_left, crop_right):
        super(CropObservationWrapper, self).__init__(env)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right

        # Update observation space after cropping
        original_observation_space = env.observation_space
        low = np.array(
            [
                original_observation_space.low[
                    self.crop_top: self.crop_bottom, self.crop_left: self.crop_right
                ]
            ]
        )
        high = np.array(
            [
                original_observation_space.high[
                    self.crop_top: self.crop_bottom, self.crop_left: self.crop_right
                ]
            ]
        )
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=original_observation_space.dtype,
        )

    def observation(self, observation):
        # Crop the observation
        cropped_observation = observation[
            self.crop_top: self.crop_bottom, self.crop_left: self.crop_right
        ]
        return cropped_observation


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert the image observation from RGB to gray scale.

    Example:
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'))
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'), keep_dim=True)
        >>> env.observation_space
        Box(0, 255, (96, 96, 1), uint8)
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim

        assert (
            isinstance(self.observation_space, Box)
            # and len(self.observation_space.shape) == 3
            and self.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations
        """
        import cv2

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
