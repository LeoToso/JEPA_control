"""CartPole visual environment wrapper."""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym


class ContinuousCartpoleVisual:
    def __init__(self, frame_skip=1, image_size=64, action_range=(-10.0,10.0),
                 mass_cart=1.0, mass_pole=0.1, pole_length=0.5,
                 gravity=9.8, dt=0.02,
                 friction_cart=0.0, friction_pole=0.0,
                 theta_threshold=None,
                 seed=None):
        self.frame_skip=frame_skip; self.image_size=image_size
        self.action_low,self.action_high=action_range
        self.mass_cart=mass_cart; self.mass_pole=mass_pole
        self.pole_length=pole_length; self.gravity=gravity; self.dt=dt
        self.friction_cart=friction_cart; self.friction_pole=friction_pole
        self.theta_threshold = theta_threshold if theta_threshold is not None else 12*2*math.pi/360
        self._env=gym.make('CartPole-v1',render_mode='rgb_array')
        self._env.unwrapped.masscart=mass_cart
        self._env.unwrapped.masspole=mass_pole
        self._env.unwrapped.length=pole_length
        self._env.unwrapped.gravity=gravity
        self._env.unwrapped.tau=dt
        self._env.unwrapped.total_mass=mass_cart+mass_pole
        self._env.unwrapped.polemass_length=mass_pole*pole_length
        self.np_random=np.random.RandomState(seed)
        self.observation_space=gym.spaces.Box(low=0,high=255,shape=(image_size,image_size,3),dtype=np.uint8)
        self.action_space=gym.spaces.Box(low=np.array([self.action_low],dtype=np.float32),high=np.array([self.action_high],dtype=np.float32),dtype=np.float32)
        self.state_space=gym.spaces.Box(low=-np.inf,high=np.inf,shape=(4,),dtype=np.float32)
        self._state=None

    def reset(self, seed=None, init_range=0.05):
        if seed is not None: self.np_random=np.random.RandomState(seed)
        x0=self.np_random.uniform(-init_range,init_range,size=4)
        return self.reset_to_state(x0)

    def reset_to_state(self, state):
        state=np.asarray(state,dtype=np.float64)
        self._env.reset()
        self._env.unwrapped.state=state.copy()
        self._state=state.copy()
        return self._render_obs(), state.astype(np.float32), {}

    def step(self, action):
        # Accept scalar or array of shape (frame_skip,) — one sub-action per physics step.
        # Scalar (or 1-element array) → ZOH: same action for all frame_skip steps.
        action_arr = np.atleast_1d(np.asarray(action, dtype=np.float64)).flatten()
        if action_arr.size == 1:
            action_arr = np.repeat(action_arr, self.frame_skip)
        action_arr = np.clip(action_arr, self.action_low, self.action_high)
        total_reward=0.0; done=False
        for i in range(self.frame_skip):
            state,done=self._physics_step(float(action_arr[i]))
            pos,vel,ang,ang_vel=state
            reward=-(pos**2+0.1*vel**2+10.0*ang**2+0.1*ang_vel**2)
            total_reward+=reward
            if done: break
        self._state=state
        return self._render_obs(), state.astype(np.float32), total_reward, done, {'state':state.copy()}

    def _physics_step(self, force):
        env=self._env.unwrapped; state=env.state
        x,x_dot,theta,theta_dot=state
        M=self.mass_cart; m=self.mass_pole; g=self.gravity; l=self.pole_length; dt=self.dt
        sin_t=math.sin(theta); cos_t=math.cos(theta)
        total_mass=M+m; ml=m*l
        # Viscous cart friction reduces effective force; viscous pole friction damps θ̈.
        # Derived from the Lagrangian: b_c acts on ẋ, b_p acts on θ̇ at the pivot.
        temp=(force - self.friction_cart*x_dot + ml*theta_dot**2*sin_t)/total_mass
        theta_acc=(g*sin_t - cos_t*temp - self.friction_pole*theta_dot/ml)/(l*(4.0/3.0-m*cos_t**2/total_mass))
        x_acc=temp-ml*theta_acc*cos_t/total_mass
        new_x=x+dt*x_dot; new_x_dot=x_dot+dt*x_acc
        new_theta=theta+dt*theta_dot; new_theta_dot=theta_dot+dt*theta_acc
        new_state=np.array([new_x,new_x_dot,new_theta,new_theta_dot],dtype=np.float64)
        env.state=new_state
        x_threshold=2.4
        done=bool(new_x<-x_threshold or new_x>x_threshold or abs(new_theta)>self.theta_threshold)
        return new_state,done

    def _render_obs(self):
        frame=self._env.render()
        if frame is None: return np.zeros((self.image_size,self.image_size,3),dtype=np.uint8)
        try:
            import cv2
            return cv2.resize(frame,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA).astype(np.uint8)
        except ImportError:
            h,w=frame.shape[:2]
            row_idx=(np.arange(self.image_size)*h//self.image_size).astype(int)
            col_idx=(np.arange(self.image_size)*w//self.image_size).astype(int)
            return frame[np.ix_(row_idx,col_idx)].astype(np.uint8)

    def get_state(self): return self._env.unwrapped.state.astype(np.float32)
    def close(self): self._env.close()
    def sample_action(self): return float(self.np_random.uniform(self.action_low,self.action_high))


def make_cartpole_visual(frame_skip=1,**kwargs): return ContinuousCartpoleVisual(frame_skip=frame_skip,**kwargs)



