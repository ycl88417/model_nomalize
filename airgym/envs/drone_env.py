import setup_path
import airsim
import numpy as np
import math
import time
import random
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image

thresh_dist = 10

def nomalize(x):
    nums = np.linalg.norm(x)
    for i in range(len(x)):
        x[i] = x[i]  / nums
    return x
#################################
class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "orientation": np.zeros(3),
            "prev_orientation":np.zeros(3)           
        }
        self.threshold = thresh_dist
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)
        self._setup_flight()
        self.image_request = airsim.ImageRequest("1",airsim.ImageType.Scene, False, False) #(相机ID，图片类型，是否使用pixels_as_float（pfm格式），是否使用压缩图片)

    def __del__(self):
        self.drone.reset()
        

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(np.random.randint(-4,-1), 3).join()
        self.drone.moveByVelocityAsync(1, 0, 0,1).join()

    def transform_obs(self, responses):
        img1d = np.fromstring(responses[0].image_data_uint8,dtype=np.uint8)
        img2d = img1d.reshape(responses[0].height,responses[0].width,3)
        image = Image.fromarray(np.uint8(img2d))
        im_final = np.array(image.resize((128,72)).convert("RGB"))
        return im_final.reshape([128, 72, 3])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])

        self.drone_state = self.drone.getMultirotorState()
        image = self.transform_obs(responses)
        ###################################
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state["prev_orientation"] = self.state["orientation"]
        self.state["orientation"] =  self.drone_state.kinematics_estimated.orientation
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        ##############################
        position = self.drone_state.kinematics_estimated.position
        position = [position.x_val, position.y_val, position.z_val]
        position = nomalize(position)
        ##############################
        velocity = self.drone_state.kinematics_estimated.linear_velocity
        velocity = [velocity.x_val, velocity.y_val, velocity.z_val]
        velocity = nomalize(velocity)
        ##############################
        ##############################
        dist = []
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )
        target = [ 
                np.array([90,-5,0]),
                np.array([33,90,0]),
                np.array([110,60,0]),
                np.array([160,46,0]),
                np.array([130,150,0]),
                np.array([115,217,0]),
                ]
        for i in range(0,len(target)):
            dist.append(float(format(np.linalg.norm(quad_pt-target[i]),'.2f')))
        dist = nomalize(dist)
        ##############################
        obs = {
                'position': np.array(position),
                'velocity': np.array(velocity),
                'waypoint': np.array(dist),
                'image': image,
                }
        return obs

    def _do_action(self, action):
        
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        #drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
        #yaw_mode = airsim.YawMode(is_rate = True,yaw_or_rate = degree)
        self.drone.moveByVelocityAsync(
            quad_vel.x_val+quad_offset[0],
            quad_vel.y_val+quad_offset[1],
            quad_vel.z_val+quad_offset[2],
            1,
            airsim.DrivetrainType.ForwardOnly,
        ).join()
        '''
        self.drone.moveByVelocityAsync(
             quad_offset[0],
             quad_offset[1],
             quad_offset[2],
            1,
            airsim.DrivetrainType.ForwardOnly,
        ).join()
        '''


    def _compute_reward(self):
        
        thresh_dist = self.threshold     #TARGET範圍
        ################################ WayPoint 順序
        target = [ 
                np.array([90,-5,0]),
                np.array([33,90,0]),
                np.array([110,60,0]),
                np.array([160,46,0]),
                np.array([130,150,0]),
                np.array([115,217,0]),
                ]
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )
        quad_pt_pre = np.array(
            list(
                (
                self.state["prev_position"].x_val,
                self.state["prev_position"].y_val,
                self.state["prev_position"].z_val,
                )
                )
            )
        ######################initialize

        dist = 0
        reward = 0


        ###################### reward
      
        #################

        if self.state["collision"]: #碰撞
            reward = -100
        else:
            dist = []
            for i in range(0,len(target)):
                dist.append(float(format(np.linalg.norm(quad_pt-target[i]),'.2f')))
            dist_min = min(dist)    #距最近target距離
            min_index = np.argmin(dist) #最近target index
            Stay_ = np.linalg.norm(quad_pt-quad_pt_pre)
            if dist_min < thresh_dist:
                reward = 100
                print("arrive")
                if Stay_ < 0.5:
                    reward = -100
                    print("stay")
                if  min_index == 5:
                    reward = 200
                    print("end")
            elif quad_pt[2] > 5 or quad_pt[2] < -15:
                reward = -100
                print("out of range")
            elif Stay_ < 0.5:
                reward = -100
                print("stay")
            
            else:
                reward = - dist_min * 0.1
        done = 0
        if reward > 199:
            done = 1
            time.sleep(1)
        if reward <= -10:
            done = 1
            time.sleep(1)
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done= self._compute_reward()
        return obs, reward, done,self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        '''
        if action == 0:
            degree = 0            
            quad_offset =(self.step_length, 0, 0)
        elif action == 1:
            degree = 30
            quad_offset = (self.step_length, self.step_length*math.cos(degree), 0)
        elif action == 2:
            degree = 60
            quad_offset = (self.step_length, self.step_length*math.cos(degree), 0)
        elif action == 3:
            degree = -30
            quad_offset = (self.step_length, self.step_length*math.cos(degree), 0)
        elif action == 4:
            degree = -60
            quad_offset = (self.step_length, self.step_length*math.cos(degree), 0)
        elif action == 5:
            degree = 0
            quad_offset = (0, 0, self.step_length)
        elif action == 6:
            degree = 0
            quad_offset = (0, 0, -self.step_length)            
       '''
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 2:
            quad_offset = (0, self.step_length, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        elif action == 4:
            quad_offset = (0, 0, -self.step_length)
        elif action == 5:
            quad_offset = (0, 0, self.step_length)
        

        return quad_offset


