import numpy
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np

def calculate_snr(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(image)
    variance = np.var(image)
    snr = 10 * np.log10(mean_intensity**2 / variance)

    return snr

def calculate_features(image):
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    keypoints, descriptors = orb.compute(image, keypoints)
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    num_features = len(keypoints)
    # print(f"Number of ORB features detected: {num_features}")
    return num_features

def CLAHE(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_image)
    if len(image.shape) == 3:
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    return clahe_image

def white_balance(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    cl = clahe.apply(l)
    balanced_lab_image = cv2.merge((cl, a, b))
    balanced_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2BGR)
    return balanced_image

def Contrast_Up(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=1.25, beta=0)
    return contrasted_image

def Contrast_Down(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=0.75, beta=0)
    return contrasted_image

def Brightness_Up(image):
    brightened_image = cv2.convertScaleAbs(image, alpha=1.0, beta=25)
    return brightened_image

def Brightness_Down(image):
    darkened_image = cv2.convertScaleAbs(image, alpha=1.0, beta=-25)
    return darkened_image

class Q_Learning_Agent:
    def __init__(self, image):
        self.actions = ['WB','C_Up','C_Down','Bs_Up','B_Down','CLAHE']
        self.states = ['F0','F1','F2','F3','F4','F5']
        self.rewards = [-5,-1,1,2,3,4,5]
        self.target_reached = 0
        self.memory = []
        self.image = image
        self.steps = 0
        self.cumulative_reward = 0
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_prob = 0.6
        self.num_episodes = 20
        self.next_img = numpy.zeros_like(self.image)
        self.actions_taken = []

        self.Q = np.zeros((len(self.states), len(self.actions)))


    def check_state(self, image):
        num_of_features = calculate_features(image)
        if num_of_features < 0:
            return 'F0'
        elif num_of_features >= 0 and num_of_features < 20 :
            return 'F1'
        elif num_of_features <=40 and num_of_features > 20:
            return 'F2'
        elif num_of_features <=80 and num_of_features > 40:
            return 'F3'
        elif num_of_features <=200 and num_of_features > 80:
            return 'F4'
        elif  num_of_features > 200:
            return 'F5'

    def get_feature_difference(self, ft1, ft2):
        i1 = self.states.index(ft1)
        i2 = self.states.index(ft2)
        return (i2-i1) * 100

    def update_reward(self, st1, st2):
        #print(st2, st1)
        feature_difference = self.get_feature_difference(st2, st1)
        print(feature_difference)
        if feature_difference < 0:
            return -5
        elif feature_difference == 0:
            return -1
        elif feature_difference <= 20 and feature_difference > 0:
            return 1
        elif feature_difference <= 80 and feature_difference > 20:
            return 2
        elif feature_difference <= 200 and feature_difference > 80:
            return 3
        elif feature_difference <= 400 and feature_difference > 200:
            return 4
        elif feature_difference > 400 :
            return 5

    def update_memory(self, a, s, sd, r):
        self.memory.append([a, s, sd, r])

    def perform_action(self, ind, img_inp):
        self.steps += 1
        if ind == 0:                                                            # 0 --> White Balanced
            denoised = white_balance(img_inp)
            return denoised
        elif ind == 1:                                                          # 1 --> Contrast Up
            denoised = Contrast_Up(img_inp)
            return denoised
        elif ind == 2:                                                          # 2 --> Contrast Down
            denoised = Contrast_Down(img_inp)
            return denoised
        elif ind == 3:                                                          # 3 --> Brightness Up
            denoised = Brightness_Up(img_inp)
            return denoised
        elif ind == 4:                                                          # 4 --> Brightness Down
            denoised = Brightness_Down(img_inp)
            return denoised
        elif ind == 5:                                                          # 5 --> CLAHE
            denoised = CLAHE(img_inp)
            return denoised

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(len(self.actions))                          # Exploration: Choose a random action
        else:
            return np.argmax(self.Q[state, :])                                  # Exploitation: Choose the action with the highest Q-value
    def next_state(self, image ,action):
        self.next_img = self.perform_action(action, image)
        return self.check_state(self.next_img), self.next_img

    def Q_train(self):
        curr_image = self.image

        for episode in range(self.num_episodes):
            init_state = self.check_state(self.image)
            state = self.states.index(self.check_state(self.image))
            
            continuous_positive_reward_counter = 0
            
            for tries in range(100):
                curr_state = self.check_state(curr_image)
                action = self.select_action(state)
                #print("Action : ",action)
                
                # Perform the selected action and observe the next state and reward
                if action == 0:
                    next_state, _ = self.next_state(curr_image,0)
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                elif action == 1:
                    next_state, _ = self.next_state(curr_image,1)
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                elif action == 2:
                    next_state, _ = self.next_state(curr_image,2)
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                elif action == 3:
                    next_state, _ = self.next_state(curr_image,3) 
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                elif action == 4:
                    next_state, _ = self.next_state(curr_image,4)
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                elif action == 5:
                    next_state, _ = self.next_state(curr_image,5)
                    #print(curr_state,next_state)
                    reward = self.update_reward(next_state,curr_state)
                    self.update_memory(self.actions[action],curr_state,next_state,reward)
                    self.cumulative_reward += reward
                    # self.actions_taken.append(action)
                    if reward > 0 and continuous_positive_reward_counter < 10:
                        self.actions_taken.append(action)
                        continuous_positive_reward_counter += 1
                        print(reward)
                        print(self.actions_taken)
                    else:
                        continuous_positive_reward_counter = 0
                        self.actions_taken = []
                #print(self.memory)
                #print('step_reward = ', reward)
                #print('cumulative_reward = ',self.cumulative_reward)   

                # Update the Q-value using the Q-learning update rule 
                self.Q[self.states.index(curr_state), action] = self.Q[self.states.index(curr_state), action] + self.learning_rate * (reward + self.discount_factor * np.max(self.Q[self.states.index(next_state), :]) - self.Q[self.states.index(curr_state), action])
                #print(self.Q)
                state = self.states.index(next_state)  # Move to the next state
                #print("After action: ",curr_state,next_state)
                #print("Feature_Diff: ",self.get_feature_difference(curr_state,init_state))
                #terminating condition
                if self.get_feature_difference(curr_state,init_state) > 200 :
                    #print("Target Reached")
                    break
                if tries == 50:
                    #print("Episode Force Stopped")
                    break

if __name__ == "__main__":
    input_img = cv2.imread('C:\\Users\\Asus\\Downloads\\frame_711.jpg')
    print(calculate_features(input_img))
    A1 = Q_Learning_Agent(input_img)
    A1.Q_train()
    print(A1.actions_taken)
    print(len(A1.actions_taken))
    print(calculate_features(A1.next_img))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(A1.next_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')

    plt.show()