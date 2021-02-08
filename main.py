import argparse
from itertools import count
import os
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from trpo import trpo_step
from utils import *
from autograd_hacks import *
from sever import *
from statsmodels.robust.scale import huber
import numpy as np
import time

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G', help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Swimmer-v3", metavar='G', help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G', help='gae parameter (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G', help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G', help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G', help='damping (default: 1e-1) of CG method')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=25000, metavar='N', help='batch size (default: 25000)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 1)')
parser.add_argument('--attack_norm', type=int, default=-1, help='delta: size of reward perturbation in adversarial episode (default: -1, meaning no attack)')
parser.add_argument('--eps', type=float, default=0.01, help='fraction of adversarial episodes per iteration (default: 0.01)')
parser.add_argument('--sever', type=int, default=0, help='whether to use FPG instead of TRPO: 1 for yes, 0 for no')
parser.add_argument('--max_iter_num', type=int, default=1000, help='num of policy gradient iterations (default: 1000)')
parser.add_argument('--random', type=int, default=0, help='whether to use random reward attack (default: 0)')

args = parser.parse_args()

if not os.path.isdir('./training_log/'): os.makedirs('./training_log/')
if not os.path.isdir('./save_model/'): os.makedirs('./save_model/')    
log_prefix = './training_log/' + args.env_name + "_attacknorm_" + str(args.attack_norm) + "_sever_" + str(args.sever) + "_eps_" + str(args.eps) + "_seed_" + str(args.seed)
model_prefix = './save_model/' + args.env_name + "_attacknorm_" + str(args.attack_norm) + "_sever_" + str(args.sever) + "_eps_" + str(args.eps) + "_seed_" + str(args.seed)

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state space size: ', num_inputs)
print('action space size: ', num_actions)
env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
# change the variance of the Gaussian policy for these two environments for better performance
if args.env_name == "HalfCheetah-v3": policy_net.log_std = -1
if args.env_name == "Ant-v3": policy_net.log_std = -2
    
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    n = actions.size(0)
    values = value_net(states)

    ############## GAE ###############
    returns = torch.Tensor(n,1)
    deltas = torch.Tensor(n,1)
    advantages = torch.Tensor(n,1)
    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]
        
    ##################################

    
    ###################### Sever ############################
    

    if args.sever == 1:
        add_hooks(policy_net)
        clear_backprops(policy_net)
        policy_net.zero_grad()
        action_means, action_log_stds, action_stds = policy_net(states)
        log_policy = normal_log_density(actions, action_means, action_log_stds, action_stds)
        torch.autograd.grad(log_policy.mean(), policy_net.parameters())
        ## compute gradient of log policy for every single data point, the trick only works for linear and conv layers
        compute_grad1(policy_net, loss_type='mean')
        actor_grad_logp = []
        for param in policy_net.parameters():
            actor_grad_logp.append(param.grad1.view(param.grad1.shape[0],-1))
        actor_grad_logp = torch.cat(actor_grad_logp,1)
        remove_hooks(policy_net)
        policy_net.zero_grad()
        
        ## standardize the advantage estimate for stable training. Anticipating outliers, use huber's robust estimate of mean and std instead of vanilla sample mean and std.
        h = huber
        h.maxiter = 100
        try:
            mean, std = h(advantages)
        except:
            "huber failed."
            mean = advantages.mean()
            std = advantages.std()
        advantages = (advantages - mean) / std
        actor_loss_grad = actor_grad_logp*(advantages) # vanilla policy gradient
        start_time = time.time()
        
        ## robust CG procedure
        search_dir, indices = Sever_CG(actor_loss_grad, actor_grad_logp, n, nsteps = 10 , r = 4, p = args.eps/2)
    else:
        advantages = (advantages - advantages.mean()) / advantages.std()
        indices = list(range(n))
        search_dir = None
    
    #########################################################
    
    
    # Use same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(states[indices])

        value_loss = (values_ - returns[indices]).pow(2).mean()
#         print('value loss:',value_loss)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    action_means, action_log_stds, action_stds = policy_net(states[indices])
    fixed_log_prob = normal_log_density(actions[indices], action_means, action_log_stds, action_stds).data.clone()
    
    # Policy loss
    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(states[indices])
        else:
            action_means, action_log_stds, action_stds = policy_net(states[indices])
                
        log_prob = normal_log_density(actions[indices], action_means, action_log_stds, action_stds)
        action_loss = -advantages[indices] * torch.exp(log_prob - fixed_log_prob)
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(states[indices])

        mean0 = mean1.data
        log_std0 = log_std1.data
        std0 = std1.data
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping, xinit = search_dir)
    num_attack = args.batch_size*args.eps
    return 1-sum(1 for i in indices if i < num_attack)/num_attack ## fraction of outlier detected

running_state = ZFilter((num_inputs,), clip=5)

all_rewards = []
all_detection_ratios = []
for i_iteration in range(args.max_iter_num):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    adversarial = True # the first episode of each iteration is adversarial
    if num_steps >= args.eps * args.batch_size: adversarial = False
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)
        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            if adversarial: 
                if args.random==1:
                    reward = np.random.normal()
                reward = -args.attack_norm * reward
            num_steps +=1
            if num_steps >= args.eps * args.batch_size: adversarial = False

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    all_rewards.append(reward_batch)
    batch = memory.sample()
    all_detection_ratios.append(update_params(batch))
#     print('detection ratio:',all_detection_ratios[-1])
    if i_iteration % args.log_interval == 0:
        
        np.savetxt(log_prefix + "_all_rewards.csv", all_rewards, delimiter=",")
        np.savetxt(log_prefix + "_detection_ratio.csv", all_detection_ratios, delimiter=",")

        print('Iteration {}\tNum of Episode: {}\tAverage reward {:.2f}'.format(
            i_iteration, num_episodes, reward_batch))

ckpt_path = model_prefix + '_model.pth.tar'
torch.save(policy_net.state_dict(), ckpt_path)
mean_path = model_prefix + '_running_mean.csv'
std_path = model_prefix + '_running_std.csv'

np.savetxt(mean_path, running_state.rs.mean, delimiter=",")
np.savetxt(std_path, running_state.rs.std, delimiter=",")
