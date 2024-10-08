import copy
import numpy as np
from typing import Optional

class BaseJOSHAgent:
    def __init__(self, messages=None):
        if not messages:
            self.messages = []
            self.messages_internal = []
        else:
            self.messages = copy.deepcopy(messages)
            self.messages_internal = copy.deepcopy(messages)
        self.recent_actions = []

    def step(self, **kwargs):
        """
        Returns
        pass_to_customer: if the agent has ended their turn and chooses to pass to the customer
        """
        self.add_message({'role': 'assistant', 'content':'hello!'})
        self.recent_actions = ['say hello']
        pass_to_customer = True
        return pass_to_customer
    
    def add_message(self, message):
        self.messages.append(message)
        self.messages_internal.append(message)

class BaseJOSHUser:
    def __init__(self):
        pass

    def step(self, agent:BaseJOSHAgent, **kwargs):
        """
        Returns 
        agent: the agent passed in
        conversation_over: whether or not the user has ended the conversation
        """
        agent.add_message({'role': 'user', 'content':'hi there!'})
        conversation_over = False
        return agent, conversation_over
    
def trim_user_msg(messages):
    if len(messages) == 0:
        return []
    for idx, dic in enumerate(reversed(messages)):
        if dic.get('role')=='user':
            continue
        break
    if idx == 0:
        return messages
    return messages[:-1*idx]

class Node:
    def __init__(self, agent:BaseJOSHAgent, parent: Optional["Node"]=None):
         self.agent = agent
         self.conversation_over = False
         self.parent = parent
         self.left = None
         self.right = None
         self.is_successful = False
         self.is_golden_path = False

    def get_leaves(self):
         if not self.left and not self.right and not self.conversation_over:  # If the node is a leaf
             return [self]

         leaves = []

         if self.left:
             leaves.extend(self.left.get_leaves())
         if self.right:
             leaves.extend(self.right.get_leaves())

         return leaves
    
    def get_tree(self):
        tree = [(trim_user_msg(copy.deepcopy(self.agent.messages_internal)), self.is_successful, self.is_golden_path)]

        if self.left:
            tree.extend(self.left.get_tree())
        if self.right:
            tree.extend(self.right.get_tree())

        return tree

class BaseRewards:
    def __init__(self, rewards):
        self.rewards = rewards
    def __len__(self):
        return len(self.rewards)
    def is_reward(self, agent_actions):
        got_reward = False
        rewards_to_delete = []
        for agent_action in agent_actions:
            got_reward_tmp = agent_action in self.rewards
            got_reward = got_reward or got_reward_tmp
            if got_reward_tmp:
                reward_index = self.rewards.index(agent_action)
                rewards_to_delete.append(self.rewards[reward_index])
        return got_reward, rewards_to_delete

    def delete_reward(self, rewards_to_delete):
        for reward_to_delete in rewards_to_delete:
            self.rewards.remove(reward_to_delete)

class JOSH():
    def __init__(self, rewards:BaseRewards, agent_step, user_step, add_error_message, root_agent: BaseJOSHAgent, user, beam_size=8, max_turn_tries=10, agent_model=None, agent_tokenizer=None, agent_env=None, debug=False):
        self.agent_step = agent_step
        self.user_step = user_step
        self.add_error_message = add_error_message
        self.root = Node(root_agent, None)
        
        self.current_reward = 0.0
        self.beam_size = beam_size
        self.training_examples = []
        self.max_turn_tries = max_turn_tries
        self.rewards = rewards
        self.num_total_rewards = len(rewards)
        self.golden_agent = None
        self.agent_model = agent_model
        self.agent_tokenizer = agent_tokenizer
        self.agent_env = agent_env
        self.user = user
        self.debug = debug

    def set_root_agent(self, agent):
        self.root.agent = agent

    def set_success_path(self, success_node):
        success_node.is_successful = True
        if success_node.parent is None:
            return
        self.set_success_path(success_node.parent)
        return 
    
    def set_golden_path(self, success_node):
        success_node.is_golden_path = True
        if success_node.parent is None:
            return
        self.set_golden_path(success_node.parent)
        return 
    
    def step_user(self):
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return True
        if self.debug:
            print(f'Running {len(leaves)} users')
        for leaf in leaves:
            leaf.agent, end_conversation = self.user_step(self.user, leaf.agent)
            leaf.conversation_over = end_conversation

        leaves = np.array(self.root.get_leaves())
        return len(leaves)==0
    
    def step_agent(self):
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return True
        # Step for each leaf
        if self.debug:
            print(f'Running {len(leaves)} agents')
        count = 0
        done = np.array([False]*len(leaves))
        training_examples = []
        collapse_root_to = None
        successful_leaves = []
        while count < self.max_turn_tries:
            unfinished_leaf_indices = np.where(done==False)[0]
            if len(unfinished_leaf_indices)==0:
                break
            unfinished_leaves = leaves[unfinished_leaf_indices]

            turn_finished = []
            for lf in unfinished_leaves:
                try:
                    lf.agent, pass_to_customer = self.agent_step(agent=lf.agent, model=self.agent_model, tokenizer=self.agent_tokenizer, env = self.agent_env)
                except:
                    pass_to_customer=None
                    
                if pass_to_customer is None:
                    turn_finished.append(True)
                    lf.conversation_over = True
                else:
                    turn_finished.append(pass_to_customer)


            for idx, turn in enumerate(unfinished_leaves):
                if turn_finished[idx]:
                    done[unfinished_leaf_indices[idx]] = True

                got_reward, rw_to_delete = self.rewards.is_reward(turn.agent.recent_actions)
                if got_reward:
                    successful_leaves.append(turn)
                    if not collapse_root_to:
                        if self.debug:
                            print(f'🌟 Got reward')
                        collapse_root_to = unfinished_leaves[idx]
                        rewards_to_delete = copy.deepcopy(rw_to_delete)
                        if len(self.rewards)==1:
                            self.golden_agent = turn.agent
                    
                if count+1 == self.max_turn_tries:
                    turn.agent = self.add_error_message(turn.agent)

            count += 1
        
        if collapse_root_to:
            if self.debug:
                print(f'🪓👷 Collapsing tree')
            # set the descendence of all successful leaves as successful
            for leaf in successful_leaves:
                self.set_success_path(leaf)
            
            self.rewards.delete_reward(rewards_to_delete)
            self.set_golden_path(collapse_root_to)
            training_examples = self.root.get_tree()
            for ex in training_examples:
                if ex not in self.training_examples:
                    self.training_examples.append(ex)
            self.root = collapse_root_to
            self.root.parent=None
            self.root.is_successful = False
            leaves = [self.root]
        
        if self.num_total_rewards != 0:
            self.current_reward = (self.num_total_rewards-len(self.rewards))/self.num_total_rewards
        else:
            self.current_reward = 0.0
        all_done = len(self.rewards) == 0
        return all_done


    def expand_tree(self):
        leaves = np.array(self.root.get_leaves())
        make_more_leaves = len(leaves)*2<=self.beam_size
        # Add messages to each leaf
        if make_more_leaves:
            if self.debug:
                print(f'🌲 Expanding tree to {len([l for l in leaves if not l.conversation_over])*2} leaves')
            for leaf in leaves:
                # If the user ended the conversation, kill the leaf and keep going
                if leaf.conversation_over:
                    continue
                # Extend leaves
                leaf.left = Node(copy.deepcopy(leaf.agent), parent=leaf)
                leaf.right = Node(copy.deepcopy(leaf.agent), parent=leaf)
        elif self.debug:
            print(f'🎄 Tree at maximum size')

    def step(self):
        self.expand_tree()
        
        all_done = self.step_agent()
        if not all_done:
            all_done = self.step_user()


        return self.current_reward, all_done