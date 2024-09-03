


import copy

import numpy as np


class Node:
    def __init__(self, agent, parent):
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
        tree = [(self.agent.messages, self.is_successful, self.is_golden_path)]

        if self.left:
            tree.extend(self.left.get_tree())
        if self.right:
            tree.extend(self.right.get_tree())

        return tree
    


class JOSH():
    def __init__(self, rewards, agent_step, user_step, add_error_message, root_agent=None, beam_size=8, max_turn_tries=10):
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
    
    def is_reward(self, agent):
        return agent.recent_action in self.rewards
            
    def step_user(self):
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return True
        for leaf in leaves:
            leaf.agent = self.user_step(leaf.agent)
            if leaf.agent.done:
                leaf.conversation_over = True

        leaves = np.array(self.root.get_leaves())
        return len(leaves)==0
    
    def step_agent(self):
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return True
        # Step for each leaf
        count = 0
        done = np.array([False]*len(leaves))
        training_examples = []
        collapse_root_to = None
        successful_leaves = []
        api_to_delete = None
        while count < self.max_turn_tries:
            unfinished_leaf_indices = np.where(done==False)[0]
            if len(unfinished_leaf_indices)==0:
                break
            unfinished_leaves = leaves[unfinished_leaf_indices]

            batch_size=2
            turn_finished = []
            for i in range(0, len(unfinished_leaves), batch_size):
                batch = unfinished_leaves[i:i+batch_size]
                for j, lf in enumerate(batch):
                    lf.agent, pass_to_customer = self.agent_step(lf.agent)
                    if pass_to_customer is None:
                        turn_finished.append(True)
                        lf.conversation_over = True
                    else:
                        turn_finished.append(pass_to_customer)


            for idx, turn in enumerate(unfinished_leaves):
                if turn.agent.done or turn_finished[idx]:
                    done[unfinished_leaf_indices[idx]] = True
                else:
                    if self.is_reward(turn.agent):
                        successful_leaves.append(turn)
                        if not collapse_root_to:
                            collapse_root_to = unfinished_leaves[idx]
                            api_to_delete = copy.deepcopy(collapse_root_to.agent.recent_action)
                            if len(self.rewards)==1:
                                self.golden_agent = turn.agent
                    # else:
                    #     # if it's made an api call that was not part of the desired actions, kill it
                    #     done[unfinished_leaf_indices[idx]] = True
                    #     turn.conversation_over=True
                        
                    
                if count+1 == self.max_turn_tries:
                    turn.agent = self.add_error_message(turn.agent)

            count += 1
        
        if collapse_root_to:
            # set the descendence of all successful leaves as successful
            for leaf in successful_leaves:
                self.set_success_path(leaf)

            self.rewards.remove(api_to_delete)
            self.set_golden_path(collapse_root_to)
            training_examples = self.root.get_tree()
            self.training_examples.extend(training_examples)
            self.root = collapse_root_to
            self.root.parent=None
            self.root.is_successful = False
            leaves = [self.root]
        
        self.current_reward = (self.num_total_rewards-len(self.rewards))/self.num_total_rewards
        all_done = len(self.rewards) == 0
        return all_done


    def expand_tree(self):
        leaves = np.array(self.root.get_leaves())
        make_more_leaves = len(leaves)*2<=self.beam_size
        # Add messages to each leaf
        if make_more_leaves:
            for leaf in leaves:
                # If the user ended the conversation, kill the leaf and keep going
                if leaf.conversation_over:
                    continue
                # Extend leaves
                leaf.left = Node(copy.deepcopy(leaf.agent), parent=leaf)
                leaf.right = Node(copy.deepcopy(leaf.agent), parent=leaf)

    def step(self):
        self.expand_tree()

        all_done = self.step_agent()
        if not all_done:
            all_done = self.step_user()


        return self.current_reward, all_done