import copy
import numpy as np
from typing import Optional, Callable, Awaitable, Any, Tuple, List

class AsyncJOSH:
    def __init__(self, rewards, agent_step: Callable, user_step: Callable, add_error_message: Callable, 
                 root_agent, user, beam_size=8, max_turn_tries=10, agent_model=None, 
                 agent_tokenizer=None, agent_env=None, debug=False):
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
    
    async def step_user(self):
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return True
        if self.debug:
            print(f'Running {len(leaves)} users')
        
        # Use asyncio.gather to run all user steps concurrently
        tasks = []
        for leaf in leaves:
            tasks.append(self._process_user_leaf(leaf))
        
        await asyncio.gather(*tasks)
        
        leaves = np.array(self.root.get_leaves())
        return len(leaves)==0
    
    async def _process_user_leaf(self, leaf):
        # Call the user_step function asynchronously
        leaf.agent, end_conversation = await self.user_step(self.user, leaf.agent)
        leaf.conversation_over = end_conversation
    
    async def step_agent(self):
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

            # Process all unfinished leaves concurrently
            tasks = []
            for lf in unfinished_leaves:
                tasks.append(self._process_agent_leaf(lf))
            
            turn_finished_results = await asyncio.gather(*tasks)
            
            # Update done status based on results
            for idx, (turn, turn_finished, got_reward, rw_to_delete) in enumerate(turn_finished_results):
                leaf_idx = unfinished_leaf_indices[idx]
                
                if turn_finished:
                    done[leaf_idx] = True
                
                if got_reward:
                    successful_leaves.append(turn)
                    if not collapse_root_to:
                        if self.debug:
                            print(f'🌟 Got reward')
                        collapse_root_to = unfinished_leaves[idx]
                        rewards_to_delete = copy.deepcopy(rw_to_delete)
                        if len(self.rewards)==1:
                            self.golden_agent = turn.agent
                
                if count+1 == self.max_turn_tries and not turn_finished:
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
    
    async def _process_agent_leaf(self, leaf):
        # Call the agent_step function asynchronously
        try:
            leaf.agent, pass_to_customer = await self.agent_step(
                agent=leaf.agent, 
                model=self.agent_model, 
                tokenizer=self.agent_tokenizer, 
                env=self.agent_env
            )
        except Exception as e:
            if self.debug:
                print(f"Error in agent step: {e}")
            pass_to_customer = None
        
        turn_finished = True if pass_to_customer is None else pass_to_customer
        
        # Check for rewards
        got_reward, rw_to_delete = await self._check_rewards(leaf.agent)
        
        return leaf, turn_finished, got_reward, rw_to_delete
    
    async def _check_rewards(self, agent):
        # Handle both synchronous and asynchronous reward checking
        if hasattr(self.rewards, 'is_reward_async'):
            return await self.rewards.is_reward_async(agent.recent_actions)
        else:
            return self.rewards.is_reward(agent.recent_actions)

    def expand_tree(self):
        leaves = np.array(self.root.get_leaves())
        make_more_leaves = len(leaves)*2 <= self.beam_size
        
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

    async def step(self):
        self.expand_tree()
        
        all_done = await self.step_agent()
        if not all_done:
            all_done = await self.step_user()

        return self.current_reward, all_done

# Node class from original JOSH implementation
class Node:
    def __init__(self, agent, parent: Optional["Node"]=None):
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

# Helper function from original JOSH implementation
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