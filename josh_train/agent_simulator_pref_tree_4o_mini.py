import yaml
import re
import copy
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from josh_train.utils import make_transcript, request_openai, parse_api_call, handle_api_calls, compute_cost
import os
class Node:
    def __init__(self, messages, called_apis, messages_for_user, parent):
         self.messages = messages
         self.messages_for_user = messages_for_user
         self.called_apis = called_apis
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
        tree = [(self.messages, self.is_successful, self.is_golden_path)]

        if self.left:
            tree.extend(self.left.get_tree())
        if self.right:
            tree.extend(self.right.get_tree())

        return tree
     

class AgentSimulator:
    def __init__(self, api_examples, api_defs, modelname, client, conversation_state):
        self.cwd = os.getcwd()
        with open(f'{self.cwd}/prompts/prompts.yaml', 'r') as file:
            prompts = yaml.safe_load(file)
        self.api_defs = api_defs
        self.api_examples = api_examples
        self.apis_to_examples = {x['name']: x for x in api_examples}
        with open(f'{self.cwd}/data/tools.json', 'r') as file:
            tools_list = json.load(file)
        self.MONO_PROMPT = prompts['mistral_mono_prompt_v2'].replace('{example_filled}', json.dumps(tools_list, indent=2))
        self.pattern = "(PLAN|APICALL|SPEAK)(.*?)(?=PLAN|APICALL|SPEAK|$)"
        self.modelname = modelname
        self.client = client
        self.before_message = '\nOutput:\nPLAN '

        # Tree changes
        self.conversation_state = conversation_state
        self.messages_full = []
        self.beam_size = 8
        self.max_turn_tries = 3
        self.root = Node(messages=[], called_apis=[], messages_for_user=[], parent=None)
        self.step_num=0
        self.cost = 0.0

    def parse_agent_message(self, output):
        commands  = re.findall(self.pattern , output , re.DOTALL)
        return commands
    
    
    def request(self, conversations):
        outputs = []
        for messages in conversations:
            output = request_openai(messages, self.modelname, self.client, max_tokens=256, temperature=1.0)
            outputs.append(output)
            self.cost += compute_cost(messages+ [{'role': 'user', 'content': output}], model=self.modelname)
        return outputs
        
    def make_messages(self, messages):
        result =[{'role':'system', 'content':self.MONO_PROMPT}]
        for idx, message in enumerate(messages):
            result.append(message)
        return result
    
    def handle_api(self, command, conversation_state):
        try:
            api_values = parse_api_call(command)
        except:
            return 'FAILURE INCORRECTLY FORMATTED APICALL'
        if api_values['api_name'] not in self.apis_to_examples:
            return 'FAILURE INCORRECTLY FORMATTED APICALL'
        returns = handle_api_calls(api_values['api_name'], api_values['api_args'], conversation_state=conversation_state)
        return returns
    

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
    
    def step(self, user_replies):
        leaves = np.array(self.root.get_leaves())
        make_more_leaves = len(leaves)*2<=self.beam_size
        # Add messages to each leaf
        for idx, leaf in enumerate(leaves):
            # If the user ended the conversation, kill the leaf and keep going
            if 'END_CONVERSATION' in user_replies[idx][-1]['content']:
                leaf.messages.extend(user_replies[idx])
                leaf.messages_for_user.extend(user_replies[idx])
                leaf.conversation_over = True
                continue
            # Extend leaves
            if make_more_leaves:
                leaf.left = Node(copy.deepcopy(leaf.messages+user_replies[idx]), copy.deepcopy(leaf.called_apis), copy.deepcopy(leaf.messages_for_user+user_replies[idx]), parent=leaf)
                leaf.right = Node(copy.deepcopy(leaf.messages+user_replies[idx]), copy.deepcopy(leaf.called_apis), copy.deepcopy(leaf.messages_for_user+user_replies[idx]), parent=leaf)
            else:
                leaf.messages.extend(user_replies[idx])
                leaf.messages_for_user.extend(user_replies[idx])

        # Make new extended list of leaves
        leaves = np.array(self.root.get_leaves())
        if len(leaves)==0:
            return None, None, None
        # Step for each leaf
        count = 0
        done = np.array([False]*len(leaves))
        training_examples = []
        collapse_root_to = None
        successful_leaves = []
        apis_to_delete = []
        percent_100_reached = False
        while count < self.max_turn_tries:
            unfinished_leaf_indices = np.where(done==False)[0]
            if len(unfinished_leaf_indices)==0:
                break
            unfinished_leaves = leaves[unfinished_leaf_indices]
            # turns = self.request([[{'role':'system', 'content':self.MONO_PROMPT}]+leaf.messages for leaf in unfinished_leaves])
            if any([len(leaf.messages) > 25 for leaf in unfinished_leaves]):
                batch_size=2
            else:
                batch_size = 4
            turns = []
            for i in range(0, len(unfinished_leaves), batch_size):
                batch = unfinished_leaves[i:i+batch_size]
                turns.extend(self.request([[{'role':'system', 'content':self.MONO_PROMPT}]+leaf.messages for leaf in batch]))
            for idx, turn in enumerate(turns):
                reply, successful_api_call, api_calls_to_delete = self._step(turn, unfinished_leaves[idx])
                apis_to_delete.extend(api_calls_to_delete)
                if reply:
                    # if there is a speak reply this leaf is done and we add the reply to it's formal message list
                    done[unfinished_leaf_indices[idx]] = True
                    unfinished_leaves[idx].messages_for_user.extend(reply)
                else:
                    # This means there was no speak action
                    if successful_api_call:
                        # This means we've hit one of the goal apis!
                        successful_leaves.append(unfinished_leaves[idx])
                        # Record the first node to hit a goal so we can collapse the root to it
                        if not collapse_root_to:
                            collapse_root_to = unfinished_leaves[idx]
                    if count+1 == self.max_turn_tries:
                        unfinished_leaves[idx].messages_for_user.append({'role':'system', 'content':'Error: Agent ran out of retries.'})
                        unfinished_leaves[idx].messages.append({'role':'system', 'content':'Error: Agent ran out of retries.'})
                    # Otherwise, we didn't hit a goal api, so we should just continue on
            count += 1
        self.step_num+=1

        # if we had a goal found, collapse the root to the node that found it
        if collapse_root_to:
            #Delete the api calls that were found
            for del_api in set(apis_to_delete):
                del self.conversation_state.apis_for_eval[del_api[0]]['success'][del_api[1]]
            # set the descendence of all successful leaves as successful
            for leaf in successful_leaves:
                self.set_success_path(leaf)

            self.set_golden_path(collapse_root_to)

            training_examples = self.root.get_tree()
            self.root = collapse_root_to
            self.root.parent=None
            self.root.is_successful = False
            leaves = [self.root]
        percent_100_reached = all([len(self.conversation_state.apis_for_eval[k]['success'])==0 for k in self.conversation_state.apis_for_eval.keys()])
        return training_examples, [leaf.messages_for_user for leaf in leaves], percent_100_reached

    
    def _step(self, turn, node):
        node.messages.append({'role':'assistant', 'content':turn})
        parsed = self.parse_agent_message(turn.replace('<COMMAND_END>', '').strip().replace('\n','').replace('\\',''))
        successful_api_call = False
        apis_to_delete_list = []
        for command_type, command in parsed:
            command_type = command_type.strip()
            command=command.strip()
            if command_type == 'SPEAK':
                return [{'role':'assistant', 'content':command}], False, apis_to_delete_list
            elif command_type == 'APICALL':
                command = command.strip().replace('\n','')
                output = self.handle_api(command, self.conversation_state)
                # add the called apis to the node
                node.called_apis.extend(self.conversation_state.called_apis)

                # Test for success if we havent already
                if not successful_api_call:
                    correct_calls, _, _, apis_to_delete = self.conversation_state.evaluate_apis(self.conversation_state.called_apis)
                    apis_to_delete_list.extend(apis_to_delete)
                    successful_api_call = correct_calls > 0

                # delete the called apis for this turn
                self.conversation_state.called_apis = []
                # Add the return
                node.messages.append({'role':'user', 'content':'APIRETURN '+json.dumps(output)})
        return None, successful_api_call, apis_to_delete_list
    