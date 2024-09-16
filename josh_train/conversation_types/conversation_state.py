import yaml
from josh_train.utils import *

class Conversation:
    def __init__(self, key, apis, delex):
        self.dbs = create_dbs()
        self.apis = apis[key]
        self.convo_key = key

        self.goals = delex[key]['goal']
        self.ground_truth_conversaiton = [{'role': 'assistant' if utterance['metadata'] else 'user', 'content':utterance['text']} for utterance in delex[key]['log']]

        self.called_apis = []

        # Filter the DBs because sometimes failure cases are present
        for domain in self.apis.keys():
            if f'search_{domain}' in self.apis[domain]['failure']:
                filter_dbs(domain, self.apis[domain]['failure'][f'search_{domain}']['parameters'], self.dbs)

    def close_convos(self):
        keys = [k for k in self.dbs.keys()]
        for k in keys:
            del self.dbs[k]

    def add_api_call(self, api_name, api_args, returned):
        if 'search' == api_name.split('_')[0] and type(returned)==list:
            returned_value = returned[0] if len(returned)!=0 else None
        else:
            returned_value = returned
        self.called_apis.append({'name':api_name, 'parameters': api_args, 'returned': returned_value})
    
    def _test_book_apis(self, goal_api, domain):
        # Loop through all the called apis
        for called_api in self.called_apis:
            if called_api['name'] == goal_api:
                successful_parameters = {k:v for k,v in self.apis[domain]['success'][goal_api]['parameters'].items() if not test_if_val_is_empty(v)}
                subset_test = is_subset(successful_parameters, called_api['parameters'])
                if subset_test:
                    return 1
        return 0
    
    def _test_search_apis(self, goal_api, domain):
        correctly_searched = 0
        unique_id_type = 'trainID' if domain == 'train' else 'name'
        # Pull the correct item
        unique_id_of_correct = self.apis[domain]['success'][f'book_{domain}']['unique_id'] if f'book_{domain}' in self.apis[domain]['success'] else ''
        for called_api in self.called_apis:
            if not called_api['returned'] or type(called_api['returned'])==str:
                continue
            if called_api['name'] == goal_api:
                goal_parameters = {k:v for k,v in self.apis[domain]['success'][goal_api]['parameters'].items() if not test_if_val_is_empty(v)}
                # if the Goal (correct) parameters are a subset of what was called
                if is_subset(goal_parameters, called_api['parameters']):
                    # if theres no "technically" correct answer, its correct
                    if len(unique_id_of_correct)==0:
                        correctly_searched = 1
                    # if theres a "correct" answer, make sure that was returned
                    else:
                        # if the unique id is correct its correct
                        if unique_id_of_correct == called_api['returned'][unique_id_type]:
                            correctly_searched = 1
                        # else it's wrong.
                # if the called api is a subset of the Goal (correct) parameters
                elif is_subset(called_api['parameters'], goal_parameters):
                    # if theres no "technically" correct answer
                    if len(unique_id_of_correct)==0:
                        # if the goals are a subset of the returned object, its "right"
                        if is_subset(goal_parameters, called_api['returned']):
                            correctly_searched = 1
                        # else it is not right
                    # if there is a "correct" answer
                    else:
                        # if the unique id is correct its correct
                        if unique_id_of_correct == called_api['returned'][unique_id_type]:
                            correctly_searched = 1
                        # else it's wrong.
                # neither, so it is incorrect
        return correctly_searched


    def evaluate_apis(self):
        number_of_successful_apis = 0
        correct_calls = 0
        failed_api_calls = []
        for domain in self.apis.keys():
            for goal_api in self.apis[domain]['success'].keys():
                #Looping through each goal api
                action, _ = goal_api.split('_')
                if action == 'book':
                    successful_call = self._test_book_apis(goal_api, domain)
                elif action == 'search':
                    successful_call = self._test_search_apis(goal_api, domain)
                else:
                    successful_call = 0

                if successful_call==0:
                    failed_api_calls.append({goal_api:self.apis[domain]['success'][goal_api]['parameters']})

                correct_calls += successful_call
                number_of_successful_apis+=1

        return correct_calls/number_of_successful_apis, failed_api_calls