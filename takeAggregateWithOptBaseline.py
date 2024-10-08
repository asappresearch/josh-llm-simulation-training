import os
import json
import numpy as np
import sys
from deepsig import bootstrap_test, aso

def get_aggregates(input_dir):
    data = []


    # for file in os.listdir('./'+input_dir+"/"):
    with open('./'+input_dir, 'r') as f:
        data = json.load(f)

    num_of_100 = 0
    scores = []
    for d in data:
        scores.append(d['reward'])
        if d['reward'] == 1.0:
            num_of_100+=1

    return data, scores, num_of_100


if __name__ == "__main__":

    input_dir = sys.argv[1]
    data, scores, num_of_100 = get_aggregates(input_dir)
    variance = np.var(scores)
    print_model_name="Model"
    print("\t".join(["Approach","Full Success","Mean Reward","Test Samples","Reward Variance","Is Model Stat Sig Better? (Lower p below 0.01 --> Model is indeed betterer)"]))
    print(print_model_name,"\t",num_of_100/len(data),"\t",np.mean(scores),"\t",len(data),"\t",variance,"\t","N/A")

    if len(sys.argv)>=3:
        for baseline_id in range(len(sys.argv)-2):
            baseline_dir = sys.argv[2+baseline_id]
            data_baseline, scores_baseline, num_of_100_baseline = get_aggregates(baseline_dir)
            variance_baseline = np.var(scores_baseline)
            scores_model = scores

            #print("Model scores contain NaN:", np.isnan(scores).any())
            #print("Model scores contain inf:", np.isinf(scores).any())
            #print("Baseline scores contain NaN:", np.isnan(scores_baseline).any())
            #print("Baseline scores contain inf:", np.isinf(scores_baseline).any())

            p_value = bootstrap_test(np.array(scores_model),np.array(scores_baseline),seed=42,num_jobs=100,num_samples=100000)
            print(p_value)
            print_model_name="Baseline"+" "+str(baseline_id)
            print(print_model_name,"\t",num_of_100_baseline/len(data_baseline),"\t",np.mean(scores_baseline),"\t",len(data_baseline),"\t",variance_baseline,"\t",p_value)

