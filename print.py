import pickle

with open("./experiment_cache/batched_1.5B_discrete_100tok_end_8traj_result.pkl", "rb") as f:
    obj = pickle.load(f)

print(obj.keys())

print(obj["avg_discrete_reward_history"][-1])