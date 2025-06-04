"""Implements a GUI which allows fast filtering of invalid entries in
a dataset.

Reads through the parquet, selects the top k entries sorted in order
of greatest reward, and displays them on the user interface. Includes
buttons to remove or keep this entry.

Default implementation not distinguish between different splits, and o-
nly considers the reward of the Skywork PRM.

If an entry is selected to be kept, it's tagged
"""




import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
import argparse
import math




parser = argparse.ArgumentParser()

parser.add_argument(
    '--parquet_path',
    type=str,
    required=True
)

args = parser.parse_args()

def sorting_func(entry):
    if entry["final_answer_correct"] == False:
        return -1
    if len(entry["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"]) == 0:
        return -1
    return entry["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"][-1] - entry["Skywork-o1-Open-PRM-Qwen-2.5-7B"][-1]

# todo: add comment section
# todo: add dataset index
# todo: add the model which was used to generate the trajectory

def preprocess():
    df = pd.read_parquet(args.parquet_path)

    df['_temp_sort_key'] = df.apply(sorting_func, axis=1)
    df_sorted = df.sort_values(by='_temp_sort_key', ascending=False)
    top = df_sorted.head(100)
    assert top.iloc[-1]["final_answer_correct"] == True, "incorrect answers included"

    df.drop(columns=["_temp_sort_key"], inplace=True)

    if "accepted" not in df.columns:
        df["accepted"] = -1
    if "comment" not in df.columns:
        df["comment"] = ""

    t = top.reset_index()
    return t, df

df, bigdf = preprocess()


def threedec(num):
    mult = 10 ** 3
    return math.floor(num * mult) / mult

def color(rewards): # red: (234, 153, 153) green: (182, 215, 168)
    red = (234, 153, 153)
    green = (147, 196, 125)
    return [str(tuple(v1 * (1 - r) + v2 * r for v1, v2 in zip(red, green))) for r in rewards]

def collect_context(data):
    data["Skywork-o1-Open-PRM-Qwen-2.5-7B"] = [threedec(n) for n in data["Skywork-o1-Open-PRM-Qwen-2.5-7B"]]
    context = {"problem1": data["problem"], "steps1": data["steps"], "rewards1": data["Skywork-o1-Open-PRM-Qwen-2.5-7B"], "color1": color(data["Skywork-o1-Open-PRM-Qwen-2.5-7B"]), "len1": len(data["steps"]), 
               "problem2": "", "steps2": [], "rewards2": [], "color2": [], "len2": 0,
               "accepted": bigdf.iloc[data['index']]['accepted'], "comment": bigdf.iloc[data['index']]['comment'],
               "id": data["id"],
               "generator": data["generator"]}

    if "aug_problem" in data:
        context["problem2"] = data["aug_problem"]
    if "aug_steps" in data:
        context["steps2"] = data["aug_steps"]
        data["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"] = [threedec(n) for n in data["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"]]
        context["rewards2"] = data["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"]
        context["color2"] = color(data["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"])
        context["len2"] = len(data["aug_steps"])
    if "shuffled_problem" in data:
        context["problem2"] = data["shuffled_problem"]
    else:
        context["problem2"] = data["problem"]

    context["max_len"] = max(context["len1"], context["len2"])
    return context

def record_comment(i, comment):
    bigdf.loc[df.iloc[i]["index"], "comment"] = (comment if comment is not None else "")

if __name__ == "__main__":
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('index.html', parquet_path=args.parquet_path)
    
    i = 0

    @app.route('/start')
    def start():
        global i
        method = request.args.get('method')
        i = 0
        if method == "continue":
            for i in range(len(df)):
                if bigdf.iloc[df.iloc[i]['index']]['accepted'] == -1:
                    break
            if i == len(df) - 1:
                i += 1
        return redirect(url_for('reviewer'))

    @app.route('/reviewer')
    def reviewer():
        global i
        if i < len(df):
            context = collect_context(df.iloc[i].to_dict())
        else:
            return redirect(url_for('done'))
        
        render = request.args.get('render')
        if render == "plaintext":
            context['mathjax'] = ""
        else:
            context['mathjax'] = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' 
        return render_template('reviewer.html', **context)
    
    @app.route('/accept')
    def accept():
        global i
        comment = request.args.get('comment')
        record_comment(i, comment)
        bigdf.loc[df.iloc[i]['index'], 'accepted'] = 1
        i += 1
        return redirect(url_for('reviewer'))

    @app.route("/reject")
    def reject():
        global i
        comment = request.args.get('comment')
        record_comment(i, comment)
        bigdf.loc[df.iloc[i]['index'], 'accepted'] = 0
        i += 1
        return redirect(url_for('reviewer'))
    
    @app.route("/prev")
    def prev():
        global i
        comment = request.args.get('comment')
        record_comment(i, comment)
        if i > 0:
            i -= 1
            return redirect(url_for('reviewer'))
        else:
            return redirect(url_for('home'))
    
    @app.route("/next")
    def next():
        global i
        comment = request.args.get('comment')
        record_comment(i, comment)
        if i < len(df):
            i += 1
            return redirect(url_for('reviewer'))
        else:
            return redirect(url_for('home'))
    
    @app.route("/save")
    def save():
        global i
        comment = request.args.get('comment')
        record_comment(i, comment)
        bigdf.to_parquet(args.parquet_path)
        return redirect(url_for('reviewer'))

    @app.route('/done')
    def done():
        bigdf.to_parquet(args.parquet_path)
        return render_template('done.html')

    app.run(debug=True)
