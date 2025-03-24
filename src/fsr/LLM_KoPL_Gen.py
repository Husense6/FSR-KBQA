import json
from tqdm import tqdm
import re



kopl_few_shot_template = """Instruction:
You are an expert in converting natural language questions into KOPL (Knowledge-Oriented Programming Language) queries. KOPL is a structured query language used for step-by-step reasoning over knowledge graphs. Translate the given question into a series of KOPL operations.

Here are the operations of KoPL and their descriptions:
{kopl_description}

Follow the format and examples below.
{example}

Task:
Translate the following natural language question into a KOPL query. Use the same format as the examples.
Question: "{question}"
"""

example = open('data/KQAPro/example_100_shots.txt', 'r').read()

def llm_gen_kopl(question, model:str='llama3.1:8b-instruct-q4_K_M'):
    template = kopl_few_shot_template.format(example=example, question=question, kopl_description="\n".join([i +": " + KOPL_FUNCTIONS[i] for i in KOPL_FUNCTIONS.keys()]))
    return generate_text(template, model)

data = json.load(open('data/CWQ/CWQ_dev_with_label_koplgen_knowledge.json', 'r'))

i = 1
for d in tqdm(data):
    question = d['question']
    while True:
        result = llm_gen_kopl(question)
        #match "KoPL: ", extract the following content
        if result is not None and len(result) > 0:
            match = re.search("KoPL: (.*)", result)
            if match:
                result = match.group(1).split('\n')[0].strip()
                break
    d['llm_kopl'] = result
    i += 1
    if i % 500 == 0:
        with open('data/CWQ/CWQ_dev_with_label_koplgen_knowledge_tmp.json', 'w') as fp:
            json.dump(data, fp, indent=4)


with open('data/CWQ/CWQ_dev_with_label_koplgen_knowledge.json', 'w') as fp:
    json.dump(data, fp, indent=4)
    