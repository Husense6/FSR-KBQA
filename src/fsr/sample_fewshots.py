import json


def program_to_sequence(program):
    func_list = []
    inputs_list = []
    for item in program:
        func_list.append(item['function'])
        inputs_list.append(item['inputs'])

    sequence = []
    
    for func, inputs in zip(func_list, inputs_list):
        if inputs:
            args = ','.join(f'{arg}' for arg in inputs)
            func_call = f"{func}({args})"
        else:
            func_call = f"{func}()"
            
        sequence.append(func_call)
    return '->'.join(sequence)

# raw KQA Pro data
data = json.load(open('data/KQAPro/train.json'))

import random


random.shuffle(data)
data = data[:100]

text = ""

for i, item in enumerate(data):
    text += '''Example {i}\nQuestion: {question}\nKoPL: {kopl}\n\n'''.format(i=i+1, kopl=program_to_sequence(item['program']), **item)

with open('data/KQAPro/train_100.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('data/KQAPro/example_100_shots.txt', 'w') as f:
    f.write(text)
    