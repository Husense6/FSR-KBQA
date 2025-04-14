import argparse
import json
import logging
import os
import time
from tqdm import tqdm

from src.fsr.answer_extract import extract_answer_str
from src.fsr.reason import llm_reason

import warnings


warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

# set cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def run_reason(args):
    questions = []
    kopls = []
    knowledge_snippets = []

    with open(args.input_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in tqdm(data):
        questions.append(item['question'])
        kb_knowledge = []
        if args.reason_flag == 'iag':
            knowledge_splits = item['knowledge_kb_iag'].split("\n")
        elif args.reason_flag == 'rag':
            knowledge_splits = item['knowledge_kb_rag'].split("\n")
        else:
            knowledge_splits = []
            
        for k in knowledge_splits:
            kb_knowledge.append(k)
        knowledge_snippets.append(kb_knowledge[:args.top_k])
        
        if 'llm_kopl' in item:
            kopls.append(item['llm_kopl'])
        else:
            raise ValueError("No kopl found.")
            
    COUNT = len(questions)

    output_json_file = open(os.path.join(args.output_dir, args.llm_name+"_"+args.reason_flag+".json"), 'w', encoding='utf-8')
    output_json = []
    
    for index, (question, kopl, knowledges) in tqdm(enumerate(zip(questions, kopls, knowledge_snippets))):
        template, result = llm_reason(question=question, kopl=kopl, llm_name=args.llm_name, reason_flag=args.reason_flag, knowledges=knowledges)
        answer = extract_answer_str(result)

        json_i = {
            'question': question,
            'prompt': template,
            'inference': result,
            'generate_answer': answer
        }
        output_json.append(json_i)
        
        if (index + 1) % 200 == 0:
            temp_json_file = open(os.path.join(args.output_dir, f"{args.llm_name}_{args.reason_flag}_temp.json"), 'w', encoding='utf-8')
            json.dump(output_json, temp_json_file, indent=4)
            temp_json_file.close()
            logging.info(f"{index+1} temporary items have been saved")

    json.dump(output_json, output_json_file, indent=4)
    output_json_file.close()
    time.sleep(2)
    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--log_dir', required=True, help='path to save logs')
    parser.add_argument('--llm_name', required = True, default='llama3.1:8b-instruct-q4_K_M')
    parser.add_argument('--reason_flag', required = True, default='iag')
    parser.add_argument('--top_k', required = True, type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.log_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    
    run_reason(args)


if __name__ == "__main__":
    main()
    
