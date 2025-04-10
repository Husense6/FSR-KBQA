# FSR-KBQA

Fuzzy Symbolic Reasoning for few-shot KBQA: A CBR-inspired Generative Approach

---

**Recommended File Sturcture:**
```
FSR-KBQA/
├── data/
│   ├── GrailQA/
│   │   ├── GarilQA_all_llmkopl_knowledge.json
│   │   ├── GarilQA_all_question_with_label.json
│   │   └── GrailQA_aliase_data6763.json
│   └── KQAPro/
│       ├── example_100_shots.txt
│       ├── KQAPro_aliase_data106173.json
│       ├── KQAPro_all_question_with_label_val.json
│       └── kqapro_val_llmkopl_knowledge.json
├── logs/
│   ├── GrailQA/
│   └── KQAPro/
├── outputs/
│   ├── GrailQA/
│   │   └── *.json
│   └── KQAPro/ 
│       ├── 8b/
│       │   └── *.json
│       ├── 70b/
│       │   └── *.json
│       └── KQApro_chatgpt_answers.txt
└── src/
    ├── evaluate/
    │   ├── eval_GrailQA.py
    │   ├── eval_KQAPro.py
    │   └── eval_KQAPro_chatgpt.py
    ├── fsr/
    │   ├── __init__.py
    │   ├── answer_extract.py
    │   ├── LLM_KoPL_Gen.py
    │   ├── ollama_api.py
    │   ├── reason.py
    │   └── sample_fewshots.py
    ├── __init__.py
    ├── run_grailqa.py
    └── run_kqapro.py
```

## Preprocessing

### Random sampling

To obtain examples for few-shot learning, we randomly sampled 100 question-KoPL pairs from trainset of [KQA Pro](https://github.com/shijx12/KQAPro_Baselines).

```
cd FSR-KBQA/src/fsr/
python sample_fewshots.py
```

### Few-shot KoPL Generation

For generating KoPL sequence, we used Llama3.1-8B.

```
cd FSR-KBQA/src/fsr/
python LLM_KoPL_Gen.py
```

## Inference

To execute inference, run the following command:

- KQA Pro
```
cd FSR-KBQA/

python -m src.run_kqapro \
--input_dir data/KQAPro/kqapro_val_llmkopl_knowledge.json \ # path to preprocessed data
--output_dir outputs/KQAPro/70b \ # path for saving result
--log_dir logs/KQAPro \ # path for saving logs
--llm_name llama3.1:70b-instruct-q4_K_M \ # name for ollama model
--reason_flag iag \ # iag for our method, optional: rag for RAG/cot for COT/default for direct QA
--top_k 40 \ # num of retrieval KB knowledge
```

- GrailQA
```
cd FSR-KBQA/

python -m src.run_grailqa \
--input_dir data/GrailQA/GarilQA_all_llmkopl_knowledge.json \ # path to preprocessed data
--output_dir outputs/GrailQA \ # path for saving result
--log_dir logs/GrailQA \ # path for saving logs
--llm_name llama3.1:70b-instruct-q4_K_M \ # name for ollama model
--reason_flag iag \
--top_k 40 \ # num of retrieval KB knowledge
```

## Evaluation

As for evaluating the generative answers, we used [extend EM](https://github.com/tan92hl/Complex-Question-Answering-Evaluation-of-GPT-family). Just run command below to get the result:
**(You can modify the path in the code as required)**

- KQA Pro
```
cd FSR-KBQA/src/evaluate/
python eval_KQAPro.py
```

- KQA Pro(ChatGPT)
```
cd FSR-KBQA/src/evaluate/
python eval_KQAPro_chatgpt.py
```

- GrailQA
```
cd FSR-KBQA/src/evaluate/
python eval_GrailQA.py
```
