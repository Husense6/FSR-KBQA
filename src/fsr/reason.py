import re
from .ollama_api import generate_text
from .retrieve_knowledge import retrieve_kb_bm25, retrieve_llm


KOPL_FUNCTIONS = {
    "FindAll": "Return all entities",
    "Find": "Return all entities with the given name",
    "FilterConcept": "Filter entities belonging to the given concept, return entities",
    "FilterStr": "Filter entities with an attribute condition of string type, return entities and corresponding facts",
    "FilterNum": "Filter entities with an attribute condition of number type, return entities and corresponding facts",
    "FilterYear": "Filter entities with an attribute condition of year type, return entities and corresponding facts",
    "FilterDate": "Filter entities with an attribute condition of date type, return entities and corresponding facts",
    "QFilterStr": "Filter entities and corresponding facts with a qualifier condition of string type, return entities and corresponding facts",
    "QFilterNum": "Filter entities and corresponding facts with a qualifier condition of string type, return entities and corresponding facts",
    "QFilterYear": "Filter entities and corresponding facts with a qualifier condition of year type, return entities and corresponding facts",
    "QFilterDate": "Filter entities and corresponding facts with a qualifier condition of date type, return entities and corresponding facts",
    "Relate": "Find entities that have a specific relation with the given entity, return entities and corresponding facts",
    "And": "Return the intersection of two entity sets",
    "Or": " Return the union of two entity sets",
    "QueryName": "Return the entity name",
    "Count": "Return the number of entities",
    "QueryAttr": "Return the attribute value of the entity",
    "QueryAttrUnderCondition": "Return the attribute value whose corresponding fact should satisfy the qualifier condition",
    "QueryRelation": "Return the relation between two entities",
    "SelectBetween": "From the two entities, find the one whose attribute value is greater or less, return its name",
    "SelectAmong": "From the entity set, find the one whose attribute value is the largest or smallest, return its name",
    "VerifyStr": "Verify whether the two strings are equal, return 'yes' or 'no'",
    "VerifyNum": "Verify whether the two numbers satisfy the condition, return 'yes' or 'no'",
    "VerifyYear": "Verify whether the two years satisfy the condition, return 'yes' or 'no'",
    "VerifyDate": "Verify whether the two dates satisfy the condition, return 'yes' or 'no'",
    "QueryAttrQualifier": "Return the qualifier value of the given fact (Entity, Key, Value)",
    "QueryRelationQualifier": "Return the qualifier value of the given fact (Entity, Pred, Entity)"
}

naive_template = """Human:
<Question>:
{question}

Guidelines for response:
- You are an expert assistant specializing in knowledge question answering.
- Please answer the Question, keep the response concise and to the point.

## Response Structure:
### Answer:
[Provide the concise answer to input question within 50 words]

Assistant:
"""

cot_template = """Human:
<Question>:
{question}
<KoPL Reasoning Steps>:
{kopl}
<KoPL Function Description>:
{kopl_description}

Guidelines for response:
- You are an expert assistant specializing in knowledge question answering, Please answer the <Question> based on your own internal knowledge.
- Based on the <KoPL Reasoning Steps>, stepwise reasoning to answer the question.
- Provide a step answer for each step and record the answer based on your internal knowledge.
- If the reasoning is insufficient, please use the LLM to generate the answer directly.
- Keep responses concise and to the point.

## Response Structure:
### Thought Process:
Step 1: [KoPL steps.]
- step answer: [Provide the intermediate concise result within 20 words.]

### Answer: [based on previous stepwise reasoning, Provide the final concise answer to input question within 50 words]

Assistant:
"""

rag_template = """Human:
<Question>:
{question}
<KB Knowledge>:
{knowledge_kb}

Guidelines for response:
- You are an expert assistant specializing in knowledge question answering. Please answer the <Question> based on the given <KB Knowledge>.
- Reasoning step-by-step base the provided knowledge to answer the <Question>. 
- Keep the response concise and to the point.

## Response Structure:
### Reasoning process:
Step 1: [Provide Sub-question.]
step answer: [Provide the intermediate result.]

### Answer:
[based on previous stepwise reasoning, Provide the final concise answer to input question within 50 words]

Assistant:
"""

iag_template = """Human:
<Question>:
{question}
<KoPL Reasoning Steps>:
{kopl}
<KoPL Function Description>:
{kopl_description}
<KB Knowledge>:
{kb_knowledge}

## Guidelines for response:
- You are an expert assistant specializing in knowledge question answering. Please answer the <Question> based on the <KB Knowledge>.
- Based on the question and KoPL, decompose the question into several sub-questions. Each sub-question should be corresponding to several independent KoPL steps.
- Provide an answer for each sub-question and record the answer.
- If the provided knowledge does not have sufficient information, use internal knowledge for reasoning.
- Keep responses concise and to the point.
- Provide the most accurate date possible in the format: 'YYYY-MM-DD'.
- For general yes/no questions, respond with a direct 'yes' or 'no'.

## Response Structure:
### Thought Process:
Step 1: [Provide Sub-question.]
- Corresponding KoPLs: [corresponding KoPLs.]
- step answer: [Provide the intermediate result.]

### Answer:
   [Provide the final concise answer to input question within 50 words]

Assistant:
"""

'''
- The provided KoPL Answer is the result of precisely executing KoPL on Knowledge base but not the standard answer.
- Combine the answers to the sub-questions and provided <KoPL Answer> to answer the question finally.'''


def add_kopl_description(kopl:str):
    kopl_descriptions = []
    for item in KOPL_FUNCTIONS.keys():
        if item in kopl:
            kopl_descriptions.append(item+ ": "+KOPL_FUNCTIONS[item])
    return "\n".join(kopl_descriptions)


def llm_reason(question: str, kopl: str, llm_name: str, reason_flag: str, knowledges: list) -> str:
    kopl_str = kopl
    knowledge_kb = "\n".join(knowledges)
    
    kopl_descriptions = add_kopl_description(kopl)

    if reason_flag == 'rag':
        template = rag_template.format(
        question=question,
        knowledge_kb=knowledge_kb
    )
    elif reason_flag == 'iag':
        template = iag_template.format(
            question=question,
            kopl=kopl,
            kopl_description=kopl_descriptions,
            kb_knowledge=knowledge_kb
        )
    elif reason_flag == 'cot':
        template = cot_template.format(
            question=question,
            kopl=kopl,
            kopl_description=kopl_descriptions
        )
        
    else:
        template = naive_template.format(question=question)

    template = '\n'.join([line for line in template.split('\n') if line.strip()])
    result = generate_text(template, model=llm_name)

    return template, result


def llm_reason_grailqa(question: str, kopl: str, llm_name: str, reason_flag: str, knowledges: list) -> str:
    kopl_str = kopl
    knowledge_kb = "\n".join(knowledges)

    kopl_descriptions = add_kopl_description(kopl)

    if reason_flag == 'rag':
        template = rag_template.format(
        question=question,
        knowledge_kb=knowledge_kb
    )
    elif reason_flag == 'iag':
        template = iag_template.format(
            question=question,
            kopl=kopl_str,
            kopl_description=kopl_descriptions,
            kb_knowledge=knowledge_kb
        )
    elif reason_flag == 'cot':
        template = cot_template.format(
            question=question,
            kopl=kopl_str,
            kopl_description=kopl_descriptions
        )
        
    else:
        template = naive_template.format(question=question)

    template = '\n'.join([line for line in template.split('\n') if line.strip()])
    result = generate_text(template, model=llm_name)

    return template, result
