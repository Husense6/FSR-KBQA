import json
import re


def extract_largest_json(text):
    max_json = None
    max_length = 0
    
    left = 0
    while left < len(text):
        # Find possible start position of JSON
        while left < len(text) and text[left] != '{':
            left += 1
            
        if left >= len(text):
            break
            
        # Try different end position
        right = left
        stack = []
        
        while right < len(text):
            if text[right] == '{':
                stack.append('{')
            elif text[right] == '}':
                if stack:
                    stack.pop()
                    if not stack:  # Find complete JSON
                        try:
                            json_str = text[left:right + 1]
                            json.loads(json_str)  # Verify if it is a valid JSON
                            if len(json_str) > max_length:
                                max_json = json_str
                                max_length = len(json_str)
                        except json.JSONDecodeError:
                            pass
            right += 1
            
        left += 1
        
    return max_json

    
def extract_answer_str(response_text):
    response_text = response_text.strip()
    
    parts = re.split(r'(Answer|answer is|Therefore)\s*', response_text)

    if len(parts) > 1:
        result = parts[-1].strip("*> ):")
        return " ".join(result.split())

    sentences = re.split(r'[.!?](?!\d)\s*', response_text)
    answer_sentences = [s for s in sentences if 'answer' in s.lower()]
    if answer_sentences:
        return ' '.join(answer_sentences[-1].split()).strip() + '.'

    meaningful_sentences = [s.strip('*').strip().replace("\n", "").strip() for s in sentences if s.strip()]
    if meaningful_sentences:
        return meaningful_sentences[-1]

    return "No answer found."


# if __name__ == "__main__":
#     with open('outputs/KQAPro/val_all/llama3.1:8b-instruct-q4_K_M_iag.json', 'r') as f:
#         data = json.load(f)
#         for item in tqdm(data):
#             answer = extract_answer(item['inference'])
#             item['extracted_answer'] = answer
    
#     with open('outputs/KQAPro/val_all/llama3.1:8b-instruct-q4_K_M_iag_e.json', 'w') as f:
#         json.dump(data, f, indent=4)
        
