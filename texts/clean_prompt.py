import json
import os
from multiprocessing import Pool
from tqdm import tqdm

# Organize prompts
def process_image_file(args):
    matched_data = []
    all_cleaned_data, answer_data, question_data = args
    if question_data['image'] in all_cleaned_data:
        matched_data.append({
            question_data['image']: answer_data['text'],
        })
        all_cleaned_data.remove(question_data['image'])
    return matched_data

def process_file(all_cleaned_data, answers_data, questions_data):
    with Pool(processes=64) as pool:  # the number of parallel processes is set to 64
        args = [(all_cleaned_data, 
                 answer_data, question_data) for answer_data, question_data in zip(answers_data, questions_data)]
        results = list(tqdm(pool.imap(process_image_file, args), total=len(args), desc="Processing images"))
        all_matched_data = [item for sublist in results for item in sublist]

    return all_matched_data

def main():
    # Read all image files recorded in 'cleaned/'
    cleaned_path = 'cleaned/'
    cleaned_files = os.listdir(cleaned_path)
    all_cleaned_data = []
    for cleaned_file in cleaned_files:
        category_name = os.path.splitext(cleaned_file)[0]
        with open(os.path.join(cleaned_path, cleaned_file), 'r') as f:
            for line in f:
                all_cleaned_data.append(os.path.join('train', category_name, line.strip()))
    print(f"==> Found {len(all_cleaned_data)} cleaned images..")

    # Read answers.jsonl file
    answers_file = 'answers.jsonl'
    with open(answers_file, 'r') as f:
        answers_data = [json.loads(line) for line in f]
    print(f"==> Found {len(answers_data)} answers..")

    # Read imagenet_context_question.jsonl file
    questions_file = 'imagenet_context_question.jsonl'
    with open(questions_file, 'r') as f:
        questions_data = [json.loads(line) for line in f]
    print(f"==> Found {len(questions_data)} questions..")

    all_matched_data = process_file(all_cleaned_data, answers_data, questions_data)
    print(f"==> Found {len(all_matched_data)} matched data..")

    # Process all_matched_data, e.g., save to file or further process
    output_file = f'matched_info.jsonl'
    with open(output_file, 'w') as f:
        for data in all_matched_data:
            f.write(json.dumps(data) + '\n')
    print(f"==> Saved matched data to {output_file}.")

if __name__ == '__main__':
    main()