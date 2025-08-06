def split_text(text_path):
    with open(text_path, 'r') as file:
        content = file.read()
    labels = []
    for line in content.split('\n'):
        if line:
            labels.append(line.split(':')[1])
    labels = list(set(labels))

    total = len(labels)
    half_lines = total // 2

    first_half = '\n'.join(labels[:half_lines])
    second_half = '\n'.join(labels[half_lines:])
    return first_half, second_half

file_name = 'neg_labels_for_gener_ood'
input_file = f'{file_name}.txt'
output_file1 = f'{file_name}_part1.txt'
output_file2 = f'{file_name}_part2.txt'    

# Usage example
first_half, second_half = split_text(input_file)
with open(output_file1, 'w') as file:
        file.write(first_half)

with open(output_file2, 'w') as file:
        file.write(second_half)
