import json

# Define the input and output file paths
file_name = 'matched_info'
input_file = f'{file_name}.jsonl'
output_file1 = f'{file_name}_part1.jsonl'
output_file2 = f'{file_name}_part2.jsonl'

# Read the input file and split the lines
with open(input_file, 'r') as file:
    lines = file.readlines()

# Split the lines into two equal parts
half = len(lines) // 2
part1 = lines[:half]
part2 = lines[half:]

# Write the first part to the output file 1
with open(output_file1, 'w') as file:
    file.writelines(part1)

# Write the second part to the output file 2
with open(output_file2, 'w') as file:
    file.writelines(part2)