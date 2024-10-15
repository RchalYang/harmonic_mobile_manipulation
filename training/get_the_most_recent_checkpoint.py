import os
import sys
import re

# output/checkpoints/CleaningTable/time/output/checkpoints/CleaningTable/time/

# Check if an argument is provided
if len(sys.argv) < 2:
    print("Please provide a directory path as an argument.")
    sys.exit(1)

# The first command line argument is the directory path
directory_path = sys.argv[1]
target_stage = sys.argv[2] if len(sys.argv) >= 3 else "00"
max_step = int(sys.argv[3]) if len(sys.argv) >= 4 else None
# directory_path = os.path.isdir(directory_path) # Output Root

# Check if the provided path is a valid directory
if not os.path.isdir(directory_path):
    print(f"The provided path '{directory_path}' is not a valid directory.")
    sys.exit(1)

checkpoints_path = os.path.join(directory_path, "checkpoints")
# print(checkpoints_path)
task_directory = os.path.join(checkpoints_path, os.listdir(checkpoints_path)[0]) # Assume only one task directory
# print(task_directory)
# List all subdirectories


def find_subdir_with_most_files(directory):
    max_file_count = 0
    subdir_with_most_files = None

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            file_count = len([file for file in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, file))])
            if file_count > max_file_count:
                max_file_count = file_count
                subdir_with_most_files = subdir

    return subdir_with_most_files, max_file_count


# exp_subdirectories = [
#     d for d in os.listdir(task_directory) if os.path.isdir(os.path.join(task_directory, d))
# ]

# # Sort the subdirectories in descending order
# sorted_subdirectories = sorted(exp_subdirectories, reverse=True)

# # The most recent directory
# most_recent_directory = os.path.join(task_directory, sorted_subdirectories[0])
most_file_directory, count = find_subdir_with_most_files(task_directory)
most_file_directory = os.path.join(task_directory, most_file_directory)
# print(most_file_directory)
# print(most_recent_directory)

def find_largest_checkpoint(directory, target_stage, max_step):
    # Regular expression to match the desired file format and extract stage and step count
    pattern = re.compile(r'stage_([\d]{2})__steps_([\d]+)\.pt')

    largest_step_count = -1
    largest_file = None

    for file in os.listdir(directory):
        match = pattern.search(file)
        if match:
            stage, steps = match.groups()
            steps = int(steps)

            if stage == target_stage and steps > largest_step_count:
                if max_step is not None and steps > max_step:
                    continue
                largest_step_count = steps
                largest_file = file

    return largest_file

largest_checkpoint = find_largest_checkpoint(most_file_directory, target_stage=target_stage, max_step=max_step)
# print("Most recent directory:", most_recent_directory)
largest_checkpoint = os.path.join(most_file_directory, largest_checkpoint)
# print(most_recent_checkpoint)
print(largest_checkpoint)