#!/bin/bash

# Set the number of files per subfolder
BATCH_SIZE=200

# Get all .sdf files (sorted by name) and count the total number
files=(*.sdf)
total_files=${#files[@]}

# Check if there are any .sdf files
if [ $total_files -eq 0 ]; then
    echo "Error: No .sdf files found in the current directory."
    exit 1
fi

# Calculate the number of subfolders needed
batch_count=$(( (total_files + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Processing $total_files .sdf files, grouping into $batch_count batches..."

# Process files in batches
for (( batch=1, start=0; batch <= batch_count; batch++, start+=BATCH_SIZE )); do
    # Create a subfolder
    dir_name="batch_${batch}"
    mkdir -p "$dir_name"

    # Calculate the end index for this batch
    end=$((start + BATCH_SIZE))
    [ $end -gt $total_files ] && end=$total_files

    # Move files (using array slicing for efficiency)
    mv -v -- "${files[@]:start:BATCH_SIZE}" "$dir_name/" 2>/dev/null

    # Display progress
    moved_files=$((end - start))
    echo "Batch ${batch}: Moved ${moved_files} files to ${dir_name}/"
done

echo "Operation completed! All files have been grouped successfully."
