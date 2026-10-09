#!/bin/bash

# Function to extract specific data
extract_data() {
    local start_line=$1
    local step=$2
    local column=$3
    local output_file=$4

    # Extract rows based on start and step
    sed -n "${start_line}~${step}p" ./STATIS > temp_data

    # Extract specific column
    awk -v col="$column" '{print $col}' temp_data > "$output_file"

    # Add header and format
    sed -i '1 i\header' "$output_file"
    sed -i '1,$s/ /\t/g' "$output_file"

    # Clean up
    rm temp_data
}

# Example usage for temperature extraction
# so the line below is saying start on line 4, skip 13 lines, read column 2, make a file called npt_temp_2
extract_data 4 13 2 npt_temp_2
extract_data 13 13 1 npt_side_a_2
extract_data 13 13 5 npt_side_c_2
extract_data 4 13 1 npt_energy

# Add more calls here for different rows/columns

# if this wont run due to errors after editing, run "sed -i 's/\r$//' npt_data_extract.bash" first
# obviously make sure the file name is the right one for however you decide to name this
