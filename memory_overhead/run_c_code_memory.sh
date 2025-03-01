#!/bin/bash

# Create a log file for memory usage
log_file="memory_usage.log"
echo "Iteration,Memory_Usage(KB)" > "$log_file"

for i in {1..100}
do
    # Run the program and capture its memory usage
    /usr/bin/time -f "$i,%M" -o "$log_file" -a ./c_code
done

echo "Memory usage recorded in $log_file"
