# RT_Information_Leakage
*Investigating Timing-Based Information Leakage in Data Flow-Driven Real-Time Systems*

## Synthetic taskset generation: 

Generate taskset using the script `taskset_generation.py` . This script will generate two separate files for the WCET and period of each task for 100 task sets. These files need to be used in `evaluation.py` to perform different synthetic experiments.

## Timing Overhead:
To measure timing overhead run the shell script `run_c_code.sh`

## Memory Overhead:
To measure memory overhead run the shell script `run_c_code_memory.sh`

## Demonstration: Manufacturing Robot: 

We demonstrated the existence of information leakage and how attackers can use that information using a manufacturing robot (PiARM from Sunfounder).

Check out this video:  
[RT Information Leakage](https://www.youtube.com/@RT_Information_Leakage)

