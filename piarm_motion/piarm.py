from robot_hat import Servo
import time
import heapq
import random 


# Initialize the servos on the appropriate channels
base_servo = Servo(0)       # Servo on P00 (channel 0)
zeroing_pin = Servo(3)      # Servo on P03 (channel 3)
fixed_servo = Servo(7)      # Servo on P07 (channel 7)
arm_servo = Servo(5)        # Servo on P05 (channel 5)

# Functions to control servos
def move_base_left():
    print("Moving base left...")
    base_servo.angle(90)   # Move base left
    time.sleep(2)          # Delay for 2 seconds

def move_shovel_down():
    print("Moving shovel up...")
    zeroing_pin.angle(70)  # Move shovel down
    time.sleep(2)          # Delay for 2 seconds

def move_shovel_up():
    print("Moving shovel down...")
    zeroing_pin.angle(0)   # Move shovel up
    time.sleep(2)          # Delay for 2 seconds

def move_base_right():
    print("Moving base right...")
    base_servo.angle(0)    # Move base right
    time.sleep(2)          # Delay for 2 seconds

def maintain_fixed_position():
    print("Maintaining fixed position...")
    fixed_servo.angle(90)  # Set servo to a fixed position
    time.sleep(1)          # Delay for 1 second

def move_arm_up():
    print("Moving arm up...")
    arm_servo.angle(40)    # Move arm up
    time.sleep(2)          # Delay for 2 seconds

def move_arm_down():
    print("Moving arm down...")
    arm_servo.angle(10)    # Move arm down
    time.sleep(2)          # Delay for 2 seconds


import time
def freeze_servo(Servo):
    print("freezing servo...")
    zeroing_pin.angle(70)  # Move shovel down
    while True:
      time.sleep(3)
      print("freezing servo...") 




# Task 1 function incorporating the robot arm movements
def task1():
    if random.random() < 0.20:
    	print("Executing Task 1: Robot Arm Control: Critcal Execution")
    	#maintain_fixed_position()
    	move_base_left()
    	move_arm_down()
    	move_shovel_down()
    	move_base_right()
    	move_arm_up()
    	move_shovel_up()
    else:
    	print("Executing Task 1: Robot Arm Control: Typical Execution")
    	move_arm_down()
    	move_shovel_down()
    	move_arm_up()
    	move_shovel_up()


def task2():
    print("Executing Task 2")
    time.sleep(5) 

def task3():
    global count
    rt=predict(victim_response_time)

    if rt[count]>threshold:
      print("observer predicts critcal arrival")
      freeze_servo(Servo)
    count+=1
    print("Executing Task 3")
    time.sleep(2)


tasks = {
    "Task 1": {"period": 30, "execution_time": 13, "function": task1, "next_arrival": 0},  # Task 1 with robot arm movements
    "Task 2": {"period": 50, "execution_time": 5, "function": task2, "next_arrival": 0},   # Task 2 with sleep
    "Task 3": {"period": 100, "execution_time": 2, "function": task3, "next_arrival": 0}   # Task 3 with sleep
}

# RM Scheduling Priority
task_priority = ["Task 1", "Task 2", "Task 3"]

# Global start time to simulate clock starting at zero
system_start_time = None
count=0 

# Function to simulate task execution
def execute_task(task_name, current_time):
    count=0
    global system_start_time
    task = tasks[task_name]
    
    if system_start_time is None:
        system_start_time = time.time()
    
    start_time = time.time()
    relative_start_time = start_time - system_start_time
    print(f"Starting {task_name} at time: {relative_start_time:.2f} seconds")
    task["function"]()  # Call the task's function
    
    finish_time = time.time()
    relative_finish_time = finish_time - system_start_time
    elapsed_time = finish_time - start_time  # Calculate the task's execution duration
    task["next_arrival"] += task["period"]  # Schedule the next arrival time for the task
    print(f"Finished {task_name} execution at time: {relative_finish_time:.2f} seconds)")
    print(f"Elapsed time for {task_name}: {elapsed_time:.2f} seconds\n")

def rm_scheduler(simulation_time):
    # Initialize task queues
    task_queue = []

    # Initialize current time
    current_time = 0

    while current_time < simulation_time:
        # Check for tasks that are ready to run
        for task_name in task_priority:
            task = tasks[task_name]

            # If the task's next arrival time has been reached, add it to the queue
            if current_time >= task["next_arrival"]:
                heapq.heappush(task_queue, (task["period"], task_name))

        # Execute the highest priority task in the queue if available
        if task_queue:
            # Get the task with the highest priority (lowest period)
            _, task_name = heapq.heappop(task_queue)

            # Check if the current time is still valid for this task
            if current_time >= tasks[task_name]["next_arrival"]:
                execute_task(task_name, current_time)
                # Update current time by the task's execution time
                current_time += tasks[task_name]["execution_time"]
            else:
                # Handle any potential inconsistencies, if needed
                pass
        else:
            next_task_time = min(task["next_arrival"] for task in tasks.values())  # Find the next task arrival time
            time_to_wait = max(0, next_task_time - current_time)  # Calculate time to wait
            time.sleep(time_to_wait)  # Sleep for the calculated time
            current_time = next_task_time  # Advance the current time to the next task's arrival time


simulation_duration = 1500
rm_scheduler(simulation_duration)
