# Debug Tools
from yachalk import chalk


v = [0] * 10
index = 0
increment = 0
increments = [0.1, 0.01, 0.001, 1]

v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = v

def select(i):
    global index
    index = i
    print(f"dbg: v{index+1} = {v[index]}")

def update_values():
    global v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = v

def up(inc_offset):
    v[index] += increments[(increment + inc_offset) % len(increments)]
    update_values()
    print(chalk.green(f"dbg: v{index+1} = {v[index]}"))

def down(inc_offset):
    v[index] -= increments[(increment + inc_offset) % len(increments)]
    update_values()
    print(chalk.red(f"dbg: v{index+1} = {v[index]}"))

def cycle_increment():
    global increment
    increment = (increment + 1) % len(increments)
    print(f"dbg: increment = {increments[increment]}")
