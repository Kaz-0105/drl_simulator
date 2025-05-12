import time
from concurrent.futures import ThreadPoolExecutor

def task(name, delay):
    print(f"{name} started")
    time.sleep(delay)
    print(f"{name} finished")

def run_parallel():
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(task, "Task 1", 2),
            executor.submit(task, "Task 2", 2)
        ]
        for f in futures:
            f.result()  # 各タスクの終了を待つ

run_parallel()



print('test')