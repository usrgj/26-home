# run.py
import sys

tasks = {
    "1": "task1",
    "2": "task2",
    "3": "task3",
    "4": "task4",
}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else input("选择任务 (1-4): ")
    module = __import__(f"{tasks[choice]}.main", fromlist=["main"])
    module.main()