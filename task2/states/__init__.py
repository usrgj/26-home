from task2.states.init_and_wait import InitAndWait
from task2.states.kitchen_task import KitchenTask
from task2.states.release import Release
from task2.states.finished import Finished
from task2.states.error_recovery import ErrorRecovery

ALL_STATES = {
    "init":           InitAndWait(),
    "kitchen_task":   KitchenTask(),
    "release":        Release(),
    "finished":       Finished(),
    "error_recovery": ErrorRecovery(),
}
