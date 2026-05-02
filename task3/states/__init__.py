from task3.states.init_and_wait import InitAndWait
from task3.states.nav_to_washer import NavToWasher
from task3.states.pick_from_washer import PickFromWasher
from task3.states.transport_to_table import TransportToTable
from task3.states.decide_next import DecideNext
from task3.states.release import Release
from task3.states.finished import Finished
from task3.states.error_recovery import ErrorRecovery

ALL_STATES = {
    "init":               InitAndWait(),
    "nav_to_washer":      NavToWasher(),
    "pick_from_washer":   PickFromWasher(),
    "transport_to_table": TransportToTable(),
    "decide_next":        DecideNext(),
    "release":            Release(),
    "finished":           Finished(),
    "error_recovery":     ErrorRecovery(),
}
