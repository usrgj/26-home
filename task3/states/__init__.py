from task3.states.idle import Idle
from task3.states.nav_to_laundry import NavToLaundry
from task3.states.pick_from_basket import PickFromBasket
from task3.states.open_washer import OpenWasher
from task3.states.pick_from_washer import PickFromWasher
from task3.states.transport_to_table import TransportToTable
from task3.states.fold_one import FoldOne
from task3.states.decide_next import DecideNext
from task3.states.finished import Finished
from task3.states.error_recovery import ErrorRecovery

ALL_STATES = {
    "idle":               Idle(),
    "nav_to_laundry":     NavToLaundry(),
    "pick_from_basket":   PickFromBasket(),
    "open_washer":        OpenWasher(),
    "pick_from_washer":   PickFromWasher(),
    "transport_to_table": TransportToTable(),
    "fold_one":           FoldOne(),
    "decide_next":        DecideNext(),
    "finished":           Finished(),
    "error_recovery":     ErrorRecovery(),
}
