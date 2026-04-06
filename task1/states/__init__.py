from task1.states.init_and_wait import InitAndWait
from task1.states.receive_guest import ReceiveGuest
from task1.states.introduce_guests import IntroduceGuests
from task1.states.receive_bag import ReceiveBag
from task1.states.follow_and_place import FollowAndPlace
from task1.states.finished import Finished
from task1.states.error_recovery import ErrorRecovery

ALL_STATES = {
    "init":             InitAndWait(),
    "receive_guest":    ReceiveGuest(),
    "introduce":        IntroduceGuests(),
    "receive_bag":      ReceiveBag(),
    "follow_and_place": FollowAndPlace(),
    "finished":         Finished(),
    "error_recovery":   ErrorRecovery(),
}
