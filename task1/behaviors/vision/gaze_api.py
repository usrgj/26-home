import threading
import time
from gaze_tracking import get_person_direction

def start_gaze_tracking_thread(head_controller, tracker, cam, guest_name, duration=30):
    """
    能力接口：启动实时目光追踪线程，外部可用stop_event控制停止
    用法：
        gaze_thread, stop_event = start_gaze_tracking_thread(...)
        # ...需要停止时:
        stop_event.set()
        gaze_thread.join()
    """
    stop_event = threading.Event()
    def worker():
        start_time = time.time()
        current_pos_h = 0
        current_pos_v = 0
        threshold = 25
        step = 0x400
        while not stop_event.is_set() and time.time() - start_time < duration:
            info = tracker.get_person_info(tracker.target_guests.get(guest_name))
            ret, frame = cam.read()
            if not info or not ret:
                time.sleep(0.1)
                continue
            ctrl = get_person_direction(info['bbox'], frame.shape[1], frame.shape[0])
            offset_x = ctrl['offset_x']
            offset_y = ctrl['offset_y']
            if abs(offset_x) > threshold:
                if offset_x > 0:
                    current_pos_h -= step
                else:
                    current_pos_h += step
                head_controller.rotate_horizontal(current_pos_h)
            if abs(offset_y) > threshold:
                if offset_y > 0:
                    current_pos_v -= step
                else:
                    current_pos_v += step
                head_controller.rotate_vertical(current_pos_v)
            time.sleep(0.1)
        print(f"✓ {guest_name.upper()} 追踪线程结束")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_event
