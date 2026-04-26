"""
gaze_controller.py

可直接import的目光追踪能力接口：
- smooth_gaze_tracking: 单次对准
- continuous_gaze_tracking: 持续追踪
- start_gaze_tracking_thread: 启动实时追踪线程
- get_person_direction_from_frame: 单帧方向判断
"""


import cv2
import time
import threading
from .gaze_tracking import get_person_direction

def start_gaze_tracking_thread(head_controller, tracker, cap, guest_name, model, duration=30):
    """
    能力接口：一行代码启动实时追踪线程，外部可用stop_event控制停止
    """
    stop_event = threading.Event()
    def worker():
        continuous_gaze_tracking(
            head_controller,
            tracker,
            None,
            model,
            cap,
            guest_name,
            duration=duration,
            stop_event=stop_event,
        )
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_event
def get_major_direction(yolo, frame, n=5, conf=0.5, threshold=0.15, k=0.15):
    directions = []
    for _ in range(n):
        results = yolo(frame, conf=conf, classes=0, verbose=False)
        bboxes = []
        for r in results:
            for b in r.boxes:
                bboxes.append(tuple(map(int, b.xyxy[0])))
        if not bboxes:
            directions.append('center')
            continue
        x1, y1, x2, y2 = bboxes[0]
        H = y2 - y1
        cx = (x1 + x2) // 2
        cy = int(y1 + k * H)
        frame_center_x = frame.shape[1] // 2
        offset_x = cx - frame_center_x
        if offset_x < -50:
            directions.append('left')
        elif offset_x > 50:
            directions.append('right')
        else:
            directions.append('center')
        import time; time.sleep(0.03)
    return max(set(directions), key=directions.count)

def start_gaze_tracking_with_major_direction(yolo, head_controller, frame, current_pos, step=0x500, n=5):
    major_dir = get_major_direction(yolo, frame, n)
    print(f"首次方向判定: {major_dir}")
    if major_dir == 'left':
        current_pos += step * 2
        head_controller.rotate_horizontal(current_pos)
    elif major_dir == 'right':
        current_pos -= step * 2
        head_controller.rotate_horizontal(current_pos)

def smooth_gaze_tracking(head_controller, tracker, frame, model, cap, guest_name, axis='horizontal'):
    """平滑目光追踪 - 让人物移动到画面中心"""
    info = tracker.get_person_info(tracker.target_guests.get(guest_name))
    if not info:
        print(f"❌ {guest_name}未绑定")
        return
    
    print(f"\n=== 平滑转向{guest_name}（{axis}） ===")
    
    current_pos = 0
    max_iter = 20  # ★ 增加迭代次数
    step = 0x400
    threshold = 15  # ★ 减小阈值到15px，更精确
    
    for iteration in range(max_iter):
        ctrl = get_person_direction(info['bbox'], frame.shape[1], frame.shape[0])
        
        # ★ 根据轴选择偏移量
        if axis == 'horizontal':
            offset = ctrl['offset_x']
        else:
            offset = ctrl['offset_y']
        
        print(f"迭代{iteration+1}: 偏移{offset:.0f}px")
        
        if abs(offset) < threshold:
            print(f"✓ 转向完成，误差{abs(offset):.0f}px < {threshold}px")
            break
        
        try:
            if abs(offset) > 150:
                current_step = step * 2
            elif abs(offset) > 80:
                current_step = step
            else:
                current_step = step // 2
            
            # 反向：人在右边/下边，头向左/上转
            if offset < -threshold:
                print(f"  → 向正方向转 {current_step:X}")
                current_pos += current_step
            elif offset > threshold:
                print(f"  → 向负方向转 {current_step:X}")
                current_pos -= current_step
            
            if axis == 'horizontal':
                head_controller.rotate_horizontal(current_pos)
            else:
                head_controller.rotate_vertical(current_pos)
            
            print(f"  当前位置: {current_pos:X}")
            
            import time
            time.sleep(0.2)
            
            ret, frame = cap.read()
            if ret:
                dets = model(frame, conf=0.5, classes=0, verbose=False)
                detections = [{'bbox': tuple(map(int, b.xyxy[0]))} for r in dets for b in r.boxes]
                matched = tracker.update(detections, frame)
                info = tracker.get_person_info(tracker.target_guests.get(guest_name))
                if not info:
                    print("❌ 人物丢失")
                    break
        
        except Exception as e:
            print(f"  ❌ 转向失败: {e}")
            break
    
    print(f"✓ {guest_name}转向完成\n")

def continuous_gaze_tracking(
    head_controller,
    tracker,
    frame,
    model,
    cap,
    target,
    duration=30,
    stop_event=None,
):
    """持续目光追踪 - 自动跟随人物移动
    Args:
        head_controller: 头部控制器实例
        tracker: ReID追踪器实例
        frame: 当前帧
        model: YOLO模型
        cap: 摄像头对象
        target: 目标客人名称 ('tom' 或 'jack')
        duration: 持续时间（秒）
    """
    import time
    start_time = time.time()
    
    current_pos_h = 0
    current_pos_v = 0
    
    threshold = 25
    step = 0x300
    
    print(f"🎯 追踪 {target.upper()}，持续 {duration} 秒（按ESC提前退出）...")
    
    frame_count = 0
    while time.time() - start_time < duration:
        if stop_event is not None and stop_event.is_set():
            print(f"✓ {target.upper()} 追踪线程被外部终止")
            break

        info = tracker.get_person_info(tracker.target_guests.get(target))
        if not info:
            print("⚠️ 人物丢失，等待重新检测...")
            time.sleep(0.1)
            continue

        ctrl = get_person_direction(info['bbox'], frame.shape[1], frame.shape[0])
        offset_x = ctrl['offset_x']
        offset_y = ctrl['offset_y']
        
        adjusted = False
        
        if abs(offset_x) > threshold:
            if offset_x > 0:
                current_pos_h -= step
            else:
                current_pos_h += step
            head_controller.rotate_horizontal(current_pos_h)
            adjusted = True
            print(f"  ↔ 水平调整: {offset_x:.0f}px")
        
        if abs(offset_y) > threshold:
            if offset_y > 0:
                current_pos_v -= step
            else:
                current_pos_v += step
            head_controller.rotate_vertical(current_pos_v)
            adjusted = True
            print(f"  ↕ 竖直调整: {offset_y:.0f}px")
        
        ret, frame = cap.read()
        if ret:
            dets = model(frame, conf=0.5, classes=0, verbose=False)
            detections = [{'bbox': tuple(map(int, b.xyxy[0]))} for r in dets for b in r.boxes]
            matched = tracker.update(detections, frame)
            
            # 显示画面
            for pid in matched:
                info_display = tracker.get_person_info(pid)
                if info_display:
                    x1, y1, x2, y2 = info_display['bbox']
                    name = info_display['name'] or f"ID{pid}"
                    col = (0, 255, 0) if info_display['name'] else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    
                    # ★ 显示偏移信息
                    if info_display['name'] == target:
                        cv2.putText(frame, f"Offset: X={offset_x:.0f} Y={offset_y:.0f}", 
                                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # ★ 显示追踪状态
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"TRACKING: {target.upper()} ({elapsed}s/{duration}s)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('RoboCup@Home', frame)
            
            # ★ 按ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹ 用户中断追踪")
                break
        
        time.sleep(0.1 if adjusted else 0.05)
    
    print(f"✓ {target.upper()} 追踪完成")



def introduce_guests(head_controller, tracker, frame, model, cap):
    """自动介绍流程 - 完整的相互介绍
    
    Args:
        head_controller: 头部控制器实例
        tracker: ReID追踪器实例
        frame: 当前帧
        model: YOLO模型
        cap: 摄像头对象
    """
    print("\n" + "="*50)
    print("🎤 开始介绍流程")
    print("="*50)
    
    # 1. 看Tom，介绍Jack
    print("\n[1/6] 看向Tom，准备介绍Jack...")
    continuous_gaze_tracking(head_controller, tracker, frame, model, cap, 'tom', duration=3)
    print("💬 机器人说：Tom, this is Jack.")
    
    # 2. 看Jack（让Tom看到Jack）
    print("\n[2/6] 转向Jack...")
    continuous_gaze_tracking(head_controller, tracker, frame, model, cap, 'jack', duration=4)
    
    # 3. 看回Tom
    print("\n[3/6] 转回Tom...")
    continuous_gaze_tracking(head_controller, tracker, frame, model, cap, 'tom', duration=2)
    
    # 4. 看Jack，介绍Tom
    print("\n[4/6] 看向Jack，准备介绍Tom...")
    continuous_gaze_tracking(head_controller, tracker, frame, model, cap, 'jack', duration=3)
    print("💬 机器人说：Jack, this is Tom.")
    
    # 5. 看Tom（让Jack看到Tom）
    print("\n[5/6] 转向Tom...")
    continuous_gaze_tracking(head_controller, tracker, frame, model, cap, 'tom', duration=4)
    
    # 6. 回中
    print("\n[6/6] 回中...")
    head_controller.home()
    
    print("\n" + "="*50)
    print("✅ 介绍流程完成")
    print("="*50 + "\n")