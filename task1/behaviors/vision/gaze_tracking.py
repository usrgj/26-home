import cv2
import numpy as np

def get_person_direction(person_bbox, frame_width, frame_height=None, threshold=0.15, k=0.15):
    """
    判断人体在画面中的位置，目标点为比例法估算的人脸中心
    k: 人脸中心在人体框高度的比例 (通常0.12~0.18)
    """
    x1, y1, x2, y2 = person_bbox
    H = y2 - y1
    person_center_x = (x1 + x2) / 2
    person_center_y = y1 + k * H  # 这里是脸部中心的y坐标

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2 if frame_height else 0

    offset_x = person_center_x - frame_center_x
    offset_y = person_center_y - frame_center_y

    if offset_x < -50:
        h_direction = 'left'
    elif offset_x > 50:
        h_direction = 'right'
    else:
        h_direction = 'center'

    if offset_y < -50:
        v_direction = 'up'
    elif offset_y > 50:
        v_direction = 'down'
    else:
        v_direction = 'center'

    angle = (offset_x / frame_width) * 60
    need_turn = abs(offset_x) > 30 or abs(offset_y) > 30

    return {
        'direction': h_direction,
        'vertical': v_direction,
        'angle': angle,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'need_turn': need_turn
    }

def draw_direction_indicator(frame, person_bbox, control_info):
    """绘制方向指示"""
    x1, y1, x2, y2 = person_bbox
    person_center_x = int((x1 + x2) / 2)
    person_center_y = int((y1 + y2) / 2)
    
    cv2.circle(frame, (person_center_x, person_center_y), 5, (0, 255, 0), -1)
    
    direction = control_info['direction']
    if direction == 'left':
        cv2.arrowedLine(frame, (person_center_x, person_center_y), 
                       (person_center_x - 50, person_center_y), 
                       (0, 0, 255), 2)
    elif direction == 'right':
        cv2.arrowedLine(frame, (person_center_x, person_center_y), 
                       (person_center_x + 50, person_center_y), 
                       (0, 0, 255), 2)
    else:
        cv2.circle(frame, (person_center_x, person_center_y), 10, (0, 255, 0), 2)
    
    info_text = f"Dir: {control_info['direction']} | Angle: {control_info['angle']:.1f}°"
    cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame
def get_face_center_from_body_bbox(bbox, k=0.15):
    x1, y1, x2, y2 = bbox
    H = y2 - y1
    cx = int((x1 + x2) / 2)
    cy = int(y1 + k * H)
    return cx, cy


