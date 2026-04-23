# seat_manager.py
from typing import List, Tuple

class SeatManager:
    """
    统一的座位管理能力接口，支持团队状态机直接import使用。
    """
    def __init__(self, seat_coords: List[Tuple[int, int, int, int]], min_empty=0, max_overlap=0.3):
        """
        seat_coords: 所有已知座位的像素坐标[(x1, y1, x2, y2), ...]
        min_empty: 至少空N个座位
        max_overlap: 超过这个IOU就认为占座
        """
        self.seat_coords = seat_coords
        self.min_empty = min_empty
        self.max_overlap = max_overlap
        self.seat_status = ["unknown"] * len(seat_coords)  # "occupied", "empty"
        self.frame_memory = []  # 多帧融合投票

    def update_from_detections(self, person_boxes: List[Tuple[int, int, int, int]]):
        """
        输入所有人体框，更新所有座位是否有人（建议3帧融合后取结果更稳）
        """
        status = []
        for seat in self.seat_coords:
            max_ov = max((self.calc_iou(seat, p) for p in person_boxes), default=0)
            if max_ov > self.max_overlap:
                status.append("occupied")
            else:
                status.append("empty")
        self.frame_memory.append(status)
        if len(self.frame_memory) > 3:
            self.frame_memory.pop(0)
        self.seat_status = self.vote_frames(self.frame_memory)
        if self.seat_status.count("empty") < self.min_empty:
            self.seat_status = self.apply_deduction(self.seat_status)

    def vote_frames(self, memory: List[List[str]]) -> List[str]:
        """多帧投票决定状态"""
        voted = []
        for j in range(len(self.seat_coords)):
            st_in_memory = [frame[j] for frame in memory]
            chosen = max(set(st_in_memory), key=st_in_memory.count)
            voted.append(chosen)
        return voted

    def apply_deduction(self, status_list: List[str]) -> List[str]:
        """至少要有min_empty个空座位，强行补空"""
        for i, st in enumerate(status_list):
            if st != "empty":
                status_list[i] = "empty"
                if status_list.count("empty") >= self.min_empty:
                    break
        return status_list

    def get_empty_indices_by_distance(self, robot_pose: Tuple[int, int]) -> List[int]:
        """
        返回空座位按距离排序的索引（robot_pose为像素坐标）
        """
        empty_indices = [i for i, st in enumerate(self.seat_status) if st == "empty"]
        dists = []
        for i in empty_indices:
            x1, y1, x2, y2 = self.seat_coords[i]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dist = (cx - robot_pose[0]) ** 2 + (cy - robot_pose[1]) ** 2
            dists.append((dist, i))
        dists.sort()
        return [idx for (_, idx) in dists]

    def calc_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

