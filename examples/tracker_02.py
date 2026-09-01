import math


class Tracker:
    def __init__(self, max_distance=80, max_missed=10):
        self.center_points = {}
        self.missed_frames = {}
        self.id_count = 0
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, objects_rect):
        objects_bbs_ids = []
        used_ids = set()

        for x1, y1, x2, y2 in objects_rect:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            matched_id = None
            closest_distance = self.max_distance

            for object_id, point in self.center_points.items():
                if object_id in used_ids:
                    continue

                distance = math.hypot(cx - point[0], cy - point[1])
                if distance < closest_distance:
                    closest_distance = distance
                    matched_id = object_id

            if matched_id is None:
                matched_id = self.id_count
                self.id_count += 1

            self.center_points[matched_id] = (cx, cy)
            self.missed_frames[matched_id] = 0
            used_ids.add(matched_id)
            objects_bbs_ids.append([x1, y1, x2, y2, matched_id])

        for object_id in list(self.center_points):
            if object_id not in used_ids:
                self.missed_frames[object_id] += 1
                if self.missed_frames[object_id] > self.max_missed:
                    del self.center_points[object_id]
                    del self.missed_frames[object_id]

        return objects_bbs_ids