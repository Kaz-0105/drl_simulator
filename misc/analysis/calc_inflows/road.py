class Road:
    def __init__(self, id, direction, road_id, segment_id):
        self.id = id
        self.direction = direction
        self.road_id = road_id
        self.segment_id = segment_id
        self.inflow = None
        self.exact_inflow = None
        return