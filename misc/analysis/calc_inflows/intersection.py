class Intersection:
    def __init__(self, id, row_id, col_id, road_map, num_segments_map, route_selection_list):
        self.id = id
        self.row_id = row_id
        self.col_id = col_id
        self.route_selection_list = route_selection_list

        self._initRoadMap(road_map, num_segments_map)
        return
    
    def _initRoadMap(self, road_map, num_segments_map):
        self.input_road_map = {
            'north': road_map['north', self.col_id, self.row_id],
            'south': road_map['south', self.col_id, num_segments_map['south'] - self.row_id],
            'east': road_map['east', self.row_id, num_segments_map['east'] - self.col_id],
            'west': road_map['west', self.row_id, self.col_id]
        }
        self.output_road_map = {
            'north': road_map['south', self.col_id, num_segments_map['south'] - self.row_id + 1],
            'south': road_map['north', self.col_id, self.row_id + 1],
            'east': road_map['west', self.row_id, self.col_id + 1],
            'west': road_map['east', self.row_id, num_segments_map['east'] - self.col_id + 1]
        }
        return

    @property
    def inflow(self):
        inflow = 0
        for input_road in self.input_road_map.values():
            if input_road.inflow is not None:
                inflow += input_road.inflow
        return inflow

    @property
    def exact_inflow(self):
        exact_inflow = 0
        for input_road in self.input_road_map.values():
            if input_road.inflow is not None:
                exact_inflow += input_road.exact_inflow
        return exact_inflow