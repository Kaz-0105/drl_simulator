import sys
from pathlib import Path
root_dir_path = (Path(__file__).parent / '..' / '..' / '..').resolve()
sys.path.append(str(root_dir_path))

import yaml
import pandas as pd
import sympy as sp

from misc.analysis.calc_inflows.road import Road
from misc.analysis.calc_inflows.intersection import Intersection

# get config_yaml
config_yaml_file_path = root_dir_path / 'misc' / 'analysis' / 'calc_inflows' / 'config.yaml'
if not config_yaml_file_path.exists():
    raise FileNotFoundError(f"{config_yaml_file_path} does not exist.")

with open(config_yaml_file_path, 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)

# get input_inflows_df
input_inflows_file_path = root_dir_path / 'misc' / 'analysis' / 'calc_inflows' / 'input_inflows.csv'
if not input_inflows_file_path.exists():
    raise FileNotFoundError(f"{input_inflows_file_path} does not exist.")

with open(input_inflows_file_path, 'r', encoding='utf-8') as f:
    input_inflows_df = pd.read_csv(f)

direction_list = ['north', 'east', 'south', 'west']
direction_order_map = {direction: idx + 1 for idx, direction in enumerate(direction_list)}
input_inflows_df['direction'] = pd.Categorical(input_inflows_df['direction'], categories=direction_list, ordered=True)
input_inflows_df = input_inflows_df.sort_values(by=['direction', 'road_id'], ascending=[True, True]).reset_index(drop=True)

# get num_roads_map
num_roads_map = {
    direction: len(input_inflows_df[input_inflows_df['direction'] == direction])
    for direction in input_inflows_df['direction'].unique()
}

if num_roads_map['north'] != num_roads_map['south'] or num_roads_map['east'] != num_roads_map['west']:
    raise ValueError("The number of roads in north and south directions must be the same, and the number of roads in east and west directions must be the same.")

# get num_segments_map
num_segments_map = {}
for direction in direction_list:
    if direction in ['north', 'south']:
        num_segments_map[direction] = num_roads_map['east'] + 1
    elif direction in ['east', 'west']:
        num_segments_map[direction] = num_roads_map['north'] + 1

# get road_map
road_map = {}
for direction in direction_list:
    tmp_input_inflows_df = input_inflows_df[input_inflows_df['direction'] == direction]
    for road_id in tmp_input_inflows_df['road_id'].tolist():
        for segment_id in range(1, num_segments_map[direction] + 1):
            road = Road(
                id=len(road_map) + 1, 
                direction=direction, 
                road_id=road_id, 
                segment_id=segment_id
            )

            road_map[direction, road_id, segment_id] = road

            if segment_id != 1:
                continue

            road.inflow = tmp_input_inflows_df[tmp_input_inflows_df['road_id'] == road_id]['inflow'].values[0]

# get intersection_map
intersection_map = {}
for row_id in range(1, num_roads_map['east'] + 1):
    for col_id in range(1, num_roads_map['north'] + 1):
        intersection = Intersection(
            id=len(intersection_map) + 1, 
            row_id=row_id,
            col_id=col_id,
            road_map=road_map,
            num_segments_map=num_segments_map,
            route_selection_list=config_yaml['route_selection']
        )
        intersection_map[row_id, col_id] = intersection

# make system of linear equations
A_mat = sp.zeros(len(road_map), len(road_map))
B_vec = sp.zeros(len(road_map), 1)

constraints_counter = 0
for road in road_map.values():
    if road.inflow is None:
        continue

    A_mat[constraints_counter, road.id - 1] = 1
    B_vec[constraints_counter, 0] = sp.S(str(road.inflow))
    constraints_counter += 1

for (row_id, col_id), intersection in intersection_map.items():
    for output_direction, output_road in intersection.output_road_map.items():
        A_mat[constraints_counter, output_road.id - 1] = 1
        sum_route_selection = sum(intersection.route_selection_list)
        for input_direction, input_road in intersection.input_road_map.items():
            if input_direction == output_direction:
                continue
            
            route_id = (direction_order_map[output_direction] - direction_order_map[input_direction] ) % 4 
            A_mat[constraints_counter, input_road.id - 1] = - sp.S(str(intersection.route_selection_list[route_id - 1])) / sp.S(str(sum_route_selection))
                
        constraints_counter += 1

inflows_vec = A_mat.LUsolve(B_vec)

for road in road_map.values():
    road.exact_inflow = inflows_vec[road.id - 1, 0]
    road.inflow = float(road.exact_inflow.evalf())


# print max-inflow intersection
max_intersection = None
max_inflow = -1
for intersection in intersection_map.values():
    if intersection.inflow > max_inflow:
        max_intersection = intersection
        max_inflow = intersection.inflow

print('=================================================')
print('max-inflow intersection')
print(f"id : {max_intersection.id}")
print(f"(row_id, col_id) : ({max_intersection.row_id}, {max_intersection.col_id})")
print(f"exact inflow : {int(max_intersection.exact_inflow // 1)} + {max_intersection.exact_inflow - int(max_intersection.exact_inflow // 1)}")
print(f"inflow : {max_intersection.inflow}")

# print average-inflow intersection
sum_exact_inflow = sum(intersection.exact_inflow for intersection in intersection_map.values())
average_exact_inflow = sum_exact_inflow / len(intersection_map)

print('=================================================')
print('average intersection')
print(f"exact inflow : {int(average_exact_inflow // 1)} + {average_exact_inflow - int(average_exact_inflow // 1)}")
print(f"inflow : {float(average_exact_inflow.evalf())}")




