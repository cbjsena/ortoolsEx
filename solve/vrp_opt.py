from common_run_opt import get_solving_time_sec
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp # OR-Tools VRP
import datetime  # 파일명 생성 등에 사용 가능
from math import sqrt
import json
from common_run_opt import *
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)


def run_vrp_optimizer(input_data):
    """
    OR-Tools를 사용하여 VRP 문제를 해결합니다.
    depot_location: {'x': 0, 'y': 0}
    customer_locations: [{'id': 'C1', 'x': 1, 'y': 2}, ...]
    num_vehicles: 차량 수
    """
    logger.info(f"Running VRP Optimizer.")
    depot_location=input_data.get('depot_location')
    customer_locations=input_data.get('customer_locations')
    num_vehicles=input_data.get('num_vehicles')

    data = {}
    # 위치 데이터: 차고지를 0번 인덱스로 추가
    data['locations'] = [(depot_location['x'], depot_location['y'])] + \
                        [(loc['x'], loc['y']) for loc in customer_locations]

    data['num_vehicles'] = num_vehicles
    data['depot'] = 0  # 차고지 인덱스
    logger.debug(f"Locations for solver (depot at index 0): {data['locations']}")

    # --- 거리 행렬 생성 (유클리드 거리) ---
    num_locations = len(data['locations'])
    distance_matrix = [[0] * num_locations for _ in range(num_locations)]
    for from_node in range(num_locations):
        for to_node in range(num_locations):
            if from_node == to_node:
                distance_matrix[from_node][to_node] = 0
            else:
                loc1 = data['locations'][from_node]
                loc2 = data['locations'][to_node]
                dist = sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
                distance_matrix[from_node][to_node] = int(round(dist * 100))  # 정수로 변환 (OR-Tools는 정수 비용 선호)
                # 실제 거리로 사용하려면 float 처리 및 솔버 설정 필요
    data['distance_matrix'] = distance_matrix
    logger.debug("Distance matrix created.")

    # --- OR-Tools VRP 모델 생성 ---
    try:
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    except Exception as e:
        logger.error(f"Error creating RoutingIndexManager or RoutingModel: {e}", exc_info=True)
        return None, f"OR-Tools 라우팅 모델 생성 중 오류: {e}", 0.0

    # --- 거리 콜백 정의 ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    logger.debug("Distance callback registered.")

    # --- 검색 파라미터 설정 (선택 사항) ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.FromSeconds(5) # 시간 제한

    solution, status, processing_time = ortools_routing_solving_log(routing, search_parameters, "VRP")

    # --- 결과 추출 ---
    vrp_results = {'routes': [], 'total_distance': 0, 'dropped_nodes': []}
    error_msg = None

    if solution and status in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
                                         routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL]:
        logger.info(f'Objective (total distance): {solution.ObjectiveValue()}')
        vrp_results['total_distance'] = solution.ObjectiveValue() / 100.0  # 원래 거리로 환산

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_for_vehicle_nodes = []
            route_for_vehicle_coords = []
            route_distance = 0

            current_node = manager.IndexToNode(index)
            route_for_vehicle_nodes.append(current_node)
            route_for_vehicle_coords.append(data['locations'][current_node])

            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))

                current_node = manager.IndexToNode(index)
                route_for_vehicle_nodes.append(current_node)
                route_for_vehicle_coords.append(data['locations'][current_node])

                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            if len(route_for_vehicle_nodes) > 2:  # 차고지 -> 차고지 외에 실제 방문지가 있을 경우
                vrp_results['routes'].append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_for_vehicle_nodes,  # 노드 인덱스 (0: depot, 1~N: customers)
                    'route_locations': route_for_vehicle_coords,  # 실제 (x,y) 좌표
                    'distance': route_distance / 100.0  # 원래 거리로 환산
                })

        # 방문하지 못한 노드 확인 (모든 고객을 방문해야 하는 기본 VRP에서는 발생하지 않아야 함)
        # routing.AddDisjunction() 등을 사용하면 발생 가능
        for node in range(1, len(data['locations'])):  # 고객 노드만 (차고지 제외)
            if routing.IsStart(solution.Value(routing.NextVar(manager.NodeToIndex(node)))) and \
                    routing.IsEnd(solution.Value(routing.NextVar(manager.NodeToIndex(node)))):
                if node not in vrp_results['dropped_nodes']:  # AddDisjunction으로 인해 drop된 경우
                    logger.warning(f'Node {node} was dropped.')
                    vrp_results['dropped_nodes'].append(manager.NodeToIndex(node))


    elif routing.status() == routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT:
        error_msg = "솔버 시간 제한으로 인해 해를 찾지 못했습니다."
    elif routing.status() == routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL:
        error_msg = "내부 오류 또는 제약 조건 충족 불가로 해를 찾지 못했습니다 (ROUTING_FAIL)."
    elif routing.status() == routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID:
        error_msg = "모델 또는 파라미터가 유효하지 않습니다 (ROUTING_INVALID)."
    else:
        error_msg = f"최적 경로를 찾지 못했습니다. (솔버 상태: {routing.status()})"

    if error_msg:
        logger.error(f"VRP optimization failed or no solution: {error_msg}")

    return vrp_results, error_msg, processing_time


with open('../test_data/routing_vrp_data/dep1_cus3_veh1.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)
# test_data = json.loads("testcase/matcing_cf_tft_test1.json")

vrp_results_data, error_msg_opt, processing_time_ms = run_vrp_optimizer(input_data)

if error_msg_opt:
    logger.info(error_msg_opt)
elif vrp_results_data and vrp_results_data['routes']:
    logger.info(f"VRP optimization successful. Total distance: {vrp_results_data['total_distance']:.2f}")

    # --- 차트용 데이터 준비 ---
    plot_data = {'locations': [], 'routes': [], 'depot_index': 0}
    # 모든 위치 (차고지 + 고객)
    depot_location = input_data.get('depot_location')
    customer_locations = input_data.get('customer_locations')
    plot_data['locations'].append({'id': 'Depot', 'x': depot_location['x'], 'y': depot_location['y']})
    for i, cust_loc in enumerate(customer_locations):
        plot_data['locations'].append({'id': cust_loc['id'], 'x': cust_loc['x'], 'y': cust_loc['y']})

    # 경로 데이터 (좌표 시퀀스)
    for route_info in vrp_results_data['routes']:
        plot_data['routes'].append({
            'vehicle_id': route_info['vehicle_id'],
            'path_coords': route_info['route_locations']  # 이미 (x,y) 튜플의 리스트임
        })
    json_data = json.dumps(plot_data)  # JSON 문자열로 전달
    logger.info(f"Plot data prepared: {plot_data}")
