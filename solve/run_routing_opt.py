from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp # OR-Tools VRP
import json
from logging_config import setup_logger
import logging
import datetime  # 파일명 생성 등에 사용 가능
from math import sqrt

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
    demands=input_data.get('demands')
    vehicle_capacities=input_data.get('vehicle_capacities')

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

    logger.info("Solving VRP model...")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"VRP Solver finished. Status: {routing.status()}, Time: {processing_time_ms:.2f} ms")

    # --- 결과 추출 ---
    vrp_results = {'routes': [], 'total_distance': 0, 'dropped_nodes': []}
    error_msg = None

    if solution and routing.status() in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
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

    return vrp_results, error_msg, processing_time_ms


def run_cvrp_optimizer(input_data):
    """
    OR-Tools를 사용하여 VRP 문제를 해결합니다.
    depot_location: {'x': 0, 'y': 0}
    customer_locations: [{'id': 'C1', 'x': 1, 'y': 2}, ...]
    num_vehicles: 차량 수
    """
    logger.info("Running CVRP Optimizer.")
    logger.info(input_data)
    depot_location = input_data.get('depot_location')
    customer_locations = input_data.get('customer_locations')
    num_vehicles = input_data.get('num_vehicles')

    demands_input = [0]  # 차고지 수요는 0
    for cust in customer_locations:
        demands_input.append(cust.get('demand', 0))

    vehicle_capacities=input_data.get('vehicle_capacity')
    if not isinstance(vehicle_capacities, list):  # 단일 값으로 용량이 주어졌을 경우
        vehicle_capacities = [vehicle_capacities] * num_vehicles
    elif len(vehicle_capacities) != num_vehicles:
        logger.error("Number of vehicle capacities does not match number of vehicles. Using first capacity for all.")
        # 또는 오류 처리. 여기서는 첫 번째 용량을 모든 차량에 적용하거나, 평균값을 사용하는 등 정책 필요.
        # 가장 간단하게는 모든 차량 용량이 같다고 가정하고 하나의 값만 받도록 단순화할 수 있음.
        # 여기서는 첫 번째 용량을 사용.
        cap = vehicle_capacities[0] if vehicle_capacities else 100  # 기본값
        vehicle_capacities = [cap] * num_vehicles

    data = {}
    # 위치 데이터: 차고지를 0번 인덱스로 추가
    data['locations'] = [(depot_location['x'], depot_location['y'])] + \
                        [(loc['x'], loc['y']) for loc in customer_locations]
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0 # 차고지 인덱스
    data['demands'] = demands_input  # 수요량 추가
    data['vehicle_capacities'] = vehicle_capacities  # 차량 용량 추가

    logger.debug(f"Locations: {data['locations']}")
    logger.debug(f"Demands: {data['demands']}")
    logger.debug(f"Vehicle Capacities: {data['vehicle_capacities']}")

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
                distance_matrix[from_node][to_node] = int(round(dist * 100)) # 정수로 변환 (OR-Tools는 정수 비용 선호)
                # 실제 거리로 사용하려면 float 처리 및 솔버 설정 필요
    data['distance_matrix'] = distance_matrix
    logger.debug("Distance matrix created.")

    # --- OR-Tools CVRP 모델 생성 ---
    try:
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    except Exception as e:
        logger.error(f"Error creating RoutingIndexManager or RoutingModel for CVRP: {e}", exc_info=True)
        return None, f"OR-Tools 라우팅 모델 생성 중 오류 (CVRP): {e}", 0.0

    # --- 거리 콜백 정의 ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    logger.debug("Distance callback registered.")

    # --- 수요량 콜백 및 용량 차원(Dimension) 추가 ---
    def demand_callback(from_index):
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack (0이면 용량 초과 시 패널티 없음, 용량 제약 위반 불가)
        data['vehicle_capacities'],  # 각 차량의 용량 리스트
        True,  # start cumul to zero (차고지에서 시작 시 누적 수요 0)
        'Capacity')  # 차원의 이름
    logger.debug("Demand callback and Capacity dimension registered for CVRP.")

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.FromSeconds(5) # 시간 제한 증가 고려

    logger.info("Solving CVRP model...")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"CVRP Solver finished. Status: {routing.status()}, Time: {processing_time_ms:.2f} ms")

    # --- 결과 추출 (기존 VRP와 유사하나, 경로별 적재량 등 추가 정보 포함 가능) ---
    cvrp_results = {'routes': [], 'total_distance': 0, 'dropped_nodes': [], 'total_demand_served': 0}
    error_msg = None

    if solution and routing.status() in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
                                         routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL]:
        cvrp_results['total_distance'] = solution.ObjectiveValue() / 100.0
        total_demand_served_on_routes = 0

        capacity_dimension = routing.GetDimensionOrDie('Capacity')

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_nodes = []
            route_coords = []
            route_distance = 0
            route_load = 0  # 현재 차량의 적재량

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_nodes.append(node_index)
                route_coords.append(data['locations'][node_index])

                # 현재 노드(고객)의 수요를 차량 적재량에 추가 (차고지 제외)
                if node_index != data['depot']:
                    route_load += data['demands'][node_index]

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            # 마지막으로 차고지 노드 추가
            end_node_index = manager.IndexToNode(index)
            route_nodes.append(end_node_index)
            route_coords.append(data['locations'][end_node_index])

            if len(route_nodes) > 2:  # 실제 운행한 경로만
                cvrp_results['routes'].append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_nodes,
                    'route_locations': route_coords,
                    'distance': route_distance / 100.0,
                    'load': route_load,  # 해당 차량이 처리한 총 수요량
                    'capacity': data['vehicle_capacities'][vehicle_id]  # 해당 차량의 용량
                })
                total_demand_served_on_routes += route_load
        cvrp_results['total_demand_served'] = total_demand_served_on_routes

        # Drop된 노드 확인 (AddDisjunction 사용 시)
        # 또는 모든 고객 수요가 충족되었는지 확인 필요
        # for node_idx in range(1, num_locations): # 고객 노드만
        #     if routing.IsStart(solution.Value(routing.NextVar(manager.NodeToIndex(node_idx)))):
        #        # routing.NextVar(manager.NodeToIndex(node_idx))가 자기 자신을 가리키면 drop된 것.
        #        # 더 정확한 방법은 Penalty를 설정하고 solution.ObjectiveValue()와 비교하거나,
        #        # 모든 고객이 방문되었는지 확인하는 것입니다.
        #        pass


    else:  # 실패 시 처리 (기존과 유사)
        # ... (solver_status_map 사용하여 error_msg 설정) ...
        status_text = routing.status()  # 정수 값
        status_name_map = {getattr(routing_enums_pb2.RoutingSearchStatus, name): name
                           for name in dir(routing_enums_pb2.RoutingSearchStatus) if not name.startswith('_')}
        status_str = status_name_map.get(status_text, f"UNKNOWN_STATUS_{status_text}")

        if status_text == routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT:
            error_msg = "솔버 시간 제한으로 인해 해를 찾지 못했습니다."
        elif status_text == routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL:
            error_msg = "내부 오류 또는 제약 조건(예: 용량 초과) 충족 불가로 해를 찾지 못했습니다."
        elif status_text == routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID:
            error_msg = "모델 또는 파라미터가 유효하지 않습니다."
        else:
            error_msg = f"최적 경로를 찾지 못했습니다. (솔버 상태: {status_str})"
        logger.error(f"CVRP optimization failed or no solution: {error_msg}")

    return cvrp_results, error_msg, processing_time_ms

def run_pdp_optimizer():
    return None

with open('../test_data/routing_vrp_data/dep1_cus5_veh3.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)
model_name=input_data.get('problem_type')
results_data, error_msg_opt, processing_time_ms = run_vrp_optimizer(input_data)

if results_data and results_data['routes']:
    success_message = f"{model_name} 최적 경로 계산 완료! 총 거리: {results_data['total_distance']:.2f}"
    logger.info(success_message)

    # 차트용 데이터 준비
    plot_data = {'locations': [], 'routes': [], 'depot_index': 0}
    # 모든 위치 (차고지 + 고객)
    depot_location = input_data.get('depot_location')
    customer_locations = input_data.get('customer_locations')
    plot_data['locations'].append({'id': 'Depot', 'x': depot_location['x'], 'y': depot_location['y']})
    for i, cust_loc in enumerate(customer_locations):
        plot_data['locations'].append(
            {'id': cust_loc['id'], 'x': cust_loc['x'], 'y': cust_loc['y']})

    # 경로 데이터 (좌표 시퀀스)
    for route_info in results_data['routes']:
        plot_data['routes'].append({
            'vehicle_id': route_info['vehicle_id'],
            'path_coords': route_info['route_locations']  # 이미 (x,y) 튜플의 리스트임
        })
