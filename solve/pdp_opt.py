from common_run_opt import get_solving_time_sec
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp # OR-Tools VRP
import datetime  # 파일명 생성 등에 사용 가능
from math import sqrt
import json
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)


def run_pdp_optimizer(input_data):
    """
    OR-Tools를 사용하여 Pickup and Delivery Problem을 해결합니다.
    """
    logger.info("Running PDP Optimizer.")
    depot_location = input_data.get('depot_location')
    pickup_delivery_pairs = input_data.get('pickup_delivery_pairs')
    num_vehicles = input_data.get('num_vehicles')
    vehicle_capacities = input_data.get('vehicle_capacities')

    if not all([depot_location, pickup_delivery_pairs, num_vehicles, vehicle_capacities]):
        return None, "오류: 필수 입력 데이터(차고지, 수거/배송 쌍, 차량 정보)가 누락되었습니다.", 0.0

    # --- 데이터 구조 생성 ---
    data = {}
    # 위치 리스트: [depot, p1, d1, p2, d2, ...]
    locations = [(depot_location['x'], depot_location['y'])]
    demands = [0]  # 차고지 수요는 0
    pd_pair_indices = []  # [(p1_idx, d1_idx), (p2_idx, d2_idx), ...]

    current_node_index = 1
    for pair in pickup_delivery_pairs:
        # 수거 지점 추가
        locations.append((pair['pickup']['x'], pair['pickup']['y']))
        demands.append(pair['demand'])  # 수거 시 수요량은 양수
        pickup_node_index = current_node_index
        current_node_index += 1

        # 배송 지점 추가
        locations.append((pair['delivery']['x'], pair['delivery']['y']))
        demands.append(-pair['demand'])  # 배송 시 수요량은 음수
        delivery_node_index = current_node_index
        current_node_index += 1

        pd_pair_indices.append((pickup_node_index, delivery_node_index))

    data['locations'] = locations
    data['demands'] = demands
    data['pickups_deliveries'] = pd_pair_indices
    data['num_vehicles'] = num_vehicles
    data['vehicle_capacities'] = vehicle_capacities
    data['depot'] = 0

    logger.debug(f"PDP Data prepared. Locations: {len(data['locations'])}, Pairs: {len(data['pickups_deliveries'])}")

    # --- 거리 행렬 생성 ---
    num_locations = len(data['locations'])
    distance_matrix = [[0] * num_locations for _ in range(num_locations)]
    for from_node in range(num_locations):
        for to_node in range(num_locations):
            if from_node != to_node:
                dist = sqrt((locations[from_node][0] - locations[to_node][0]) ** 2 + (
                            locations[from_node][1] - locations[to_node][1]) ** 2)
                distance_matrix[from_node][to_node] = int(round(dist * 100))

    data['distance_matrix'] = distance_matrix

    # --- OR-Tools PDP 모델 생성 ---
    try:
        manager = pywrapcp.RoutingIndexManager(num_locations, data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
    except Exception as e:
        logger.error(f"Error creating Routing Model for PDP: {e}", exc_info=True)
        return None, f"OR-Tools 라우팅 모델 생성 중 오류 (PDP): {e}", 0.0

    # 거리 콜백 및 비용 설정
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 수요 콜백 및 용량 차원 설정
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
    )

    # --- 수거 및 배송 제약 조건 설정 ---
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        # 같은 차량으로 방문해야 한다는 제약
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        # 시간 차원을 추가하면 선행 제약도 자동으로 걸림
        # 여기서는 AddPickupAndDelivery가 기본적인 선행/쌍 제약을 처리함.
        # routing.AddDisjunction([pickup_index], penalty) # 방문하지 않을 경우 패널티 (선택)
    logger.debug("Pickup and Delivery constraints added.")

    # 검색 파라미터 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    search_parameters.time_limit.FromSeconds(5)  # 시간 제한

    logger.info("Solving PDP model...")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"PDP Solver finished. Status: {routing.status()}, Time: {processing_time_ms:.2f} ms")

    # --- 결과 추출 ---
    pdp_results = {'routes': [], 'total_distance': 0}
    error_msg = None

    if solution and routing.status() in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
                                         routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL]:
        pdp_results['total_distance'] = solution.ObjectiveValue() / 100.0
        capacity_dimension = routing.GetDimensionOrDie('Capacity')

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_nodes = []
            route_locations = []
            route_loads = []
            route_distance = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                load_var = capacity_dimension.CumulVar(index)
                route_nodes.append(node_index)
                route_locations.append(data['locations'][node_index])
                route_loads.append(solution.Value(load_var))

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            end_node_index = manager.IndexToNode(index)
            load_var = capacity_dimension.CumulVar(index)
            route_nodes.append(end_node_index)
            route_locations.append(data['locations'][end_node_index])
            route_loads.append(solution.Value(load_var))

            if len(route_nodes) > 2:
                pdp_results['routes'].append({
                    'vehicle_id': vehicle_id,
                    'route_nodes': route_nodes,
                    'route_locations': route_locations,
                    'route_loads': route_loads,  # 각 지점 도착 시 적재량
                    'distance': route_distance / 100.0,
                    'capacity': data['vehicle_capacities'][vehicle_id]
                })
    else:
        # ... (이전 CVRP의 오류 메시지 처리와 유사) ...
        error_msg = f"PDP 최적 경로를 찾지 못했습니다. (솔버 상태: {routing.status()})"
        logger.error(f"PDP optimization failed: {error_msg}")

    return pdp_results, error_msg, processing_time_ms


with open('../test_data/routing_pdp_data/dep1_pair3_veh3.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)
# test_data = json.loads("testcase/matcing_cf_tft_test1.json")

results_data, error_msg_opt, processing_time_ms = run_pdp_optimizer(input_data)

if error_msg_opt:
    logger.info(error_msg_opt)
elif results_data:
    logger.info(f"PDP 최적 경로 계산 완료! 총 거리: {results_data.get('total_distance', 0):.2f}")

    # 차트용 데이터 준비
    plot_data = {'locations': [], 'routes': [], 'depot_index': 0, 'pairs': []}
    plot_data['locations'].append(
        {'id': 'Depot', 'x': input_data['depot_location']['x'], 'y': input_data['depot_location']['y']})

    node_idx_counter = 1
    for i, pair in enumerate(input_data['pickup_delivery_pairs']):
        plot_data['locations'].append(
            {'id': f"{pair['id']}-P", 'x': pair['pickup']['x'], 'y': pair['pickup']['y']})
        plot_data['locations'].append(
            {'id': f"{pair['id']}-D", 'x': pair['delivery']['x'], 'y': pair['delivery']['y']})
        plot_data['pairs'].append({'p_idx': node_idx_counter, 'd_idx': node_idx_counter + 1})
        node_idx_counter += 2

    for route_info in results_data.get('routes', []):
        plot_data['routes'].append({
            'vehicle_id': route_info['vehicle_id'],
            'path_coords': route_info['route_locations']
        })
    json_data = json.dumps(plot_data)
    logger.info(f"Plot data prepared: {plot_data}")
