from common_utils.common_data_utils import save_json_data
import logging
import datetime

logger = logging.getLogger(__name__)

preset_depot_location = {"id": "D1","x": 300.0,"y": 250.0}
preset_customer_locations = [
        {"id": "C1","x": 103.0,"y": 120.0, "demand": 30},    {"id": "C2","x": 510.0,"y": 150.0, "demand": 30},
        {"id": "C3","x": 171.0,"y": 317.0, "demand": 40},    {"id": "C4","x": 486.0,"y": 283.0, "demand": 40},
        {"id": "C5","x": 384.0,"y": 45.0, "demand": 30},     {"id": "C6","x": 302.0,"y": 145.0, "demand": 20},
        {"id": "C7","x": 129.0,"y": 221.0, "demand": 30},   {"id": "C8","x": 398.0,"y": 231.0, "demand": 30},
        {"id": "C9","x": 341.0,"y": 329.0, "demand": 20},    {"id": "C10","x": 537.0,"y": 365.0, "demand": 20}
    ]
preset_num_customers=5
preset_num_vehicles=3
preset_num_depots=1
preset_vehicle_capacity=100

preset_num_pairs=3
preset_pair_locations = [
        {"id": "Pair1","px": 103.0,"py": 120.0, "dx": 510.0,"dy": 150.0, "demand": 30},
        {"id": "Pair2","px": 171.0,"py": 317.0, "dx": 486.0,"dy": 283.0, "demand": 40},
        {"id": "Pair3","px": 384.0,"py": 45.0, "dx": 302.0,"dy": 145.0, "demand": 30},
        {"id": "Pair4","px": 129.0,"py": 221.0, "dx": 398.0,"dy": 231.0, "demand": 30},
        {"id": "Pair5","px": 341.0,"py": 329.0, "dx": 537.0,"dy": 365.0, "demand": 20},
    ]

def create_vrp_json_data(form_data):
    # --- 1. 입력 데이터 파싱 및 기본 유효성 검사 ---
    num_depots=1
    parsed_depot_location = {
        'x': float(form_data.get('depot_x', '0')),
        'y': float(form_data.get('depot_y', '0'))
    }

    num_customers = int(form_data.get('num_customers', preset_num_customers))
    parsed_customer_locations=[]
    for i in range(num_customers):
        demand_val_str = form_data.get(f'cust_{i}_demand', '0')
        parsed_customer_locations.append({
            'id': form_data.get(f'cust_{i}_id', f'C{i + 1}'),
            'x': float(form_data.get(f'cust_{i}_x')),  # 필수값으로 가정
            'y': float(form_data.get(f'cust_{i}_y')),  # 필수값으로 가정
            'demand': int(demand_val_str) if demand_val_str.isdigit() else 0
        })

    num_vehicles = int(form_data.get('num_vehicles',preset_num_vehicles))
    if num_vehicles <= 0:
        raise ValueError("차량 수는 1대 이상이어야 합니다.")

    vehicle_capacities = create_vehicle_capacities(int(form_data.get('vehicle_capacity', preset_vehicle_capacity)), num_vehicles)
    logger.debug(f"Depot: {parsed_depot_location}, Customers: {num_customers}, Vehicles: {num_vehicles}, Capacities: {vehicle_capacities[0]}")

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "depot_location": parsed_depot_location,
        "customer_locations": parsed_customer_locations,
        "num_vehicles": num_vehicles,
        "num_depots":num_depots,
        "vehicle_capacities":vehicle_capacities,
        # 추가적으로 저장하고 싶은 다른 form_data 항목들
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data


def create_pdp_json_data(form_data):
    # --- 1. 입력 데이터 파싱 및 기본 유효성 검사 ---
    num_depots = 1
    parsed_depot_location = {
        'x': float(form_data.get('depot_x', '0')),
        'y': float(form_data.get('depot_y', '0'))
    }

    num_vehicles = int(form_data.get('num_vehicles', preset_num_vehicles))
    if num_vehicles <= 0:
        raise ValueError("차량 수는 1대 이상이어야 합니다.")

    vehicle_capacities = create_vehicle_capacities(int(form_data.get('vehicle_capacity', preset_vehicle_capacity)), num_vehicles)

    pickup_delivery_pairs = []
    num_pairs = int(form_data.get('num_pairs', preset_num_pairs))
    for i in range(num_pairs):
        pickup_delivery_pairs.append({
            'id': form_data.get(f'pair_{i}_id'),
            'pickup': {'x': float(form_data.get(f'pair_{i}_px')), 'y': float(form_data.get(f'pair_{i}_py'))},
            'delivery': {'x': float(form_data.get(f'pair_{i}_dx')), 'y': float(form_data.get(f'pair_{i}_dy'))},
            'demand': int(form_data.get(f'pair_{i}_demand'))
        })

    logger.debug(
        f"Depot: {parsed_depot_location}, Pairs: {num_pairs}, Vehicles: {num_vehicles}, Capacities: {vehicle_capacities[0]}")

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "depot_location": parsed_depot_location,
        "pickup_delivery_pairs": pickup_delivery_pairs,
        "num_vehicles": num_vehicles,
        "num_depots": num_depots,
        "vehicle_capacities": vehicle_capacities,
        # 추가적으로 저장하고 싶은 다른 form_data 항목들
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data

def create_vehicle_capacities(vehicle_capacity, num_vehicles):
    vehicle_capacities=vehicle_capacity
    if not isinstance(vehicle_capacities, list):  # 단일 값으로 용량이 주어졌을 경우
        vehicle_capacities = [vehicle_capacities] * num_vehicles
    elif len(vehicle_capacities) != num_vehicles:
        logger.error("Number of vehicle capacities does not match number of vehicles. Using first capacity for all.")
        # 또는 오류 처리. 여기서는 첫 번째 용량을 모든 차량에 적용하거나, 평균값을 사용하는 등 정책 필요.
        # 가장 간단하게는 모든 차량 용량이 같다고 가정하고 하나의 값만 받도록 단순화할 수 있음.
        # 여기서는 첫 번째 용량을 사용.
        cap = vehicle_capacities[0] if vehicle_capacities else 100  # 기본값
        vehicle_capacities = [cap] * num_vehicles
    return vehicle_capacities

def save_vrp_json_data(input_data):
    num_depots = input_data.get('num_depots')
    num_vehicles = input_data.get('num_vehicles')
    filename_pattern=''
    if "PDP" == input_data.get('problem_type'):
        num_pairs = len(input_data.get('pickup_delivery_pairs'))
        filename_pattern = f"dep{num_depots}_pair{num_pairs}_veh{num_vehicles}"
    elif input_data.get('problem_type') in ("CVRP","VRP"):
        num_customers = len(input_data.get('customer_locations'))
        filename_pattern = f"dep{num_depots}_cus{num_customers}_veh{num_vehicles}"

    if "VRP" == input_data.get('problem_type'):
        dir ='routing_vrp_data'
    elif "CVRP" == input_data.get('problem_type'):
        dir ='routing_cvrp_data'
    elif "PDP" == input_data.get('problem_type'):
        dir ='routing_pdp_data'

    return save_json_data(input_data, dir, filename_pattern)



