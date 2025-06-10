import os
from pathlib import Path
import json
import datetime  # 파일명 생성 등에 사용 가능
import random
import default_data
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

TEST_URL = Path(__file__).resolve().parent.parent / 'test_data'
DATA_DIR_MAP ={'matching_cf_tft_data':TEST_URL/'match_cf_tft_data',
               'allocation_datacenter_data':TEST_URL/'allocation_datacenter_data',
               'allocation_budjet_data':TEST_URL/'allocation_budjet_data',
               'routing_vrp_data':TEST_URL/'routing_vrp_data',
               'routing_cvrp_data':TEST_URL/'routing_cvrp_data',
               'routing_pdp_data':TEST_URL/'routing_pdp_data'}

for demo_key, dir_path in DATA_DIR_MAP.items():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
def create_panel_data(panel_id_prefix, num_panels, rows, cols, rate):
    panels = []
    for i_panel in range(1, num_panels + 1):
        defect_map = []
        for r_idx in range(rows):
            row_map = []
            for c_idx in range(cols):
                if random.randint(1, 100) <= rate:
                    row_map.append(1)
                else:
                    row_map.append(0)
            defect_map.append(row_map)
        panels.append({
            "id": f"{panel_id_prefix}{i_panel}",
            "rows": rows,
            "cols": cols,
            "defect_map": defect_map
        })
    return panels

def create_matching_cf_tft_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate):

    generated_cf_panels = create_panel_data("CF", num_cf_panels, panel_rows, panel_cols, defect_rate)
    generated_tft_panels = create_panel_data("TFT", num_tft_panels, panel_rows, panel_cols, defect_rate)

    generated_data = {
        "panel_dimensions": {"rows": panel_rows, "cols": panel_cols},
        "cf_panels": generated_cf_panels,
        "tft_panels": generated_tft_panels,
        "num_cf_panels": num_cf_panels,
        "num_tft_panels": num_tft_panels,
        "defect_rate_percent": defect_rate,
        "panel_rows": panel_rows,
        "panel_cols": panel_cols,
    }

    return generated_data

def save_matching_cf_tft_json_data(json_data):
    num_cf_panels = json_data.get('num_cf_panels')
    num_tft_panels = json_data.get('num_tft_panels')
    panel_rows = json_data.get('panel_rows')
    panel_cols = json_data.get('panel_cols')
    defect_rate_percent = json_data.get('defect_rate_percent')
    filename_pattern = f"matching_cf{num_cf_panels}_tft{num_tft_panels}_row{panel_rows}_col{panel_cols}_rate{defect_rate_percent}"
    return save_json_data(json_data, 'matching_cf_tft_data', filename_pattern)

def create_allocation_budjet_json_data(total_budget, items_data):
    generated_data = {
        "model_info": {
            "problem_type": "BudjetAllocationPlanning",
            "num_items_data": len(items_data)
        },
        "total_budget": total_budget,
        "items_data": items_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return generated_data

def save_allocation_budjet_json_data(json_data):
    num_item = len(json_data.get('items_data'))
    filename_pattern = f"item{num_item}"
    return save_json_data(json_data, 'allocation_budjet_data', filename_pattern)

def create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data):
    generated_data = {
        "model_info": {
            "problem_type": "DataCenterCapacityPlanning",
            "num_server_types": 2,
            "num_services": 2
        },
        "global_constraints": global_constraints,
        "server_types": server_types_data,
        "service_demands": service_demands_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return generated_data

def save_allocation_data_center_json_data(json_data):
    num_server = len(json_data.get('server_types'))
    num_service = len(json_data.get('service_demands'))
    filename_pattern = f"svr{num_server}_svc{num_service}"
    return save_json_data(json_data, 'allocation_datacenter_data', filename_pattern)


def save_vrp_json_data(input_data):
    num_depots = input_data.get('num_depots')
    num_vehicles = input_data.get('num_vehicles')
    num_customers = len(input_data.get('customer_locations'))
    filename_pattern = f"dep{num_depots}_cus{num_customers}_veh{num_vehicles}"
    return save_json_data(input_data, 'routing_vrp_data', filename_pattern)

def save_json_data(generated_data, model_data_type, filename_pattern):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    data_dir_path_str = DATA_DIR_MAP[model_data_type]
    if not data_dir_path_str:
        logger.warning(f"{data_dir_path_str} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)
        seq = 0
        while True:
            if seq == 0:
                potential_filename = f"{filename_pattern}.json"
            else:
                potential_filename = f"{filename_pattern}_seq{seq}.json"

            filepath = os.path.join(data_dir_path_str, potential_filename)
            if not os.path.exists(filepath):
                loaded_filename = potential_filename  # 저장될 (또는 사용될) 파일명
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(generated_data, f, indent=2)
                logger.info(f"Generated data saved to: {filepath}")
                # 파일 목록을 즉시 업데이트하기 위해 다시 로드 (선택 사항)
                files = [f for f in os.listdir(data_dir_path_str) if
                         f.endswith('.json') and f.startswith('test_cf')]
                break
            seq += 1
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Input data saved to: {filepath}")
    except IOError as e:
        logger.error(f"Failed to save input data to {potential_filename}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"


if __name__ == "__main__":
    data_type ='routing_vrp_data'
    logger.info(f'data_type:{data_type}')
    if data_type=='matching_cf_tft_data':
        num_cf_panels = 5
        num_tft_panels = 5
        panel_rows = 3
        panel_cols = 3
        defect_rate = 10
        json_data = create_matching_cf_tft_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate)
        save_matching_cf_tft_json_data(json_data)
    elif data_type=='allocation_datacenter_data':
        global_constraints = default_data.global_constraints
        server_types_data = default_data.datacenter_servers_preset
        service_demands_data = default_data.datacenter_services_preset
        json_data = create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data)
        save_allocation_data_center_json_data(json_data)
    elif data_type=='allocation_budjet_data':
        total_budget=default_data.budjet_total_budget_preset
        budjet_items = default_data.budjet_items_preset
        json_data = create_allocation_budjet_json_data(total_budget, budjet_items)
        save_allocation_budjet_json_data(json_data)
    elif data_type=='routing_vrp_data':
        depot_location= default_data.depot_location
        customer_locations= default_data.customer_locations
        num_vehicles= default_data.num_vehicles
        num_depots= default_data.num_depots
        json_data = create_vrp_json_data(depot_location, customer_locations, num_vehicles, num_depots)
        save_vrp_json_data(json_data)