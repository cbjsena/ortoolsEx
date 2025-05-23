import os
from pathlib import Path
import json
import logging
import datetime  # 파일명 생성 등에 사용 가능

logger = logging.getLogger(__name__)

TEST_URL = Path(__file__).resolve().parent.parent / 'test_data'
MATCH_CF_TFT_DATA_DIR = os.path.join(TEST_URL, 'match_cf_tft_test_data')
ALLOCATION_DATA_CENTER_DATA_DIR = os.path.join(TEST_URL, 'allocation_data_center_data')

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
    save_allocation_data_center_json_data()
    return generated_data


def save_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data,
                                          directory_setting_name):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    filename_prefix = 'datacenter_input'
    generated_data=create_allocation_data_center_json_data(global_constraints, server_types_data, service_demands_data)
    data_dir_path_str = ALLOCATION_DATA_CENTER_DATA_DIR
    if not data_dir_path_str:
        logger.warning(f"{directory_setting_name} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)

        num_server = len(generated_data.get('server_types'))
        num_service = len(generated_data.get('service_demands'))
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_svr{num_server}_svc{num_service}_{timestamp_str}.json"
        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Input data saved to: {filepath}")
        return filename, None # 성공 시 파일명과 None (오류 없음) 반환
    except IOError as e:
        logger.error(f"Failed to save input data to {filepath}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"

