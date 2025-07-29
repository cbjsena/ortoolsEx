

import os
import json
import logging

from solve import settings

logger = logging.getLogger(__name__)

def save_json_data(generated_data, model_data_type, filename_pattern):
    """
    입력 데이터를 JSON 파일로 저장합니다.
    성공 시 저장된 파일명을, 실패 시 None을 반환합니다.
    """
    data_dir_path_str = settings.DEMO_DIR_MAP[model_data_type]
    if not data_dir_path_str:
        logger.warning(f"{data_dir_path_str} not configured in settings. Input data will not be saved.")
        return None, "서버 저장 경로가 설정되지 않아 입력 데이터를 저장할 수 없습니다."

    try:
        data_dir = str(data_dir_path_str)
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir_path_str, f"{filename_pattern}.json")
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
        return get_save_info(filepath), None
    except IOError as e:
        logger.error(f"Failed to save input data to {filepath}: {e}", exc_info=True)
        return None, f"입력 데이터를 파일로 저장하는 데 실패했습니다: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
        return None, f"입력 데이터 저장 중 예상치 못한 오류 발생: {e}"

def get_save_info(filepath):
    return f"입력 데이터가 '{filepath}'으로 서버에 저장되었습니다."