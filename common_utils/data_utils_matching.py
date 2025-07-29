from common_utils.common_data_utils import save_json_data
import logging
import random
import datetime

logger = logging.getLogger(__name__)


preset_trans_assign_items=3
preset_trans_assign_drivers=["김기사", "이배달", "박운송", "최신속", "정안전"]
preset_trans_assign_zones = ["강남구", "서초구", "송파구", "마포구", "영등포구"]

preset_num_resources=7
preset_num_projects=3
preset_resources = [
        {'id': 'R1', 'name': '김개발', 'cost': '100', 'skills': 'Python,ML'},
        {'id': 'R2', 'name': '이엔지', 'cost': '120', 'skills': 'Java,SQL,Cloud'},
        {'id': 'R3', 'name': '박기획', 'cost': '90', 'skills': 'SQL,Tableau'},
        {'id': 'R4', 'name': '최신입', 'cost': '70', 'skills': 'Python'},
        {'id': 'R5', 'name': '정고급', 'cost': '150', 'skills': 'Cloud,Python,K8s'},
        {'id': 'R6', 'name': '한디자', 'cost': '105', 'skills': 'UI,AWS,UX,React'},
        {'id': 'R7', 'name': '백엔드', 'cost': '110', 'skills': 'Java,Spring,SQL'},
        {'id': 'R8', 'name': '프론트', 'cost': '90', 'skills': 'React,JavaScript'},
        {'id': 'R9', 'name': '데브옵', 'cost': '140', 'skills': 'K8s,AWS,Cloud'},
        {'id': 'R10', 'name': '데이터', 'cost': '130', 'skills': 'SQL,Python,Tableau'},
    ]
preset_projects = [
        {'id': 'P1', 'name': 'AI 모델 개발', 'required_skills': 'Python,ML,SQL'},
        {'id': 'P2', 'name': '데이터베이스 마이그레이션', 'required_skills': 'AWS,SQL,Cloud'},
        {'id': 'P3', 'name': '웹 서비스 프론트엔드', 'required_skills': 'React,JavaScript'},
        {'id': 'P4', 'name': '클라우드 인프라 구축', 'required_skills': 'AWS,K8s'},
        {'id': 'P5', 'name': 'BI 대시보드 제작', 'required_skills': 'SQL,Tableau'},
    ]

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


def create_cf_tft_matching_json_data(num_cf_panels, num_tft_panels, panel_rows, panel_cols, defect_rate):
    generated_cf_panels = create_panel_data("CF", num_cf_panels, panel_rows, panel_cols, defect_rate)
    generated_tft_panels = create_panel_data("TFT", num_tft_panels, panel_rows, panel_cols, defect_rate)

    generated_data = {
        "panel_dimensions": {"rows": panel_rows, "cols": panel_cols},
        "cf_panels": generated_cf_panels,
        "tft_panels": generated_tft_panels,
        "settings": {
            "num_cf_panels": num_cf_panels,
            "num_tft_panels": num_tft_panels,
            "defect_rate_percent": defect_rate,
            "panel_rows": panel_rows,
            "panel_cols": panel_cols,
        }
    }
    return generated_data


def create_transport_assignment_json_data(form_data, submitted_num_items):
    num_items = submitted_num_items
    cost_matrix = [[0] * num_items for _ in range(num_items)]
    driver_names = []
    zone_names = []

    for i in range(num_items):
        driver_names.append(form_data.get(f'driver_name_{i}', f'기사 {i + 1}'))
        zone_names.append(form_data.get(f'zone_name_{i}', f'구역 {i + 1}'))
        for j in range(num_items):
            cost_val = form_data.get(f'cost_{i}_{j}')
            if cost_val is None or not cost_val.isdigit():
                raise ValueError(f"'{driver_names[i]}' -> '{zone_names[j]}' 비용이 유효한 숫자가 아닙니다.")
            cost_matrix[i][j] = int(cost_val)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "driver_names": driver_names,
        "zone_names": zone_names,
        "cost_matrix": cost_matrix,
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data


def create_resource_skill_matching_json_data(form_data, num_resources, num_projects):
    resources_data = []
    for i in range(num_resources):
        skills_str = form_data.get(f'res_{i}_skills', '')
        resources_data.append({
            'id': form_data.get(f'res_{i}_id'),
            'name': form_data.get(f'res_{i}_name'),
            'cost': int(form_data.get(f'res_{i}_cost')),
            'skills': [s.strip() for s in skills_str.split(',') if s.strip()]  # 쉼표로 구분된 문자열을 리스트로
        })

    projects_data = []
    for i in range(num_projects):
        req_skills_str = form_data.get(f'proj_{i}_required_skills', '')
        projects_data.append({
            'id': form_data.get(f'proj_{i}_id'),
            'name': form_data.get(f'proj_{i}_name'),
            'required_skills': [s.strip() for s in req_skills_str.split(',') if s.strip()]
        })
    num_resources = len(resources_data)
    num_projects = len(projects_data)
    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_resources": num_resources,
        "num_projects": num_projects,
        "resources_data": resources_data,
        "projects_data": projects_data,
        "form_parameters": {
            key: value for key, value in form_data.items() if key not in ['csrfmiddlewaretoken']
        }
    }
    return input_data


def validate_required_skills(input_data):
    """
    각 프로젝트의 required_skills 중 resources_data의 skills에 없는 항목을 찾아 반환합니다.
    반환값: {스킬명: [포함하지 않은 프로젝트ID, ...], ...}
    """
    resources_data = input_data['resources_data']
    projects_data = input_data['projects_data']
    # 모든 리소스의 스킬을 집합으로 만듦
    all_skills = set()
    for res in resources_data:
        all_skills.update(res.get('skills', []))

    unmatched = {}
    for proj in projects_data:
        proj_id = proj.get('id')
        req_skills = set(proj.get('required_skills', []))
        missing = req_skills - all_skills
        for skill in missing:
            if skill not in unmatched:
                unmatched[skill] = []
            unmatched[skill].append(proj_id)

    # JSON을 key: value 형태의 HTML로 변환
    if isinstance(unmatched, dict):
        formatted_html = "<ul>"
        for k, v in unmatched.items():
            formatted_html += f"<li><strong>{k}</strong>: {v}</li>"
        formatted_html += "</ul>"
    elif isinstance(unmatched, list):
        formatted_html = "<ul>"
        for item in unmatched:
            if isinstance(item, dict):
                for k, v in item.items():
                    formatted_html += f"<li><strong>{k}</strong>: {v}</li>"
            else:
                formatted_html += f"<li>{item}</li>"
        formatted_html += "</ul>"
    else:
        formatted_html = str(unmatched)
    formatted_html = formatted_html.replace("'", "")
    return unmatched, formatted_html


def save_matching_assignment_json_data(input_data):
    problem_type = input_data.get('problem_type')
    dir = f'matching_{problem_type}_data'
    filename_pattern = ''
    if "transport assignment" == problem_type:
        num_driver = len(input_data.get('driver_names'))
        num_zone = len(input_data.get('zone_names'))
        filename_pattern = f"driver{num_driver}_zone{num_zone}"
    elif "resource skill" == problem_type:
        num_resources = input_data.get('num_resources')
        num_projects = input_data.get('num_projects')
        filename_pattern = f"resource{num_resources}_project{num_projects}"

    return save_json_data(input_data, dir, filename_pattern)

