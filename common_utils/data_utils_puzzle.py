from collections import Counter

from common_utils.common_data_utils import save_json_data
import logging
import math
from settings import *

logger = logging.getLogger(__name__)

preset_diet_nutrient_number = 2
preset_diet_food_number = 4
preset_diet_nutrients_data = [
    {'name': '칼로리(kcal)', 'min': '100', 'max': '2500'},
    {'name': '단백질(g)', 'min': '20', 'max': '100'},
    {'name': '지방(g)', 'min': '20', 'max': '70'},
    {'name': '탄수화물(g)', 'min': '20', 'max': '350'},
    {'name': '나트륨(mg)', 'min': '0', 'max': '2000'}
]
preset_diet_foods_data = [
    {'name': '우유(100ml)', 'cost': '150', 'min_intake': '2', 'max_intake': '10',
     'nutrients': ['60', '3.2', '3.5', '4.8', '50']},
    {'name': '계란(1개)', 'cost': '300', 'min_intake': '0', 'max_intake': '5', 'nutrients': ['80', '6', '6', '0.5', '65']},
    {'name': '식빵(1장)', 'cost': '200', 'min_intake': '0', 'max_intake': '10',
     'nutrients': ['70', '2.5', '1', '13', '150']},
    {'name': '닭가슴살(100g)', 'cost': '1500', 'min_intake': '2', 'max_intake': '5',
     'nutrients': ['110', '23', '1.5', '0', '70']},
    {'name': '바나나(1개)', 'cost': '500', 'min_intake': '0', 'max_intake': '4',
     'nutrients': ['90', '1', '0.3', '23', '1']},
    {'name': '아몬드(10g)', 'cost': '200', 'min_intake': '0', 'max_intake': '5', 'nutrients': ['60', '2', '5', '2', '1']},
    {'name': '두부(100g)', 'cost': '500', 'min_intake': '0', 'max_intake': '3',
     'nutrients': ['80', '8', '4.5', '2', '5']},
    {'name': '현미밥(100g)', 'cost': '400', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['130', '2.5', '1', '28', '3']},
    {'name': '시금치(100g)', 'cost': '300', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['25', '2.9', '0.4', '3.6', '80']},
    {'name': '올리브 오일(10g)', 'cost': '100', 'min_intake': '0', 'max_intake': '5',
     'nutrients': ['90', '0', '10', '0', '0']},
]
preset_sport_schedule_max_consecutive=3
preset_sport_schedule_objective_choice='minimize_travel'
preset_sport_schedule_type='double'
preset_sport_schedule_type_options_list = [
    {'value': 'single', 'name': '싱글 라운드 로빈 (팀당 1경기)'},
    {'value': 'double', 'name': '더블 라운드 로빈 (팀당 2경기)'},
]
preset_sport_schedule_objective_list = [
    {'value': 'minimize_travel', 'name': '총 이동 거리 최소화'},
    {'value': 'fairness', 'name': '연속 홈/원정 최소화'},
    {'value': 'distance_gap', 'name': '팀간 이동거리 차이 최소화'},
]
preset_sport_schedule_solver_type_options_list = [
    {'value': SOLVER_ORTOOLS, 'name': 'OR-Tools'},
    {'value': SOLVER_GUROBI, 'name': 'Gurobi'},
]
preset_sport_schedule_num_teams = 4
preset_sport_schedule_team_list = [
    "한화", "LG", "롯데", "KIA", "삼성",
    "KT", "SSG", "NC", "두산", "키움"
]
# 각 팀의 연고지 도시 (거리 계산용, 순서는 위와 동일)
preset_sport_schedule_cities = ['대전', '서울', '부산', '광주', '대구', '수원', '인천', '창원', '서울', '서울']
# 도시 간 대략적인 거리 행렬 (km) - 예시 데이터
preset_sport_schedule_distance_matrix_km = [
    # 대전, 서울, 부산, 광주, 대구, 수원, 인천, 창원
    [  0, 160, 200, 140, 100, 130, 200, 150], # 대전
    [160,   0, 325, 270, 240,  30,  30, 290], # 서울
    [200, 325,   0, 200,  95, 300, 350,  40], # 부산
    [140, 270, 200,   0, 150, 240, 300, 160], # 광주
    [100, 240,  95, 150,   0, 210, 265,  60], # 대구
    [130,  30, 300, 240, 210,   0,  40, 260], # 수원
    [200,  30, 350, 300, 265,  40,   0, 320], # 인천
    [150, 290,  40, 160,  60, 260, 320,   0], # 창원
]

preset_sport_schedule_dist_map_10 = [[0] * 10 for _ in range(10)]
preset_sport_schedule_city_map = [0, 1, 2, 3, 4, 5, 6, 7, 1, 1]  # 10개 팀의 도시 인덱스
for i in range(10):
    for j in range(10):
        city_i = preset_sport_schedule_city_map[i]
        city_j = preset_sport_schedule_city_map[j]
        if city_i < 8 and city_j < 8:
            preset_sport_schedule_dist_map_10[i][j] = preset_sport_schedule_distance_matrix_km[city_i][city_j]

preset_tsp_all_cities= [
    {'name': '서울', 'lat': 37.5665, 'lon': 126.9780}, {'name': '부산', 'lat': 35.1796, 'lon': 129.0756},
    {'name': '대구', 'lat': 35.8714, 'lon': 128.6014}, {'name': '광주', 'lat': 35.1595, 'lon': 126.8526},
    {'name': '창원', 'lat': 35.2283, 'lon': 128.6811}, {'name': '포항', 'lat': 36.0320, 'lon': 129.3648},
    {'name': '강릉', 'lat': 37.7519, 'lon': 128.8761}, {'name': '춘천', 'lat': 37.8813, 'lon': 127.7298},
    {'name': '청주', 'lat': 36.6424, 'lon': 127.4890}, {'name': '전주', 'lat': 35.8242, 'lon': 127.1480},
    {'name': '안동', 'lat': 36.5681, 'lon': 128.7294}, {'name': '원주', 'lat': 37.3424, 'lon': 127.9201},
    {'name': '제천', 'lat': 37.1325, 'lon': 128.1921}, {'name': '여주', 'lat': 37.2982, 'lon': 127.6373},
    {'name': '상주', 'lat': 36.4107, 'lon': 128.1592}, {'name': '남원', 'lat': 35.4082, 'lon': 127.3941},
    {'name': '당진', 'lat': 36.8906, 'lon': 126.6270}, {'name': '군산', 'lat': 35.9678, 'lon': 126.7369},
    {'name': '목포', 'lat': 34.8118, 'lon': 126.3922}, {'name': '보령', 'lat': 36.3364, 'lon': 126.5133}
]

# preset_tsp_cities=['서울', '청주', '제천', '원주', '남원', '안동', '청주', '대구', '강릉', '군산', '목포']
preset_tsp_cities=['서울', '당진', '남원', '창원', '춘천']
# 순서: 서울,부산,대구,광주,창원,포항,강릉,춘천,청주,전주,안동,원주,제천,여주,상주,남원,당진,군산,목포,보령
preset_tsp_distance_matrix = [
    [0, 400, 295, 330, 390, 330, 215, 105, 125, 230, 230, 130, 160, 80, 200, 280, 120, 190, 330, 160], # 서울
    [400, 0, 95, 100, 30, 90, 340, 450, 300, 145, 185, 370, 330, 415, 140, 80, 355, 290, 160, 300], # 부산
    [295, 95, 0, 195, 85, 65, 270, 360, 180, 130, 95, 280, 200, 280, 60, 115, 270, 180, 170, 180], # 대구
    [330, 100, 195, 0, 90, 260, 480, 410, 190, 40, 260, 340, 310, 350, 170, 50, 200, 145, 95, 150], # 광주
    [390, 30, 85, 90, 0, 80, 360, 440, 280, 130, 175, 380, 320, 405, 130, 70, 320, 260, 150, 270], # 창원
    [330, 90, 65, 260, 80, 0, 225, 350, 210, 220, 80, 270, 215, 315, 85, 210, 300, 240, 260, 275], # 포항
    [215, 340, 270, 480, 360, 225, 0, 150, 250, 420, 190, 90, 110, 150, 200, 430, 230, 310, 490, 290], # 강릉
    [105, 450, 360, 410, 440, 350, 150, 0, 200, 320, 205, 55, 60, 75, 230, 370, 150, 230, 420, 240], # 춘천
    [125, 300, 180, 190, 280, 210, 250, 200, 0, 125, 160, 130, 100, 90, 95, 150, 90, 130, 220, 110], # 청주
    [230, 145, 130, 40, 130, 220, 420, 320, 125, 0, 200, 250, 220, 255, 130, 60, 150, 80, 90, 110], # 전주
    [230, 185, 95, 260, 175, 80, 190, 205, 160, 200, 0, 130, 100, 180, 50, 190, 230, 150, 240, 170], # 안동
    [130, 370, 280, 340, 380, 270, 90, 55, 130, 250, 130, 0, 40, 70, 150, 280, 150, 190, 350, 200], # 원주
    [160, 330, 200, 310, 320, 215, 110, 60, 100, 220, 100, 40, 0, 110, 115, 250, 130, 160, 320, 170], # 제천
    [80, 415, 280, 350, 405, 315, 150, 75, 90, 255, 180, 70, 110, 0, 155, 290, 80, 160, 360, 140], # 여주
    [200, 140, 60, 170, 130, 85, 200, 230, 95, 130, 50, 150, 115, 155, 0, 110, 175, 105, 160, 120], # 상주
    [280, 80, 115, 50, 70, 210, 430, 370, 150, 60, 190, 280, 250, 290, 110, 0, 230, 140, 60, 190], # 남원
    [120, 355, 270, 200, 320, 300, 230, 150, 90, 150, 230, 150, 130, 80, 175, 230, 0, 70, 205, 65], # 당진
    [190, 290, 180, 145, 260, 240, 310, 230, 130, 80, 150, 190, 160, 160, 105, 140, 70, 0, 115, 45], # 군산
    [330, 160, 170, 95, 150, 260, 490, 420, 220, 90, 240, 350, 320, 360, 160, 60, 205, 115, 0, 125], # 목포
    [160, 300, 180, 150, 270, 275, 290, 240, 110, 110, 170, 200, 170, 140, 120, 190, 65, 45, 125, 0], # 보령
]
preset_sudoku_size_options = [9, 16, 25]
preset_sudoku_examples = {
    16: [
        [ 1,  2,  3,  4,   5,  6,  7,  8,   9, 10, 11, 12,  13, 14, 15, 16],
        [ 5,  6,  7,  8,   9, 10, 11, 12,  13, 14, 15, 16,   1,  2,  3,  4],
        [ 9, 10, 11, 12,  13, 14, 15, 16,   1,  2,  3,  4,    5,  6,  7,  8],
        [13, 14, 15, 16,   1,  2,  3,  4,    5,  6,  7,  8,    9, 10, 11, 12],

        [ 2,  1,  4,  3,   6,  5,  8,  7,  10,  9, 12, 11,  14, 13, 16, 15],
        [ 6,  5,  8,  7,  10,  9, 12, 11,  14, 13, 16, 15,   2,  1,  4,  3],
        [10,  9, 12, 11,  14, 13, 16, 15,   2,  1,  4,  3,    6,  5,  8,  7],
        [14, 13, 16, 15,   2,  1,  4,  3,    6,  5,  8,  7,  10,  9, 12, 11],

        [ 3,  4,  1,  2,   7,  8,  5,  6,  11, 12,  9, 10,  15, 16, 13, 14],
        [ 7,  8,  5,  6,  11, 12,  9, 10,  15, 16, 13, 14,   3,  4,  1,  2],
        [11, 12,  9, 10,  15, 16, 13, 14,   3,  4,  1,  2,    7,  8,  5,  6],
        [15, 16, 13, 14,   3,  4,  1,  2,    7,  8,  5,  6,  11, 12,  9, 10],

        [ 4,  3,  2,  1,   8,  7,  6,  5,  12, 11, 10,  9,  16, 15, 14, 13],
        [ 8,  7,  6,  5,  12, 11, 10,  9,  16, 15, 14, 13,   4,  3,  2,  1],
        [12, 11, 10,  9,  16, 15, 14, 13,   4,  3,  2,  1,    8,  7,  6,  5],
        [16, 15, 14, 13,   4,  3,  2,  1,    8,  7,  6,  5,  12, 11, 10,  9]
    ],
    25: [
        [1, 16, 17, 15, 18, 7, 13, 2, 19, 25, 3, 4, 22, 6, 14, 12, 21, 24, 8, 10, 11, 9, 20, 23, 5],
        [8, 7, 13, 23, 25, 1, 5, 24, 21, 12, 18, 10, 20, 19, 17, 6, 15, 14, 11, 9, 2, 22, 3, 4, 16],
        [22, 5, 4, 20, 24, 9, 14, 15, 11, 8, 2, 12, 25, 1, 23, 13, 3, 17, 19, 16, 6, 21, 7, 10, 18],
        [2, 14, 11, 19, 6, 3, 17, 4, 10, 23, 15, 9, 21, 16, 24, 7, 22, 5, 18, 20, 13, 1, 12, 25, 8],
        [3, 9, 10, 21, 12, 18, 16, 22, 6, 20, 7, 5, 11, 8, 13, 1, 25, 2, 4, 23, 14, 17, 24, 19, 15],
        [10, 3, 21, 14, 4, 15, 2, 16, 12, 1, 9, 6, 8, 25, 5, 18, 11, 19, 24, 7, 23, 13, 22, 17, 20],
        [15, 20, 16, 7, 11, 4, 3, 6, 5, 22, 10, 21, 13, 14, 12, 23, 2, 1, 17, 25, 8, 19, 18, 9, 24],
        [17, 2, 5, 18, 8, 11, 9, 10, 23, 19, 22, 3, 1, 24, 7, 14, 6, 21, 20, 13, 16, 25, 15, 12, 4],
        [9, 6, 22, 25, 13, 8, 7, 17, 20, 24, 4, 2, 18, 23, 19, 3, 16, 12, 15, 5, 21, 14, 10, 11, 1],
        [12, 23, 1, 24, 19, 14, 21, 25, 18, 13, 11, 17, 15, 20, 16, 8, 4, 10, 9, 22, 7, 6, 5, 3, 2],
        [7, 10, 14, 8, 9, 16, 18, 11, 25, 21, 1, 24, 4, 13, 2, 5, 17, 15, 6, 12, 22, 3, 19, 20, 23],
        [16, 22, 12, 11, 1, 13, 20, 7, 2, 15, 17, 25, 5, 9, 21, 19, 23, 18, 14, 3, 24, 10, 4, 8, 6],
        [19, 21, 18, 17, 2, 22, 4, 5, 24, 6, 8, 7, 23, 10, 3, 16, 9, 20, 25, 1, 12, 15, 14, 13, 11],
        [5, 24, 20, 3, 23, 10, 1, 12, 8, 17, 14, 19, 6, 15, 22, 2, 7, 11, 13, 4, 25, 16, 9, 18, 21],
        [25, 4, 6, 13, 15, 19, 23, 3, 9, 14, 16, 11, 12, 18, 20, 21, 24, 22, 10, 8, 5, 2, 1, 7, 17],
        [13, 11, 8, 10 ,14, 23, 12, 21, 15, 7, 5, 20, 17, 3, 6, 4, 19, 16, 2, 18, 1, 24, 25, 22, 9],
        [4, 1, 15, 2, 3, 6, 11, 9, 13, 5, 25, 8, 7, 22, 10, 24, 12, 23, 21, 14, 20, 18, 17, 16, 19],
        [6, 19, 7, 5, 16, 2, 8, 1, 4, 10, 23, 18, 24, 21, 9, 25, 20, 3, 22, 17, 15, 12, 11, 14, 13],
        [20, 18, 23, 12, 17, 24, 25, 14, 22, 3, 19, 16, 2, 4, 1, 15, 13, 9, 5, 11, 10, 8, 6, 21, 7],
        [24, 25, 9, 22, 21, 20, 19, 18, 17, 16, 13, 15, 14, 12, 11, 10, 8, 7, 1, 6, 4, 5, 23, 2, 3],
        [21, 13, 2, 6, 22, 12, 10, 8, 1, 18, 24, 23, 9, 7, 25, 11, 5, 4, 3, 19, 17, 20, 16, 15, 14],
        [11, 15, 3, 4, 10, 5, 24, 13, 7, 9, 6, 14, 16, 17, 8, 20, 18, 25, 12, 21, 19, 23, 2, 1, 22],
        [14, 12, 25, 9, 7, 21, 6, 23, 3, 4, 20, 22, 19, 5, 15, 17, 1, 8, 16, 2, 18, 11, 13, 24, 10],
        [18, 17, 19, 16, 5, 25, 15, 20, 14, 11, 21, 1, 3, 2, 4, 22, 10, 13, 23, 24, 9, 7, 8, 6, 12],
        [23, 8, 24, 1, 20, 17, 22, 19, 16, 2, 12, 13, 10, 11, 18, 9, 14, 6, 7, 15, 3, 4, 21, 5, 25]
    ]
}

preset_sudoku_difficulty_options = [
    {'value': 'easy', 'name': '쉬움 (Easy)'},
    {'value': 'medium', 'name': '중간 (Medium)'},
    {'value': 'hard', 'name': '어려움 (Hard)'},
]

preset_sudoku_difficulty ='easy'
preset_sudoku_size ='9'

def create_diet_json_data(form_data):
    logger.debug("Creating and validating Diet Problem input data from form.")
    num_foods = int(form_data.get('num_foods', 0))
    num_nutrients = int(form_data.get('num_nutrients', 0))

    nutrient_reqs=[]
    # 1. 영양소 요구사항 파싱
    for i in range(num_nutrients):
        try:
            nutrient_reqs.append({
                'name': form_data.get(f'nutrient_{i}_name'),
                'min': float(form_data.get(f'nutrient_{i}_min')),
                'max': float(form_data.get(f'nutrient_{i}_max'))
            })
        except (ValueError, TypeError):
            raise ValueError(f"영양소 {i + 1}의 최소/최대 요구량 값이 올바른 숫자가 아닙니다.")

    # 2. 식품 데이터 파싱
    food_items=[]
    for i in range(num_foods):
        try:
            food_item = {
                'name': form_data.get(f'food_{i}_name'),
                'cost': float(form_data.get(f'food_{i}_cost')),
                'min_intake': float(form_data.get(f'food_{i}_min_intake', 0)),
                'max_intake': float(form_data.get(f'food_{i}_max_intake', 10000)),  # 충분히 큰 값
                'nutrients': []
            }
            for j in range(num_nutrients):
                nutrient_val = float(form_data.get(f'nutrient_val_{i}_{j}'))
                food_item['nutrients'].append(nutrient_val)
            food_items.append(food_item)
        except (ValueError, TypeError):
            raise ValueError(f"식품 '{form_data.get(f'food_{i}_name')}'의 입력값이 올바르지 않습니다.")

    input_data = {
        "problem_type": "diet_problem",
        "num_foods": num_foods,
        "num_nutrients": num_nutrients,
        "food_items": food_items,
        "nutrient_reqs": nutrient_reqs
    }

    logger.info("End Diet Problem Demo Input data processing.")
    return input_data


def calculate_manual_diet(input_data, manual_intakes):
    """수동 입력 식단의 비용과 영양 정보를 계산합니다."""
    logger.info("Calculating manual diet plan.")

    foods = input_data['food_items']
    nutrients = input_data['nutrient_reqs']
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    manual_results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    total_cost = 0.0

    for i in range(num_foods):
        intake = manual_intakes.get(f'food_{i}_intake', 0)
        if intake > 0:
            food_item = foods[i]
            cost = intake * food_item['cost']
            total_cost += cost
            manual_results['diet_plan'].append({
                'name': food_item['name'],
                'intake': intake,
                'cost': round(cost, 2)
            })

    manual_results['total_cost'] = round(total_cost, 2)

    for i in range(num_nutrients):
        total_nutrient_intake = 0
        for j in range(num_foods):
            intake = manual_intakes.get(f'food_{j}_intake', 0)
            total_nutrient_intake += foods[j]['nutrients'][i] * intake

        reqs = nutrients[i]
        status = "OK"
        if total_nutrient_intake < reqs['min']:
            status = "Minimum not met"
        elif total_nutrient_intake > reqs['max']:
            status = "Maximum exceeded"

        manual_results['nutrient_summary'].append({
            'name': reqs['name'],
            'min_req': reqs['min'],
            'max_req': reqs['max'],
            'actual_intake': round(total_nutrient_intake, 2),
            'status': status
        })

    return manual_results


def create_sports_scheduling_json_data(form_data, num_teams, objective, schedule_type):
    # 팀 이름 리스트 생성
    teams_list = [form_data.get(f'team_{i}_name') for i in range(num_teams)]
    # 선택된 팀에 해당하는 거리 행렬 슬라이싱
    selected_dist_matrix = [[0] * num_teams for _ in range(num_teams)]
    for i in range(num_teams):
        for j in range(num_teams):
            # default_teams_pool에서의 원래 인덱스를 찾아야 함
            original_idx_i = preset_sport_schedule_team_list.index(teams_list[i]) if teams_list[i] in preset_sport_schedule_team_list else -1
            original_idx_j = preset_sport_schedule_team_list.index(teams_list[j]) if teams_list[j] in preset_sport_schedule_team_list else -1
            if original_idx_i != -1 and original_idx_j != -1:
                selected_dist_matrix[i][j] = preset_sport_schedule_dist_map_10[original_idx_i][original_idx_j]

    input_data = {
        'problem_type': 'sports_scheduling',
        'teams': teams_list,
        'num_teams': num_teams,
        'distance_matrix': selected_dist_matrix,  # 거리 행렬 추가
        'objective_choice': objective,
        'schedule_type': schedule_type,
        'max_consecutive': int(form_data.get('max_consecutive'))
    }

    return input_data


def create_tsp_json_data(selected_cities_data):
    # 선택된 도시의 전체 데이터(이름, 좌표)를 추출
    all_city_names = [city['name'] for city in preset_tsp_all_cities]
    selected_indices = [all_city_names.index(city['name']) for city in selected_cities_data]
    sub_matrix = [[preset_tsp_distance_matrix[i][j] for j in selected_indices] for i in selected_indices]
    num_cities = len(selected_cities_data)
    input_data = {
        'problem_type': 'tsp',
        'sub_matrix': sub_matrix,
        'num_cities': num_cities
    }

    return  input_data


def create_sudoku_json_data(form_data):
    input_grid = []
    num_size = int(form_data.get('size', preset_sudoku_size))
    for i in range(num_size):
        row = []
        for j in range(num_size):
            cell_value = form_data.get(f'cell_{i}_{j}', '0')
            row.append(int(cell_value) if cell_value.isdigit() else 0)
        input_grid.append(row)

    num_size = len(input_grid)  # 그리드 크기를 입력에서 직접 가져옴
    subgrid_size = int(math.sqrt(num_size))
    # N이 완전 제곱수가 아니면 에러 처리
    if subgrid_size * subgrid_size != num_size:
        raise ValueError(f"{num_size}x{num_size}는 유효한 스도쿠 크기가 아닙니다.")

    # 사용자 입력 유효성 검사
    validation_error_msg = validate_sudoku_input(input_grid)

    input_data = {
        'problem_type': 'sudoku',
        'input_grid': input_grid,
        'difficulty': form_data.get('difficulty'),
        'num_size': num_size,
        'subgrid_size': subgrid_size
    }

    return validation_error_msg, input_data


def validate_sudoku_input(board):
    """
    사용자가 입력한 스도쿠 그리드의 유효성을 검사하고,
    오류가 있을 경우 HTML 형식의 문자열로 상세 내용을 반환합니다.
    """
    N = len(board)
    subgrid_size = int(math.sqrt(N))
    error_messages = []

    def find_duplicates(name, index, values):
        # 0을 제외한 값들만 필터링
        filtered = [v for v in values if v != 0]
        # 중복된 숫자 찾기
        counter = Counter(filtered)
        duplicates = sorted([num for num, count in counter.items() if count > 1])
        if duplicates:
            # 오류 메시지를 리스트에 추가
            msg = (
                f"<b>❌ {name} {index}</b>에 중복된 숫자가 있습니다.<br>"
                f"&nbsp;&nbsp;▶ 중복된 숫자: {duplicates}"
            )
            error_messages.append(msg)

    # 1. 행 검사
    for i, row in enumerate(board):
        find_duplicates("행", i + 1, row)

    # 2. 열 검사
    for j in range(N):
        col = [board[i][j] for i in range(N)]
        find_duplicates("열", j + 1, col)

    # 3. 서브그리드(박스) 검사
    for box_row in range(0, N, subgrid_size):
        for box_col in range(0, N, subgrid_size):
            square = [board[box_row + i][box_col + j] for i in range(subgrid_size) for j in range(subgrid_size)]
            box_label = f"(행 {box_row + 1}~{box_row + subgrid_size}, 열 {box_col + 1}~{box_col + subgrid_size})"
            find_duplicates("박스", box_label, square)

    if error_messages:
        # 오류가 하나라도 있으면, 모든 오류 메시지를 합쳐서 반환
        header = "입력한 퍼즐에 다음과 같은 오류가 있어 해결할 수 없습니다:<br><br>"
        return header + "<br>".join(error_messages)

    # 오류가 없으면 유효함
    return None

def save_puzzle_json_data(input_data):
    problem_type = input_data.get('problem_type')
    dir = f'puzzles_{problem_type}_data'
    filename_pattern = ''

    if "diet_problem" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"
    elif "sports_scheduling" == problem_type:
        num_teams = input_data.get('num_teams')
        objective_choice = input_data.get('objective_choice')
        schedule_type = input_data.get('schedule_type')
        filename_pattern = f"{objective_choice}_{schedule_type}_team{num_teams}"
    elif "nurse_rostering" == problem_type:
        num_foods = input_data.get('num_foods')
        num_nutrients = input_data.get('num_nutrients')
        filename_pattern = f"food{num_foods}_nutrient{num_nutrients}"
    elif "tsp" == problem_type:
        num_cities = input_data.get('num_cities')
        filename_pattern = f"{num_cities}_cities"
    elif "sudoku" == problem_type:
        num_size = input_data.get('num_size')
        difficulty = input_data.get('difficulty')
        filename_pattern = f"{num_size}_{difficulty}"

    return save_json_data(input_data, dir, filename_pattern)