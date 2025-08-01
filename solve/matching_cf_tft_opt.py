# OptDemo/matching_app/views.py

import json
from ortools.linear_solver import pywraplp  # OR-Tools MIP solve
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

with open('../test_data/match_cf_tft_data/cf100_tft100_row4_col4_rate10.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
# test_data = json.loads("testcase/matcing_cf_tft_test1.json")
cf_panels = test_data.get('cf_panels')
tft_panels = test_data.get('tft_panels')

# --- run_matching_algorithm 함수 정의 ---
def run_optimization(cf_panels, tft_panels):
    num_cf = len(cf_panels)
    num_tft = len(tft_panels)

    if num_cf == 0 or num_tft == 0:
        return [], 0, "오류: CF 또는 TFT 패널 데이터가 없습니다."

    # --- 1. 수율 매트릭스 (C_ij) 계산 ---
    # C_ij = CF 패널 i와 TFT 패널 j를 매칭했을 때의 양품 셀 개수
    # 결함맵: 0 = 양품, 1 = 결함
    yield_matrix = [[0] * num_tft for _ in range(num_cf)]

    for i in range(num_cf):
        cf_panel = cf_panels[i]
        cf_map = cf_panel.get('defect_map', [])
        # 각 패널의 row, col 정보가 다를 수 있으므로 패널별로 가져옴
        cf_rows = cf_panel.get('rows', 0)
        cf_cols = cf_panel.get('cols', 0)

        if not cf_map or cf_rows == 0 or cf_cols == 0:
            # 해당 CF 패널 데이터가 유효하지 않으면 이 CF 패널에 대한 모든 매칭 수율은 0으로 처리
            # 또는 오류를 발생시킬 수 있습니다.
            for j in range(num_tft):
                yield_matrix[i][j] = -1  # 매칭 불가능 표시 (매우 낮은 값)
            continue

        for j in range(num_tft):
            tft_panel = tft_panels[j]
            tft_map = tft_panel.get('defect_map', [])
            tft_rows = tft_panel.get('rows', 0)
            tft_cols = tft_panel.get('cols', 0)

            if not tft_map or tft_rows == 0 or tft_cols == 0:
                yield_matrix[i][j] = -1
                continue

            # 두 패널의 크기가 다르면 매칭 불가능 (수율 0 또는 매우 낮은 값)
            if cf_rows != tft_rows or cf_cols != tft_cols:
                yield_matrix[i][j] = -1  # 매칭 불가능 (매우 낮은 값으로 설정하여 선택되지 않도록)
                continue

            current_yield = 0
            for r in range(cf_rows):
                for c in range(cf_cols):
                    # defect_map의 각 행이 cf_cols 길이를 가지고 있는지, r, c가 범위 내인지 확인
                    if r < len(cf_map) and c < len(cf_map[r]) and \
                            r < len(tft_map) and c < len(tft_map[r]):
                        is_cf_cell_good = (cf_map[r][c] == 0)
                        is_tft_cell_good = (tft_map[r][c] == 0)
                        if is_cf_cell_good and is_tft_cell_good:
                            current_yield += 1
                    else:
                        # 결함맵 데이터 구조 오류 처리
                        yield_matrix[i][j] = -1  # 이 쌍은 매칭 불가능으로 처리
                        current_yield = -1  # 루프 탈출 또는 플래그 설정용
                        break
                if current_yield == -1:
                    break

            if current_yield != -1:
                yield_matrix[i][j] = current_yield

    # --- 2. OR-Tools MIP 모델 구성 ---
    # SCIP, CBC, GLPK, Gurobi 등 다양한 솔버 사용 가능 (설치 필요)
    # Create the mip solve with the SCIP backend.
    try:
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # SCIP이 없으면 CBC 시도
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                return [], 0, "오류: MIP 솔버(SCIP 또는 CBC)를 생성할 수 없습니다. OR-Tools 설치를 확인하세요."
    except Exception as e:
        return [], 0, f"오류: 솔버 생성 중 예외 발생 - {str(e)}"

    # --- 3. 변수 생성 (X_ij) ---
    # x[i][j]는 CF_i와 TFT_j가 매칭되면 1, 아니면 0
    x = {}
    for i in range(num_cf):
        for j in range(num_tft):
            if yield_matrix[i][j] >= 0:  # 유효한 매칭 쌍에 대해서만 변수 생성
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # --- 4. 제약 조건 설정 ---
    # 각 CF 패널은 최대 하나의 TFT 패널과 매칭
    for i in range(num_cf):
        solver.Add(sum(x[i, j] for j in range(num_tft) if (i, j) in x) <= 1)

    # 각 TFT 패널은 최대 하나의 CF 패널과 매칭
    for j in range(num_tft):
        solver.Add(sum(x[i, j] for i in range(num_cf) if (i, j) in x) <= 1)

    # --- 5. 목표 함수 설정 ---
    objective = solver.Objective()
    for i in range(num_cf):
        for j in range(num_tft):
            if (i, j) in x:  # 유효한 변수에 대해서만 계수 설정
                objective.SetCoefficient(x[i, j], yield_matrix[i][j])
    objective.SetMaximization()

    # --- 6. 문제 해결 ---
    status = solver.Solve()

    # --- 7. 결과 추출 ---
    matched_pairs_info = []
    total_yield_val = 0
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL:
        total_yield_val = solver.Objective().Value()
        for i in range(num_cf):
            for j in range(num_tft):
                if (i, j) in x and x[i, j].solution_value() > 0.5:  # X_ij가 1인 경우
                    matched_pairs_info.append({
                        'cf': cf_panels[i],
                        'tft': tft_panels[j],
                        'cf_id': cf_panels[i].get('id', f'CF{i + 1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j + 1}'),
                        'yield_value': yield_matrix[i][j]
                    })
    elif status == pywraplp.Solver.FEASIBLE:
        total_yield_val = solver.Objective().Value()  # 최적은 아니지만 가능한 해
        # (위와 동일하게 결과 추출)
        for i in range(num_cf):
            for j in range(num_tft):
                if (i, j) in x and x[i, j].solution_value() > 0.5:
                    matched_pairs_info.append({
                        'cf_id': cf_panels[i].get('id', f'CF{i + 1}'),
                        'tft_id': tft_panels[j].get('id', f'TFT{j + 1}'),
                        'yield_value': yield_matrix[i][j]
                    })
        error_msg = "최적해를 찾았지만, 더 좋은 해가 있을 수 있습니다 (Feasible solution)."
    else:
        error_msg = "매칭 해를 찾지 못했습니다. (Solver status: " + str(status) + ")"
        if status == pywraplp.Solver.INFEASIBLE:
            error_msg = "문제가 실행 불가능(Infeasible)합니다. 제약 조건을 확인하세요."
        elif status == pywraplp.Solver.UNBOUNDED:
            error_msg = "문제가 무한(Unbounded)합니다."
        elif status == pywraplp.Solver.NOT_SOLVED:
            error_msg = "솔버가 문제를 풀지 못했습니다."

    return matched_pairs_info, total_yield_val, error_msg


# --- run_matching_algorithm 함수 정의 끝 ---



# OR-Tools를 사용하여 매칭 알고리즘 실행
matched_pairs, total_yield, error_msg = run_optimization(cf_panels, tft_panels)

logger.info("Matched Pairs:")
for pair in matched_pairs:
    logger.info(f"CF ID: {pair['cf_id']}, TFT ID: {pair['tft_id']}, Yield: {pair['yield_value']}")
logger.info(f"Total Yield: {total_yield}")
logger.info(f"Error Message: {error_msg}")
# 예시 데이터
