from ortools.sat.python import cp_model
import datetime
import json
from logging_config import setup_logger
import logging
import settings

setup_logger()
logger = logging.getLogger(__name__)

def run_sudoku_solver_optimizer(input_data):
    """
    OR-Tools CP-SAT를 사용하여 스도쿠 퍼즐을 해결합니다.

    Args:
        initial_grid (list of list of int): 9x9 스도쿠 퍼즐. 빈 칸은 0으로 표시.

    Returns:
        tuple: (solved_grid, error_message, processing_time)
    """
    input_grid = input_data.get('input_grid')
    model = cp_model.CpModel()

    # 1. 결정 변수 생성
    # 각 셀은 1에서 9 사이의 정수 값을 가집니다.
    grid = {}
    for i in range(9):
        for j in range(9):
            grid[(i, j)] = model.NewIntVar(1, 9, f'cell_{i}_{j}')

    # 2. 제약 조건 추가
    # 각 행(row)의 모든 숫자는 달라야 합니다.
    for i in range(9):
        model.AddAllDifferent([grid[(i, j)] for j in range(9)])

    # 각 열(column)의 모든 숫자는 달라야 합니다.
    for j in range(9):
        model.AddAllDifferent([grid[(i, j)] for i in range(9)])

    # 3x3 각 서브그리드의 모든 숫자는 달라야 합니다.
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_vars = []
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    subgrid_vars.append(grid[(row, col)])
            model.AddAllDifferent(subgrid_vars)

    # 초기 퍼즐의 주어진 숫자들을 제약으로 추가합니다.
    for i in range(9):
        for j in range(9):
            if input_grid[i][j] != 0:
                model.Add(grid[(i, j)] == input_grid[i][j])

    # 3. 문제 해결
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.ORTOOLS_TIME_LIMIT

    solve_start_time = datetime.datetime.now()
    status = solver.Solve(model)
    solve_end_time = datetime.datetime.now()
    processing_time = (solve_end_time - solve_start_time).total_seconds()

    # 4. 결과 추출
    solved_grid = None
    error_message = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solved_grid = [[solver.Value(grid[(i, j)]) for j in range(9)] for i in range(9)]
    else:
        error_message = "스도쿠 퍼즐의 해를 찾을 수 없습니다. 입력된 퍼즐이 유효한지 확인해주세요."

    return solved_grid, error_message, processing_time

with open('../test_data/puzzles_sudoku_data/test.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_sudoku_solver_optimizer(input_data)
logger.info(results_data)