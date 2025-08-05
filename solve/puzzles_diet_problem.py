from common_run_opt import get_solving_time_sec
from ortools.linear_solver import pywraplp  # OR-Tools MIP solve (실제로는 LP 솔버 사용)
import json
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)


def run_diet_optimizer(input_data):
    logger.info("Running Diet Problem Optimizer.")

    foods = input_data['food_items']
    nutrients = input_data['nutrient_reqs']
    num_foods = len(foods)
    num_nutrients = len(nutrients)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None, "오류: 선형 계획법 솔버(GLOP)를 생성할 수 없습니다.", 0.0

    # 변수 x_i: i번째 음식의 섭취량
    x=[]
    for food in foods:
        var = solver.NumVar(food['min_intake'], food['max_intake'], f"x_{food['name']}")
        x.append(var)
        logger.solve(f"Var: '{var.name()}: lb:{var.lb()}, ub:{var.ub()}")

    logger.debug(f"Created {len(x)} food variables.")

    # 제약: 각 영양소의 최소/최대 섭취량 만족

    for i in range(num_nutrients):
        nutrient_terms = []
        constraint = solver.Constraint(nutrients[i]['min'], nutrients[i]['max'], nutrients[i]['name'])
        for j in range(num_foods):
            constraint.SetCoefficient(x[j], foods[j]['nutrients'][i])
            nutrient_terms.append(f"{foods[j]['nutrients'][i]}*{x[j].name()}")
        nutrient_expr_str = " + ".join(nutrient_terms)
        logger.solve(f"Eq: {constraint.name()}:{constraint.lb()} <= {nutrient_expr_str} <= {constraint.ub()}")
    logger.debug(f"Added {num_nutrients} nutrient constraints.")

    # 목표 함수: 총 비용 최소화
    objective = solver.Objective()
    for i in range(num_foods):
        objective.SetCoefficient(x[i], foods[i]['cost'])
    objective.SetMinimization()
    logger.debug("Objective set to minimize total cost.")

    # 해결
    status = solver.Solve()

    # 결과 추출
    results = {'diet_plan': [], 'total_cost': 0, 'nutrient_summary': []}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL:
        results['total_cost'] = solver.Objective().Value()

        for i in range(num_foods):
            intake = x[i].solution_value()
            if intake > 1e-6:  # 매우 작은 값은 무시
                results['diet_plan'].append({
                    'name': foods[i]['name'],
                    'intake': round(intake, 2),
                    'cost': round(intake * foods[i]['cost'], 2)
                })

        for i in range(num_nutrients):
            total_nutrient_intake = sum(foods[j]['nutrients'][i] * x[j].solution_value() for j in range(num_foods))
            results['nutrient_summary'].append({
                'name': nutrients[i]['name'],
                'min_req': nutrients[i]['min'],
                'max_req': nutrients[i]['max'],
                'actual_intake': round(total_nutrient_intake, 2)
            })
    else:
        error_msg = "최적 식단을 찾지 못했습니다. 제약 조건이 너무 엄격하거나(INFEASIBLE), 문제가 잘못 정의되었을 수 있습니다."

    return results, error_msg, solver.WallTime() * 1000

with open('../test_data/puzzles_diet_problem_data/test.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_diet_optimizer(input_data)
logger.info(results_data)