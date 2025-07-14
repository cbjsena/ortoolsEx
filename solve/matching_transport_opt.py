from common_run_opt import get_solving_time_sec, solved_log
from ortools.graph.python import linear_sum_assignment # 할당 문제 전용 솔버
from ortools.linear_solver import pywraplp
import datetime
import json
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)
def run_matching_transport_optimizer_old(input_data):
    """
    OR-Tools의 LinearSumAssignment 솔버를 사용하여 작업 배정 문제를 해결합니다.
    cost_matrix: 비용 행렬 (리스트의 리스트)
    """
    logger.info("Running Assignment Problem Optimizer.")
    logger.debug(f"Cost Matrix: {input_data}")

    num_workers = len(input_data['driver_names'])
    if num_workers == 0:
        return [], 0, "오류: 비용 행렬 데이터가 없습니다."
    num_tasks = len(input_data['zone_names'])
    cost_matrix=input_data['cost_matrix']

    solver = linear_sum_assignment.SimpleLinearSumAssignment()

    for woker in range(num_workers):
        for task in range(num_tasks):
            if cost_matrix[woker][task] is not None:
                solver.add_arc_with_cost(woker, task, int(cost_matrix[woker][task]))

    logger.info("Solving the assignment model...")
    status = solver.solve()
    logger.info(f"Solver finished. Status: {status}, Time: {solver.WallTime():.2f} ms")

    results = {'assignments':[], 'total_cost':0}
    error_msg = None

    if status == solver.OPTIMAL:
        results['total_cost'] =solver.optimal_cost()
        logger.info(f'Total cost = {results["total_cost"]}')
        for i in range(num_workers):
            assigned_task = solver.right_mate(i)
            cost = solver.assignment_cost(i)
            results['assignments'].append({
                'workder_id': f'기사{i + 1}',
                'task_id': f'구역 {assigned_task + 1}',
                'cost': cost
            })
            logger.debug(f'Worker {i} assigned to task {assigned_task} with a cost of {cost}')

    elif status == solver.INFEASIBLE:
        error_msg = "실행 불가능한 문제입니다. 모든 작업자/작업 쌍에 대한 비용이 정의되었는지 확인하세요."
    elif status == solver.POSSIBLE_OVERFLOW:
        error_msg = "계산 중 오버플로우가 발생했습니다. 비용 값의 크기를 확인하세요."
    else:
        error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"

    if error_msg:
        logger.error(f"Assignment optimization failed: {error_msg}")

    return results, error_msg, processing_time_ms

def run_matching_transport_optimizer_new(input_data):
    """
    OR-Tools의 LinearSumAssignment 솔버를 사용하여 작업 배정 문제를 해결합니다.
    cost_matrix: 비용 행렬 (리스트의 리스트)
    """
    logger.info("Running Assignment Problem Optimizer.")
    logger.debug(f"Cost Matrix: {input_data}")

    workers = input_data['driver_names']
    if len(workers) == 0:
        return [], 0, "Error: There is no workers."

    tasks = input_data['zone_names']
    if len(tasks) == 0:
        return [], 0, "Error: There is no tasks."

    costs=input_data['cost_matrix']
    num_workers = len(costs)
    num_tasks = len(costs[0])
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x ={}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i,j] = solver.IntVar(0, 1, f'{workers[i]}_{tasks[j]}')

    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) == 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    logger.info("Solving the assignment model...")
    status = solver.Solve()
    solved_log(solver,  status,'assignment')

    results = {'assignments':[], 'total_cost':0}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or pywraplp.Solver.FEASIBLE:
        results['total_cost'] = solver.Objective().Value()
        logger.info(f"Total cost = {results['total_cost']}")
        for i in range(num_workers):
            for j in range(num_tasks):
                cost = costs[i][j]
                if x[i, j].solution_value() > 0.5:
                    results['assignments'].append({
                        'worker_name': workers[i],
                        'task_name': tasks[j],
                        'cost':cost
                    })
                    logger.debug(f'Worker {i} assigned to task {j} with a cost of {cost}')

    elif status == pywraplp.Solver.INFEASIBLE:
        error_msg = "실행 불가능한 문제입니다. 모든 작업자/작업 쌍에 대한 비용이 정의되었는지 확인하세요."
    else:
        error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"

    if error_msg:
        logger.error(f"Assignment optimization failed: {error_msg}")

    return results, error_msg, get_solving_time_sec(solver)

with open('../test_data/matching_transport_data/test.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time = run_matching_transport_optimizer_new(input_data)
logger.info(processing_time)