from ortools.sat.python import cp_model # CP-SAT 솔버 사용
import json
import datetime
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

def run_single_machine_optimizer(input_data):
    """
    OR-Tools CP-SAT를 사용하여 단일 기계 스케줄링 문제를 해결합니다.
    input_data: [{'id': 'A', 'processing_time': 10, 'due_date': 20}, ...]
    objective_choice: 최소화할 목표 (예: 'total_flow_time', 'total_tardiness')
    """
    objective_choice=input_data.get('objective_choice')
    logger.info(f"Running Single Machine Scheduler for objective: {objective_choice}")
    logger.debug(f"Jobs Data: {input_data}")

    num_jobs = input_data.get('num_jobs')
    if num_jobs == 0:
        return None, "오류: 작업 데이터가 없습니다.", 0.0

    model = cp_model.CpModel()

    jobs_list = input_data.get('jobs_list')
    # --- 1. 데이터 및 모델 범위(Horizon) 설정 ---
    all_processing_times = [j['processing_time'] for j in jobs_list]
    horizon = sum(all_processing_times)  # 모든 작업이 순차적으로 끝나는 시간

    # --- 2. 결정 변수 생성 ---
    # 각 작업의 시작 시간, 종료 시간, 기간(Interval) 변수
    start_vars = [model.NewIntVar(0, horizon, f'start_{i}') for i in range(num_jobs)]
    end_vars = [model.NewIntVar(0, horizon, f'end_{i}') for i in range(num_jobs)]
    interval_vars = [
        model.NewIntervalVar(start_vars[i], jobs_list[i]['processing_time'], end_vars[i], f'interval_{i}')
        for i in range(num_jobs)
    ]
    logger.debug(f"Created {num_jobs * 3} variables (start, end, interval). Horizon: {horizon}")

    # --- 3. 제약 조건 설정 ---
    # 3.1. No Overlap 제약: 단일 기계는 한 번에 하나의 작업만 처리
    model.AddNoOverlap(interval_vars)
    logger.debug("Added NoOverlap constraint.")

    # --- 4. 목표 함수 설정 ---
    if objective_choice == 'total_flow_time':
        # 총 흐름 시간(Total Completion Time) 최소화
        model.Minimize(sum(end_vars))
        logger.debug("Objective set to: Minimize Total Flow Time.")
    elif objective_choice == 'makespan':
        # 총 완료 시간(Makespan) 최소화
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)
        logger.debug("Objective set to: Minimize Makespan.")
    elif objective_choice == 'total_tardiness':
        # 총 지연 시간(Total Tardiness) 최소화
        tardiness_vars = [model.NewIntVar(0, horizon, f'tardiness_{i}') for i in range(num_jobs)]
        for i in range(num_jobs):
            due_date = input_data[i]['due_date']
            # T_i >= C_i - d_i
            model.Add(tardiness_vars[i] >= end_vars[i] - due_date)
        model.Minimize(sum(tardiness_vars))
        logger.debug("Objective set to: Minimize Total Tardiness.")
    else:
        # 기본 목표 또는 오류 처리
        logger.warning(f"Unknown objective '{objective_choice}'. Defaulting to total_flow_time.")
        model.Minimize(sum(end_vars))

    # --- 5. 문제 해결 ---
    solver = cp_model.CpSolver()
    logger.info("Solving the Single Machine Scheduling model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve(model)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {solver.StatusName(status)}, Time: {processing_time_ms:.2f} ms")

    # --- 6. 결과 추출 ---
    results = {'schedule': [], 'objective_value': 0}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results['objective_value'] = solver.ObjectiveValue()

        for i in range(num_jobs):
            results['schedule'].append({
                'id': jobs_list[i].get('id', f'Job {i + 1}'),
                'start': solver.Value(start_vars[i]),
                'end': solver.Value(end_vars[i]),
                'processing_time': jobs_list[i]['processing_time'],
                'due_date': jobs_list[i]['due_date']
            })

        # 시작 시간 순서로 결과 정렬
        results['schedule'].sort(key=lambda item: item['start'])

    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Single Machine Scheduling failed: {error_msg}")

    return results, error_msg, processing_time_ms

with open('../test_data/production_single_machine_data/jobs4_total_flow_time.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_single_machine_optimizer(input_data)
logger.info(results_data)