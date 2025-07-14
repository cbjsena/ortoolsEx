from common_run_opt import get_solving_time_sec
from ortools.sat.python import cp_model # CP-SAT 솔버 사용
import json
import datetime
import collections
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
    status = solver.Solve(model)
    logger.info(f"Solver finished. Status: {status}, Time: {solver.WallTime():.2f} sec")

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

    return results, error_msg, solver.WallTime()


def run_single_machine_optimizer1(input_data):
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
    task_type = collections.namedtuple("task_type", "start end interval")
    all_tasks = {}
    logger.debug("--- Creating Decision Variables ---")
    for i, job in enumerate(jobs_list):
        job_id_str = str(job.get('id', i)).replace(' ', '_')  # 공백을 언더스코어로
        suffix = f'_{job_id_str}'
        start_var = model.NewIntVar(job.get('release_time', 0), horizon, f'start{suffix}')
        end_var = model.NewIntVar(0, horizon, f'end{suffix}')
        interval_var = model.NewIntervalVar(start_var, job['processing_time'], end_var, f'interval{suffix}')
        all_tasks[i] = task_type(start=start_var, end=end_var, interval=interval_var)
        logger.debug(f"  - Task {i} ({job_id_str}):")
        logger.debug(f"    - start var: 'start{suffix}', domain=[{job.get('release_time', 0)}, {horizon}]")
        logger.debug(f"    - end var: 'end{suffix}', domain=[0, {horizon}]")
        logger.debug(f"    - interval var: 'interval{suffix}', duration={job['processing_time']}")

    # --- 3. 제약 조건 설정 ---
    logger.debug("--- Creating Constraints ---")
    # 3.1. No Overlap 제약: 단일 기계는 한 번에 하나의 작업만 처리
    all_intervals = [task.interval for task in all_tasks.values()]
    model.AddNoOverlap(all_intervals)
    logger.debug(f"  - Constraint: AddNoOverlap on all intervals: {[iv.Name() for iv in all_intervals]}")

    # --- 4. 목표 함수 및 관련 제약 설정 ---
    logger.debug(f"--- Creating Objective Function ({objective_choice}) ---")
    end_vars = [task.end for task in all_tasks.values()]

    if objective_choice == 'total_flow_time':
        objective_var = sum(end_vars)
        model.Minimize(objective_var)
        logger.debug("  - Objective: Minimize(sum of all end_vars)")

    elif objective_choice == 'makespan':
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)
        logger.debug(f"  - Variable: 'makespan' = max of all end_vars")
        logger.debug("  - Objective: Minimize(makespan)")

    elif objective_choice == 'total_tardiness':
        tardiness_vars = [model.NewIntVar(0, horizon, f'tardiness_{i}') for i in range(num_jobs)]
        logger.debug(f"  - Variables (for objective): {[tv.Name() for tv in tardiness_vars]}")
        for i in range(num_jobs):
            due_date = jobs_list[i]['due_date']
            # T_i >= C_i - d_i  (C_i는 i번째 작업의 end_var)
            constraint = model.Add(tardiness_vars[i] >= all_tasks[i].end - due_date)
            logger.debug(f"    - Constraint for T_{i}: {str(constraint)}")  # 제약식 객체를 문자열로 출력

        objective_var = sum(tardiness_vars)
        model.Minimize(objective_var)
        logger.debug("  - Objective: Minimize(sum of all tardiness_vars)")

    else:
        # 기본 목표
        logger.warning(f"Unknown objective '{objective_choice}'. Defaulting to 'total_flow_time'.")
        objective_var = sum(end_vars)
        model.Minimize(objective_var)
        logger.debug("  - Objective (Default): Minimize(sum of all end_vars)")

    # --- 5. 문제 해결 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 100.0
    logger.info("Solving the Single Machine Scheduling model...")
    status = solver.Solve(model)
    logger.info(f"Solver finished. Status: {solver.StatusName(status)}, Time: {solver.WallTime():.2f} sec")

    # --- 6. 결과 추출 ---
    results = {'schedule': [], 'objective_value': 0}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results['objective_value'] = solver.ObjectiveValue()

        for i in range(num_jobs):
            results['schedule'].append({
                'id': jobs_list[i].get('id', f'Job {i + 1}'),
                'start': solver.Value(all_tasks[i].start),
                'end': solver.Value(all_tasks[i].end),
                'processing_time': jobs_list[i]['processing_time'],
                'due_date': jobs_list[i]['due_date'],
                'release_time': jobs_list[i]['release_time']
            })

        # 시작 시간 순서로 결과 정렬
        results['schedule'].sort(key=lambda item: item['start'])
        result_obj_info(results)
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Single Machine Scheduling failed: {error_msg}")

    return results, error_msg, solver.WallTime()

def run_job_shop() -> None:
    """Minimal jobshop problem."""
    # Data.
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)],  # Job2
    ]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.new_int_var(0, horizon, "start" + suffix)
            end_var = model.new_int_var(0, horizon, "end" + suffix)
            interval_var = model.new_interval_var(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.add_no_overlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )

        # Create per machine output lines.
        output = ""
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.objective_value}")
        print(output)
    else:
        print("No solution found.")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")


def result_obj_info(results):
    results['makespan '] = max(item['end'] for item in results['schedule'])
    results['total_tardiness '] = sum(max(0, job['end'] - job['due_date']) for job in results['schedule'])
    results['total_flow_time '] = sum(job['end'] - job['release_time'] for job in results['schedule'])

    return  results

with open('../test_data/production_single_machine_data/jobs3_makespan.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

type =2

if type == 1:
    results_data, error_msg_opt, processing_time_ms = run_single_machine_optimizer(input_data)
    logger.info(results_data)
elif type == 2:
    results_data, error_msg_opt, processing_time_ms = run_single_machine_optimizer1(input_data)
    logger.info(results_data)
elif type == 3:
    run_job_shop()
