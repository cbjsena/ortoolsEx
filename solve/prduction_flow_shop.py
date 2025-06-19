from ortools.sat.python import cp_model # CP-SAT 솔버 사용
import json
import datetime
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

def run_flow_shop_optimizer(input_data):
    logger.info("Running Flow Shop Optimizer.")

    processing_times = input_data['processing_times']
    num_jobs = input_data['num_jobs']
    num_machines = input_data['num_machines']

    model = cp_model.CpModel()

    # Horizon 계산
    horizon = sum(sum(job) for job in processing_times)

    # 변수 생성: C_ij (작업 i가 기계 j에서 끝나는 시간)
    completion_times = [[model.NewIntVar(0, horizon, f'C_{i}_{j}') for j in range(num_machines)] for i in
                        range(num_jobs)]

    # 제약 조건
    # 1. 기계 순서 제약 (작업 흐름)
    for i in range(num_jobs):
        for j in range(1, num_machines):
            model.Add(completion_times[i][j] >= completion_times[i][j - 1] + processing_times[i][j])

    # 2. 작업 순서 제약 (기계 독점) - 순열(Permutation) 플로우샵 가정
    # y_ik = 1 if job i is before job k
    y = {(i, k): model.NewBoolVar(f'y_{i}_{k}') for i in range(num_jobs) for k in range(num_jobs) if i < k}

    for j in range(num_machines):  # 모든 기계에서
        for i in range(num_jobs):
            for k in range(i + 1, num_jobs):
                # 작업 i가 k보다 먼저 끝나거나, k가 i보다 먼저 끝나야 함
                # C_kj >= C_ij + p_kj OR C_ij >= C_kj + p_ij
                # BigM 기법 사용
                # C_kj - (C_ij + p_ij) >= 0 또는 C_ij - (C_kj + p_kj) >= 0
                # 이 부분을 CP-SAT의 AddNoOverlap으로 더 효율적으로 모델링 가능
                pass  # 아래 NoOverlap으로 대체

    # 2. (개선된) 작업 순서 제약: NoOverlap 사용
    for j in range(num_machines):
        intervals = []
        for i in range(num_jobs):
            start_var = model.NewIntVar(0, horizon, f'start_{i}_{j}')
            # C_ij = start_ij + p_ij
            model.Add(completion_times[i][j] == start_var + processing_times[i][j])
            intervals.append(
                model.NewIntervalVar(start_var, processing_times[i][j], completion_times[i][j], f'interval_{i}_{j}'))
        model.AddNoOverlap(intervals)

    # 목표 함수: Makespan 최소화
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [completion_times[i][num_machines - 1] for i in range(num_jobs)])
    model.Minimize(makespan)

    # 해결
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0  # 시간 제한
    solve_start_time = datetime.datetime.now()
    status = solver.Solve(model)
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000

    # 결과 추출
    results = {'schedule': [], 'makespan': 0, 'sequence': []}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # 최적 순서 결정
        sequence_info = []
        for i in range(num_jobs):
            start_time_on_m0 = solver.Value(completion_times[i][0]) - processing_times[i][0]
            sequence_info.append({'job_index': i, 'start_time': start_time_on_m0})
        sequence_info.sort(key=lambda item: item['start_time'])

        optimal_sequence_indices = [item['job_index'] for item in sequence_info]
        optimal_sequence_ids = [input_data['job_ids'][i] for i in optimal_sequence_indices]

        # 계산된 최적 순서로 스케줄 및 Makespan 재계산 (결과 일관성 및 재사용성)
        results = calculate_flow_shop_schedule(
            processing_times,
            input_data['job_ids'],
            optimal_sequence_ids
        )
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"

    return results, error_msg, processing_time_ms

def calculate_flow_shop_schedule(processing_times, job_ids, sequence):
    """
    주어진 작업 순서(sequence)에 따라 Flow Shop 스케줄과 Makespan을 계산합니다.
    processing_times: [[p_ij, ...], ...]
    job_ids: ['Job 1', 'Job 2', ...]
    sequence: 순서를 나타내는 job_id 리스트. 예: ['Job 2', 'Job 1', 'Job 3']
    """
    num_jobs = len(processing_times)
    num_machines = len(processing_times[0]) if num_jobs > 0 else 0

    # job_id를 인덱스로 변환
    job_id_to_index = {job_id: i for i, job_id in enumerate(job_ids)}
    try:
        sequence_indices = [job_id_to_index[job_id] for job_id in sequence]
    except KeyError as e:
        raise ValueError(f"잘못된 작업 ID가 수동 순서에 포함되어 있습니다: {e}")

    if len(sequence_indices) != num_jobs or len(set(sequence_indices)) != num_jobs:
        raise ValueError("수동 순서에는 모든 작업이 정확히 한 번씩 포함되어야 합니다.")

    # 완료 시간 행렬 C_ij 초기화
    completion_times = [[0] * num_machines for _ in range(num_jobs)]

    # 재귀적 관계를 사용하여 완료 시간 계산
    for k in range(num_jobs):  # 순서 k (0 to n-1)
        job_idx = sequence_indices[k]
        for j in range(num_machines):  # 기계 j (0 to m-1)
            # 첫 번째 작업(k=0) 또는 첫 번째 기계(j=0)의 완료 시간
            prev_job_completion_on_same_machine = completion_times[sequence_indices[k - 1]][j] if k > 0 else 0
            prev_machine_completion_for_same_job = completion_times[job_idx][j - 1] if j > 0 else 0

            completion_times[job_idx][j] = max(prev_job_completion_on_same_machine,
                                               prev_machine_completion_for_same_job) + processing_times[job_idx][j]

    # Makespan은 마지막 순서의 작업이 마지막 기계에서 끝나는 시간
    makespan = completion_times[sequence_indices[-1]][num_machines - 1]

    # 간트 차트용 데이터 생성
    schedule = []
    for i in range(num_jobs):
        job_schedule = {'job_id': job_ids[i], 'tasks': []}
        for j in range(num_machines):
            end_time = completion_times[i][j]
            start_time = end_time - processing_times[i][j]
            job_schedule['tasks'].append({
                'machine': f'Machine {j + 1}',
                'start': start_time,
                'duration': processing_times[i][j],
                'end': end_time
            })
        schedule.append(job_schedule)
    results = {'schedule': schedule, 'makespan': makespan, 'sequence': sequence}

    return results

with open('../test_data/production_flow_shop_data/jobs4_machine3.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_flow_shop_optimizer(input_data)
logger.info(results_data)