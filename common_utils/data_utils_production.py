from common_utils.common_data_utils import save_json_data
import logging
import datetime

logger = logging.getLogger(__name__)

preset_lot_sizing_num_periods = 6
preset_lot_sizing_data=[
    {'demand': '145', 'setup_cost': '321', 'prod_cost': '9', 'holding_cost': '2', 'capacity': '243'},
    {'demand': '60', 'setup_cost': '319', 'prod_cost': '9', 'holding_cost': '5', 'capacity': '156'},
    {'demand': '116', 'setup_cost': '230', 'prod_cost': '5', 'holding_cost': '3', 'capacity': '218'},
    {'demand': '146', 'setup_cost': '224', 'prod_cost': '12', 'holding_cost': '4', 'capacity': '265'},
    {'demand': '69', 'setup_cost': '430', 'prod_cost': '7', 'holding_cost': '5', 'capacity': '181'},
    {'demand': '125', 'setup_cost': '435', 'prod_cost': '8', 'holding_cost': '5', 'capacity': '290'},
    {'demand': '65', 'setup_cost': '458', 'prod_cost': '6', 'holding_cost': '3', 'capacity': '193'},
    {'demand': '58', 'setup_cost': '307', 'prod_cost': '5', 'holding_cost': '2', 'capacity': '291'},
    {'demand': '125', 'setup_cost': '286', 'prod_cost': '12', 'holding_cost': '5', 'capacity': '257'},
    {'demand': '68', 'setup_cost': '444', 'prod_cost': '15', 'holding_cost': '3', 'capacity': '193'},
    {'demand': '116', 'setup_cost': '421', 'prod_cost': '13', 'holding_cost': '4', 'capacity': '163'},
    {'demand': '83', 'setup_cost': '431', 'prod_cost': '9', 'holding_cost': '2', 'capacity': '277'}
]

preset_single_machine_num_jobs = 5
preset_single_machine_objective_choice = 'total_flow_time'
preset_single_machine_objective = [
            {'value': 'total_flow_time', 'name': '총 흐름 시간 최소화 (SPT)'},
            {'value': 'makespan', 'name': '총 완료 시간 최소화 (Makespan)'},
            {'value': 'total_tardiness', 'name': '총 지연 시간 최소화'}
        ]
preset_single_machine_data =[
    {'id': 'Job 1', 'processing_time': '7', 'due_date': '20', 'release_time': '0'},
    {'id': 'Job 2', 'processing_time': '3', 'due_date': '5', 'release_time': '0'},
    {'id': 'Job 3', 'processing_time': '5', 'due_date': '25', 'release_time': '6'},
    {'id': 'Job 4', 'processing_time': '4', 'due_date': '10', 'release_time': '0'},
    {'id': 'Job 5', 'processing_time': '1', 'due_date': '30', 'release_time': '0'},
    {'id': 'Job 6', 'processing_time': '14', 'due_date': '60', 'release_time': '2'},
    {'id': 'Job 7', 'processing_time': '7', 'due_date': '45', 'release_time': '22'},
    {'id': 'Job 8', 'processing_time': '5', 'due_date': '28', 'release_time': '12'},
    {'id': 'Job_9', 'processing_time': '11', 'due_date': '45', 'release_time': '18'},
    {'id': 'Job_10', 'processing_time': '7', 'due_date': '70', 'release_time': '3'},
]

preset_flow_shop_num_jobs = 4
preset_flow_shop_num_machines = 3
preset_flow_shop_data = [
    {'id': 'Job_1', 'processing_times': [29,78,9,36,49]},
    {'id': 'Job_2', 'processing_times': [43,92,8,45,68]},
    {'id': 'Job_3', 'processing_times': [90,85,87,32,91]},
    {'id': 'Job_4', 'processing_times': [77,39,55,64,82]},
    {'id': 'Job_5', 'processing_times': [95,13,47,84,22]},
    {'id': 'Job_6', 'processing_times': [20,88,70,69,74]},
    {'id': 'Job_7', 'processing_times': [58,48,85,6,86]},
    {'id': 'Job_8', 'processing_times': [73,10,29,76,4]},
    {'id': 'Job_9', 'processing_times': [36,2,31,75,59]},
    {'id': 'Job_10', 'processing_times': [12,88,58,99,9]}
]

preset_job_shop_num_jobs = 4
preset_job_shop_num_machines = 3

preset_job_shop_data = [
    {'id': 'Job_1', 'processing_times': [29,78,9,36,49], 'selected_routing': '0-1-2'},
    {'id': 'Job_2', 'processing_times': [43,92,8,45,68], 'selected_routing': '2-0-1'},
    {'id': 'Job_3', 'processing_times': [90,85,87,32,91], 'selected_routing': '1-2-0'},
    {'id': 'Job_4', 'processing_times': [77,39,55,64,82], 'selected_routing': '2-0-1'},
    {'id': 'Job_5', 'processing_times': [95,13,47,84,22], 'selected_routing': '0-1-2'},
    {'id': 'Job_6', 'processing_times': [20,88,70,69,74], 'selected_routing': '1-2-0'},
    {'id': 'Job_7', 'processing_times': [58,48,85,6,86], 'selected_routing': '0-1-2'},
    {'id': 'Job_8', 'processing_times': [73,10,29,76,4], 'selected_routing': '2-0-1'},
    {'id': 'Job_9', 'processing_times': [36,2,31,75,59], 'selected_routing': '1-2-0'},
    {'id': 'Job_10', 'processing_times': [12,88,58,99,9], 'selected_routing': '2-0-1'}
]

preset_rcpsp_num_activities = 8
preset_rcpsp_num_resources = 3

# --- 기본 데이터 풀 정의 ---
preset_rcpsp_activities_data = [
    # id, duration, predecessors, [res1, res2, res3]
    {'id': 'A.기획/설계', 'duration': '5', 'predecessors': '', 'res_reqs': [1, 1, 1]},
    {'id': 'B.UI/UX디자인', 'duration': '4', 'predecessors': '1', 'res_reqs': [2, 0, 1]},
    {'id': 'C.DB설계', 'duration': '3', 'predecessors': '1', 'res_reqs': [0, 2, 0]},
    {'id': 'D.FE개발', 'duration': '8', 'predecessors': '2', 'res_reqs': [3, 0, 0]},
    {'id': 'E.BE개발', 'duration': '7', 'predecessors': '3', 'res_reqs': [0, 4, 0]},
    {'id': 'F.통합테스트', 'duration': '4', 'predecessors': '4,5', 'res_reqs': [1, 1, 3]},
    {'id': 'G.피드백반영', 'duration': '3', 'predecessors': '6', 'res_reqs': [2, 2, 1]},
    {'id': 'H.최종배포', 'duration': '2', 'predecessors': '7', 'res_reqs': [1, 1, 2]},
    # 9, 10번째 활동 데이터 (필요시 추가)
    {'id': 'I.문서화', 'duration': '5', 'predecessors': '8', 'res_reqs': [1, 1, 0]},
    {'id': 'J.마케팅준비', 'duration': '6', 'predecessors': '1', 'res_reqs': [0, 0, 1]},
]
preset_rcpsp_resource_data = [
        {'name': 'Front-End 개발자', 'availability': '4'},
        {'name': 'Back-End 개발자', 'availability': '5'},
        {'name': 'QA 엔지니어', 'availability': '3'},
    ]


def create_lot_sizing_json_data(form_data, num_periods):
    """
    폼 데이터로부터 Lot Sizing 문제 입력을 위한 딕셔너리를 생성하고 검증합니다.
    """
    logger.info("Creating and validating lot sizing input data from form.")
    demands = []
    setup_costs = []
    prod_costs = []
    holding_costs = []
    capacities = []
    for t in range(num_periods):
        try:
            demand = int(form_data.get(f'demand_{t}'))
            setup_cost = int(form_data.get(f'setup_cost_{t}'))
            prod_cost = int(form_data.get(f'prod_cost_{t}'))
            holding_cost = int(form_data.get(f'holding_cost_{t}'))
            capacity = form_data.get(f'capacity_{t}')

            if not all(isinstance(v, int) and v >= 0 for v in [demand, setup_cost, prod_cost, holding_cost]):
                raise ValueError(f"기간 {t + 1}의 입력값은 0 이상의 정수여야 합니다.")
            if capacity is None or not capacity.strip().isdigit():
                raise ValueError(f"기간 {t + 1}의 생산 능력(Capacity)이 올바른 숫자가 아닙니다.")

            demands.append(demand)
            setup_costs.append(setup_cost)
            prod_costs.append(prod_cost)
            holding_costs.append(holding_cost)
            capacities.append(int(capacity))

        except (ValueError, TypeError) as e:
            # 더 구체적인 오류 메시지를 위해 래핑
            raise ValueError(f"기간 {t + 1} 처리 중 오류 발생: {e}")

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_periods": num_periods,
        'demands': demands,
        'setup_costs': setup_costs,
        'prod_costs': prod_costs,
        'holding_costs': holding_costs,
        'capacities': capacities
    }

    return input_data


def create_single_machine_json_data(jobs_list, objective_choice, num_jobs):
    """
    폼 데이터로부터 Single Machine 문제 입력을 위한 딕셔너리를 생성하고 검증합니다.
    """
    logger.info("Creating and validating single machine input data from form.")

    for job in jobs_list:
        try:
            job["processing_time"] = int(job["processing_time"])
            job["due_date"] = int(job["due_date"])
            job["release_time"] = int(job["release_time"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Job '{job.get('id', 'Unknown')}' has invalid processing_time or due_date: {e}")

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": 'single_machine',
        "num_jobs": num_jobs,
        'objective_choice': objective_choice,
        'jobs_list': jobs_list
    }

    return input_data


def create_flow_shop_json_data(form_data):
    num_jobs = int(form_data.get('num_jobs', 3))
    num_machines = int(form_data.get('num_machines', 3))

    processing_times = []
    job_ids = []
    for i in range(num_jobs):
        job_ids.append(form_data.get(f'job_{i}_id', f'Job {i+1}'))
        job_times = []
        for j in range(num_machines):
            try:
                time_val = int(form_data.get(f'p_{i}_{j}'))
                if time_val < 0:
                    raise ValueError(f"작업 {i+1}, 기계 {j+1}의 처리 시간은 음수가 될 수 없습니다.")
                job_times.append(time_val)
            except (ValueError, TypeError):
                raise ValueError(f"작업 {i+1}, 기계 {j+1}의 처리 시간이 올바른 숫자가 아닙니다.")
        processing_times.append(job_times)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": 'flow_shop',
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "job_ids": job_ids,
        "processing_times": processing_times,
        # form_parameters는 필요 시 추가
    }
    return input_data


def create_job_shop_json_data(form_data):
    logger.debug("Creating and validating job shop input data from new form.")
    num_jobs = int(form_data.get('num_jobs', 3))
    num_machines = int(form_data.get('num_machines', 3))

    jobs_data = []
    for i in range(num_jobs):
        job_id = form_data.get(f'job_{i}_id', f'Job {i + 1}')

        # 1. 기계별 처리 시간 파싱
        processing_times_on_machines = []
        for j in range(num_machines):
            try:
                time_val = int(form_data.get(f'p_{i}_{j}'))
                if time_val <= 0: raise ValueError("처리 시간은 0보다 커야 합니다.")
                processing_times_on_machines.append(time_val)
            except (ValueError, TypeError):
                raise ValueError(f"작업 '{job_id}'의 기계 {j + 1} 처리 시간이 올바른 숫자가 아닙니다.")

        # 2. 선택된 공정 순서(라우팅) 파싱
        selected_routing_str = form_data.get(f'job_{i}_routing')
        if not selected_routing_str:
            raise ValueError(f"작업 '{job_id}'의 공정 순서가 선택되지 않았습니다.")

        # 예: "0-2-1" -> [0, 2, 1]
        try:
            routing = [int(m_idx) for m_idx in selected_routing_str.split('-')]
            if len(set(routing)) != num_machines:
                raise ValueError("하나의 작업은 각 기계를 정확히 한 번씩만 방문해야 합니다.")
        except (ValueError, TypeError):
            raise ValueError(f"작업 '{job_id}'의 공정 순서 형식이 잘못되었습니다.")

        # 최종 작업 데이터 구성: [(기계_id, 처리_시간), ...]
        job_operations = []
        for machine_idx in routing:
            job_operations.append((machine_idx, processing_times_on_machines[machine_idx]))

        jobs_data.append(job_operations)

    input_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "problem_type": form_data.get('problem_type'),
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "jobs": jobs_data,
        "job_ids": [form_data.get(f'job_{i}_id', f'Job {i + 1}') for i in range(num_jobs)]
    }
    return input_data


def create_job_shop_json_data_ori(form_data):
    logger.debug("Creating and validating job shop input data from form.")
    num_jobs = int(form_data.get('num_jobs', 3))
    num_machines = int(form_data.get('num_machines', 3))

    jobs_data = []
    for i in range(num_jobs):
        job_ops = []
        for j in range(num_machines):
            try:
                machine_id = int(form_data.get(f'op_{i}_{j}_machine'))
                processing_time = int(form_data.get(f'op_{i}_{j}_time'))
                if not (0 <= machine_id < num_machines and processing_time > 0):
                    raise ValueError(f"작업 {i + 1}, 공정 {j + 1}의 기계 ID 또는 처리 시간이 유효하지 않습니다.")
                job_ops.append((machine_id, processing_time))
            except (ValueError, TypeError):
                raise ValueError(f"작업 {i + 1}, 공정 {j + 1}의 입력값이 올바른 숫자가 아닙니다.")
        jobs_data.append(job_ops)

    # 유효성 검사: 각 작업이 모든 기계를 정확히 한 번씩 사용하는지 확인
    for i, job in enumerate(jobs_data):
        machines_used = [op[0] for op in job]
        if len(set(machines_used)) != num_machines:
            raise ValueError(f"작업 {i + 1}은 각 기계를 정확히 한 번씩만 사용해야 합니다.")

    input_data = {
        "problem_type": "job_shop",
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "jobs": jobs_data
    }
    return input_data


def create_rcpsp_json_data(form_data):
    logger.debug("Creating and validating RCPSP input data from form.")
    num_activities = int(form_data.get('num_activities', 3))
    num_resources = int(form_data.get('num_resources', 2))

    input_data = {
        'problem_type': 'rcpsp',
        'num_activities': num_activities,
        'num_resources': num_resources,
        'resource_availabilities': [],
        'activities': []
    }

    # 1. 가용 자원량 파싱
    for k in range(num_resources):
        try:
            avail = int(form_data.get(f'resource_{k}_availability'))
            if avail < 0: raise ValueError("가용 자원량은 음수가 될 수 없습니다.")
            input_data['resource_availabilities'].append(avail)
        except (ValueError, TypeError):
            raise ValueError(f"자원 {k + 1}의 가용량이 올바른 숫자가 아닙니다.")

    # 2. 활동 데이터 파싱
    for i in range(num_activities):
        activity_data = {}
        try:
            activity_data['id'] = form_data.get(f'activity_{i}_id', f'Task {i + 1}')
            activity_data['duration'] = int(form_data.get(f'activity_{i}_duration'))

            # 선후 관계 파싱 (예: "1,2" -> [0,1])
            preds_str = form_data.get(f'activity_{i}_predecessors', '')
            if preds_str:
                # 1-based index to 0-based index
                activity_data['predecessors'] = [int(p.strip()) - 1 for p in preds_str.split(',') if p.strip()]
            else:
                activity_data['predecessors'] = []

            # 자원 요구량 파싱
            activity_data['resource_reqs'] = []
            for k in range(num_resources):
                req = int(form_data.get(f'activity_{i}_res_{k}_req'))
                activity_data['resource_reqs'].append(req)

            # 유효성 검사
            if activity_data['duration'] <= 0: raise ValueError("처리 기간은 0보다 커야 합니다.")
            if any(r < 0 for r in activity_data['resource_reqs']): raise ValueError("자원 요구량은 음수가 될 수 없습니다.")

        except (ValueError, TypeError):
            raise ValueError(f"활동 '{activity_data.get('id')}'의 입력값이 올바르지 않습니다.")

        input_data['activities'].append(activity_data)

    return input_data

def save_production_json_data(input_data):
    problem_type = input_data.get('problem_type')
    dir=f'production_{problem_type}_data'
    filename_pattern = ''
    if "lot_sizing" == input_data.get('problem_type'):
        num_periods = input_data.get('num_periods')
        filename_pattern = f"periods{num_periods}"
    elif "single_machine" == input_data.get('problem_type'):
        num_jobs = input_data.get('num_jobs')
        filename_pattern = f"jobs{num_jobs}_{input_data.get('objective_choice')}"
    elif "flow_shop" == input_data.get('problem_type'):
        num_jobs = input_data.get('num_jobs')
        num_machines = input_data.get('num_machines')
        filename_pattern = f"jobs{num_jobs}_machine{num_machines}"
    elif "job_shop" == input_data.get('problem_type'):
        num_jobs = input_data.get('num_jobs')
        num_machines = input_data.get('num_machines')
        filename_pattern = f"jobs{num_jobs}_machine{num_machines}"
    elif "rcpsp" == input_data.get('problem_type'):
        num_resources = input_data.get('num_resources')
        num_activities = input_data.get('num_activities')
        filename_pattern = f"resource{num_resources}_activities{num_activities}"
    return save_json_data(input_data, dir, filename_pattern)