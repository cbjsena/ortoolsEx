from common_run_opt import get_solving_time_sec
from ortools.linear_solver import pywraplp
import datetime
import json
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

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

    # 유효성 검사 실패 시 에러 메시지 설정
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

    formatted_html = formatted_html.replace("'","")
    return unmatched, formatted_html

def run_skill_matching_optimizer0(input_data):
    """
    자원-기술 매칭 문제를 해결하여 총 비용을 최소화합니다.
    resources_data: [{'id': 'R1', 'name': '김개발', 'cost': 100, 'skills': ['Python', 'ML']}, ...]
    projects_data: [{'id': 'P1', 'name': 'AI 모델 개발', 'required_skills': ['Python', 'ML', 'Cloud']}, ...]
    """
    logger.info("Running Resource-Skill Matching Optimizer...")
    resources_data = input_data['resources_data']
    projects_data = input_data['projects_data']
    num_resources = input_data['num_resources']
    num_projects = input_data['num_projects']

    if num_resources == 0 or num_projects == 0:
        return None, "오류: 인력 또는 프로젝트 데이터가 없습니다.", 0.0

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        logger.error("CBC MIP Solver not available for skill matching.")
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    # --- 1. 결정 변수 생성 ---
    # x[i][j] = 1 if resource i is assigned to project j, 0 otherwise
    x = {}
    for i in range(num_resources):
        for j in range(num_projects):
            res_id = resources_data[i].get('id')
            pro_id = projects_data[j].get('id')
            x[i, j] = solver.BoolVar(f'x_{res_id}_{pro_id}')

    logger.debug(f"Created {len(x)} assignment variables.")

    # --- 2. 제약 조건 설정 ---
    # 제약 1: 각 인력은 최대 하나의 프로젝트에만 할당됨
    for i in range(num_resources):
        res_id = resources_data[i].get('id')
        solver.Add(sum(x[i, j] for j in range(num_projects)) <= 1, f"resource_assignment_{res_id}")

    # 제약 2: 각 프로젝트에 요구되는 모든 기술은 반드시 충족되어야 함
    all_skills = set()
    for p in projects_data:
        all_skills.update(p.get('required_skills', []))

    for j in range(num_projects):
        project = projects_data[j]
        for skill in project.get('required_skills', []):
            pro_id = projects_data[j].get('id')
            # 프로젝트 j의 기술 s 요구는, 기술 s를 가진 인력 i 중 최소 한명이 프로젝트 j에 할당되어야 충족됨
            solver.Add(
                sum(x[i, j] for i in range(num_resources) if skill in resources_data[i].get('skills', [])) >= 1,
                f"skill_requirement_{pro_id}_{skill}"
            )
    logger.debug("Added resource and skill requirement constraints.")

    # --- 3. 목표 함수 설정 ---
    # 총 비용(급여) 최소화
    objective = solver.Objective()
    for i in range(num_resources):
        for j in range(num_projects):
            objective.SetCoefficient(x[i, j], resources_data[i].get('cost', 0))
    objective.SetMinimization()
    logger.debug("Objective function set to minimize total cost.")

    # --- 4. 문제 해결 ---
    logger.info("Solving the skill matching model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time_ms:.2f} ms")

    # --- 5. 결과 추출 ---
    results = {'assignments': {}, 'total_cost': 0, 'unassigned_resources': []}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found, but it might not be optimal.")

        results['total_cost'] = solver.Objective().Value()
        assigned_resource_indices = set()

        for j in range(num_projects):
            project_id = projects_data[j].get('id', f'P{j + 1}')
            results['assignments'][project_id] = []
            for i in range(num_resources):
                if x[i, j].solution_value() > 0.5:
                    resource = resources_data[i]
                    results['assignments'][project_id].append({
                        'resource_id': resource.get('id', f'R{i + 1}'),
                        'name': resource.get('name', f'인력{i + 1}'),
                        'cost': resource.get('cost', 0),
                        'skills': resource.get('skills', [])
                    })
                    assigned_resource_indices.add(i)

        for i in range(num_resources):
            if i not in assigned_resource_indices:
                results['unassigned_resources'].append(resources_data[i])

    else:  # 해를 찾지 못한 경우
        if status == pywraplp.Solver.INFEASIBLE:
            error_msg = "실행 불가능한 문제입니다. 프로젝트의 필수 기술을 가진 인력이 없거나, 제약 조건을 만족하는 할당이 불가능합니다."
        else:
            error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"
        logger.error(f"Skill matching optimization failed: {error_msg}")

    logger.info(f"최적 팀 구성 완료! 예상 총 비용: {results.get('total_cost', 0)}")
    for projects_id, resources in results['assignments'].items():
        # prg = projects_data.get(projects_id)
        logger.info(projects_id)
        # logger.info(f'projects:{prg.get('name')}')
    # resources_data, projects_data

    return results, error_msg, processing_time_ms


def run_skill_matching_optimizer(input_data):
    """
    자원-기술 매칭 문제를 해결하여 총 비용을 최소화합니다.
    Add slack variable
    """
    logger.info("Running Resource-Skill Matching Optimizer...")
    resources_data = input_data['resources_data']
    projects_data = input_data['projects_data']
    num_resources = input_data['num_resources']
    num_projects = input_data['num_projects']

    if num_resources == 0 or num_projects == 0:
        return None, "오류: 인력 또는 프로젝트 데이터가 없습니다.", 0.0

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        logger.error("CBC MIP Solver not available for skill matching.")
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    # --- 1. 결정 변수 생성 ---
    # x[i][j] = 1 if resource i is assigned to project j, 0 otherwise
    var_res_assign = {(i, j): solver.BoolVar(f"Assign_{resources_data[i]['id']}_{projects_data[j]['id']}")
         for i in range(num_resources) for j in range(num_projects)}
    logger.solve(f"**결정 변수:** {len(var_res_assign)}개의 자원 할당 변수 생성:")
    for (pro,j), var in var_res_assign.items():
        logger.solve(f"{var.name()}")

    # --- 2. 제약 조건 설정 ---
    # 제약 1: 각 인력은 최대 하나의 프로젝트에만 할당됨
    for pro in range(num_resources):
        # 제약 추가
        expr = sum(var_res_assign[pro, j] for j in range(num_projects))
        solver.Add(expr <= 1,f"Assign_{resources_data[pro]['id']}" )
        constraint_expr = " + ".join(
            f"{var_res_assign[pro, j].name()}" for j in range(num_projects)
        )
        logger.solve(f"Assign_{resources_data[pro]['id']}: {constraint_expr} <= 1")
    logger.debug("Added resource assignment constraints.")


    # 제약 2: 각 프로젝트에 요구되는 모든 기술은 반드시 충족되어야 함
    all_skills = {skill for p in projects_data for skill in p.get('required_skills', [])}

    logger.info("Phase 1: Solving for feasibility...")

    # 각 기술 요구사항 위반 여부를 나타내는 슬랙(slack) 변수 추가
    unfulfilled_skills = {}
    for j in range(num_projects):
        for skill in projects_data[j].get('required_skills', []):
            unfulfilled_skills[j, skill] = solver.BoolVar(f"unfulfilled_{projects_data[j]['id']}_{skill}")
    # for (i,j), var in unfulfilled_skills.items():
    #     logger.solve(f"{var.name()}")


    # 수정된 기술 요구사항 제약: sum(기술 보유 인력 할당) + (슬랙 변수) >= 1
    for j in range(num_projects):
        for skill in projects_data[j].get('required_skills', []):
            solver.Add(
                sum(var_res_assign[i, j] for i in range(num_resources) if skill in resources_data[i].get('skills', []))
                + unfulfilled_skills[j, skill] >= 1
            )

    for pro in range(num_projects):
        for skill in projects_data[pro].get('required_skills', []):
            expr = (sum(var_res_assign[res, pro] for res in range(num_resources) if skill in resources_data[res].get('skills', []))
                    + unfulfilled_skills[pro, skill])
            solver.Add(expr >= 1, f"Eq_Unfulfilled_{projects_data[pro]['id']}_{skill}")
            constraint_expr = " + ".join(
                f"{var_res_assign[res, pro].name()}" for res in range(num_resources) if skill in resources_data[res].get('skills', [])
            )+ " + "+ unfulfilled_skills[pro, skill].name()
            logger.solve(f"Eq_Unfulfilled_{projects_data[pro]['id']}_{skill}: {constraint_expr} >= 1")

    # 단계 1의 목표: 위반하는 기술 요구사항 수(슬랙 변수들의 합) 최소화
    feasibility_objective = solver.Objective()
    for s in unfulfilled_skills.values():
        feasibility_objective.SetCoefficient(s, 1)
    feasibility_objective.SetMinimization()

    # 단계 1 해결
    status = solver.Solve()

    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        return None, "오류: 실행 가능성 확인 모델을 푸는 데 실패했습니다.", 0.0

    # 위반된 제약 조건 확인
    if feasibility_objective.Value() > 0:
        logger.warning("Model is INFEASIBLE. Identifying unmet skill requirements...")
        unmet_requirements = []
        for (j, skill), var in unfulfilled_skills.items():
            if var.solution_value() > 0.5:
                project_name = projects_data[j].get('name')
                unmet_requirements.append(f"'{project_name}' 프로젝트의 '{skill}' 기술")

        error_msg = f"실행 불가능한 문제입니다. 다음 요구사항을 충족할 수 없습니다: {', '.join(unmet_requirements)}"
        return None, error_msg, 0.0

    # 실행 가능 확인 완료, 원래 문제로 전환
    # 단계 1 목표 제거 (새 목표 설정 시 자동으로 덮어쓰임)
    # 슬랙 변수들이 모두 0이 되도록 제약 추가
    for s in unfulfilled_skills.values():
        solver.Add(s == 0)

    # ======================================================================
    # === 단계 2: 비용 최소화 (Original Problem) ===
    # ======================================================================
    logger.info("Phase 2: Model is feasible. Solving for minimum cost...")

    cost_objective = solver.Objective()
    for pro in range(num_resources):
        for j in range(num_projects):
            cost_objective.SetCoefficient(var_res_assign[pro, j], resources_data[pro].get('cost', 0))
    cost_objective.SetMinimization()
    logger.debug("Objective function set to minimize total cost.")

    # --- 4. 문제 해결 ---
    logger.info("Solving the skill matching model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time_ms:.2f} ms")

    # --- 5. 결과 추출 ---
    results = {'assignments': {}, 'total_cost': 0, 'unassigned_resources': []}
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found, but it might not be optimal.")

        results['total_cost'] = solver.Objective().Value()
        assigned_resource_indices = set()

        for j in range(num_projects):
            project_id = projects_data[j].get('id', f'P{j + 1}')
            results['assignments'][project_id] = []
            for pro in range(num_resources):
                if var_res_assign[pro, j].solution_value() > 0.5:
                    resource = resources_data[pro]
                    results['assignments'][project_id].append({
                        'resource_id': resource.get('id', f'R{pro + 1}'),
                        'name': resource.get('name', f'인력{pro + 1}'),
                        'cost': resource.get('cost', 0),
                        'skills': resource.get('skills', [])
                    })
                    assigned_resource_indices.add(pro)

        for pro in range(num_resources):
            if pro not in assigned_resource_indices:
                results['unassigned_resources'].append(resources_data[pro])

    else:  # 해를 찾지 못한 경우
        if status == pywraplp.Solver.INFEASIBLE:
            error_msg = "실행 불가능한 문제입니다. 프로젝트의 필수 기술을 가진 인력이 없거나, 제약 조건을 만족하는 할당이 불가능합니다."
        else:
            error_msg = f"최적 할당을 찾지 못했습니다. (솔버 상태: {status})"
        logger.error(f"Skill matching optimization failed: {error_msg}")

    return results, error_msg, get_solving_time_sec(solver)

with open('../test_data/matching_resource_data/resource5_project3.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

unmatched, formatted_html = validate_required_skills(input_data)

# if len(unmatched) >0:
#     for key, value in unmatched.items():
#         print(f"skill: {key}, project:{value}")
# else:
results_data, error_msg_opt, processing_time_ms = run_skill_matching_optimizer(input_data)
logger.info(run_skill_matching_optimizer(input_data))

