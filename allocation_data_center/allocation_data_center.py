import json
from ortools.linear_solver import pywraplp  # OR-Tools MIP solver (실제로는 LP 솔버 사용)
from logging_config import setup_logger
import logging
import datetime  # 파일명 생성 등에 사용 가능
from math import floor

setup_logger()
logger = logging.getLogger(__name__)  # settin

with open('testcase/test1.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
# test_data = json.loads("testcase/test1.json")
global_constraints = test_data.get('global_constraints')
server_data = test_data.get('server_types')
demands_data =test_data.get('service_demands')

def run_optimization(global_constraints, server_data, demand_data):
    logger.info("Running Data Center Capacity Optimizer...")
    logger.debug(f"Global Constraints: {global_constraints}")
    logger.debug(f"Server Data: {server_data}")
    logger.debug(f"Demands Data: {demand_data}")

    solver = pywraplp.Solver.CreateSolver('CBC')  # 또는 'SCIP' 등 MIP 지원 솔버
    if not solver:
        logger.error("CBC/SCIP MIP Solver not available for data center capacity planning.")
        return None, "오류: MIP 솔버를 생성할 수 없습니다.", 0.0

    infinity = solver.infinity()
    num_server_data = len(server_data)
    num_demand_data = len(demand_data)
    total_budget = global_constraints.get('total_budget')
    total_power = global_constraints.get('total_power_kva')
    total_space = global_constraints.get('total_space_sqm')

    # --- 결정 변수 ---
    # Sv[i]: 서버 i 구매 수(정수 변수)
    svr_name = [solver.IntVar(0, infinity, f'Sv{i+1}') for i in range(num_server_data)]
    logger.solve(f"SV: 서버 i의 구매 개수, 총 {len(svr_name)}개 생성")
    for i, var in enumerate(svr_name):
        ub = floor(min(total_power/server_data[i].get('power_kva'),total_space/server_data[i].get('space_sqm')))
        logger.solve(f"  - {var.name()} (서버: {server_data[i].get('id', i)}), 범위: [{var.lb()}, {ub}]")

     # Dm[s]Sv[i]: 서비스 s를 위해 서버 i에 할당된 "자원 단위" 또는 "서비스 인스턴스 수"
    # 여기서는, 각 서비스가 특정 양의 CPU, RAM, Storage를 요구하고,
    # 각 서버이 특정 양의 CPU, RAM, Storage를 제공한다고 가정.
    Alloc = {}
    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        # 서비스 s의 최대 유닛 수 (수요) 만큼 변수 생성 고려
        # 또는, 총 제공 가능한 서비스 유닛을 변수로 할 수도 있음.
        # 여기서는 서비스 s를 서버 i에서 몇 '유닛'만큼 제공할지를 변수로 설정.
        # 이 '유닛'은 해당 서비스의 요구 자원에 맞춰짐.
        max_units_s = service.get('max_units', infinity) if service.get('max_units') is not None else infinity
        for i_idx in range(num_server_data):
            # 서비스 s를 서버 i에서 몇 유닛 제공할지 (이산적인 서비스 유닛으로 가정)
            Alloc[s_idx, i_idx] = solver.IntVar(0, max_units_s if max_units_s != infinity else solver.infinity(),
                                               f'Dm{s_idx+1}_Sv{i_idx+1}')

    logger.solve(f"Alloc_ji: 서버 i에 할당된 서비스 j의 용량, 총 {len(Alloc)}개 생성")
    # 모든 변수를 출력하기는 너무 많을 수 있으므로, 일부만 예시로 출력하거나 요약
    if len(Alloc) > 10:  # 변수가 많을 경우 일부만 출력
        logger.solve(
            f"  (예시) X_s{demand_data[0].get('id', 0)}_i{server_data[0].get('id', 0)}, X_s{demand_data[0].get('id', 0)}_i{server_data[1].get('id', 1)}, ...")
    else:
        for (s_idx, i_idx), var in Alloc.items():
            logger.solve(
                f"  - {var.name()} (서비스: {demand_data[s_idx].get('id', s_idx)}, 서버: {server_data[i_idx].get('id', i_idx)}), 범위: [{var.lb()}, {var.ub()}]")
    logger.solve(f"Created {len(svr_name)} Sv[] variables and {len(Alloc)} Dm[]_Sv[] variables.")

    # --- 제약 조건 ---
    logger.solve("\n**제약 조건:**")

    # 1. 총 예산, 전력, 공간 제약
    total_budget_constraint = solver.Constraint(0, global_constraints.get('total_budget', infinity), 'total_budget')
    total_power_constraint = solver.Constraint(0, global_constraints.get('total_power_kva', infinity), 'total_power')
    total_space_constraint = solver.Constraint(0, global_constraints.get('total_space_sqm', infinity), 'total_space')
    for i in range(num_server_data):
        total_budget_constraint.SetCoefficient(svr_name[i], server_data[i].get('cost', 0))
        total_power_constraint.SetCoefficient(svr_name[i], server_data[i].get('power_kva', 0))
        total_space_constraint.SetCoefficient(svr_name[i], server_data[i].get('space_sqm', 0))

    budget_terms= []
    power_terms= []
    space_terms = []
    for i in range(num_server_data):
        cost = server_data[i].get('cost', 0)
        power = server_data[i].get('power_kva', 0)
        space = server_data[i].get('space_sqm', 0)
        if cost != 0:
            budget_terms.append(f"{cost}*{svr_name[i].name()}")
        if power != 0:
            power_terms.append(f"{power}*{svr_name[i].name()}")
        if space != 0:
            space_terms.append(f"{space}*{svr_name[i].name()}")
    budget_expr_str = " + ".join(budget_terms)
    power_expr_str = " + ".join(power_terms)
    space_expr_str = " + ".join(space_terms)
    logger.solve(
        f"total_budget: {total_budget_constraint.lb()} <= {budget_expr_str} <= {total_budget_constraint.ub()}")
    logger.solve(
        f"total_power: {total_power_constraint.lb()} <= {power_expr_str} <= {total_power_constraint.ub()}")
    logger.solve(
        f"total_space: {total_space_constraint.lb()} <= {space_expr_str} <= {total_space_constraint.ub()}")

    # 4. 각 자원(CPU, RAM, Storage)에 대한 용량 제약
    # 각 서버 i가 제공하는 총 CPU = Ns[i] * server_data[i]['cpu_cores']
    # 각 서비스 s의 유닛이 요구하는 CPU = demands_data[s]['req_cpu_cores']
    # 총 요구 CPU = sum over s,i (X_si[s,i] * demands_data[s]['req_cpu_cores'])
    # 이는 잘못된 접근. X_si는 서비스 s를 서버 i에서 몇 유닛 제공하는지.
    # 서버 i에 할당된 서비스들의 총 요구 자원이 서버 i의 총 제공 자원을 넘을 수 없음.

    # 수정된 제약: 각 서버 i에 대해, 해당 서버에 할당된 모든 서비스의 자원 요구량 합계는
    # 해당 서버의 총 구매된 용량을 초과할 수 없음.
    resource_types = ['cpu_cores', 'ram_gb', 'storage_tb']
    for i_idx in range(num_server_data):  # 각 서버에 대해
        server_type = server_data[i_idx]
        for res_idx, resource in enumerate(resource_types):  # 각 자원 유형에 대해
            # # 서버 i가 제공하는 총 자원량
            # # Ns[i_idx] * server.get(resource, 0)
            # # 서버 i에 할당된 모든 서비스 유닛들이 소모하는 총 자원량
            # # sum (X_si[s_idx, i_idx] * demands_data[s_idx].get(f'req_{resource}', 0) for s_idx in range(num_services))
            # constraint_res = solver.Constraint(-infinity, 0, f'res_{resource}server{i_idx}')
            # # 제공량 (우변으로 넘기면 <= 0)
            # constraint_res.SetCoefficient(Ns[i_idx], -server.get(resource, 0))  # 제공량은 음수로
            # # 소비량 (좌변에 그대로)
            # for s_idx in range(num_services):
            #     if (s_idx, i_idx) in X_si:  # 해당 변수가 존재할 때만
            #         service = demands_data[s_idx]
            #         constraint_res.SetCoefficient(X_si[s_idx, i_idx], service.get(f'req_{resource}', 0))  # 소비량은 양수로
            # logger.solve(f"Added resource constraint for {resource} on server type {server.get('id', i_idx)}.")

            # 제약 조건을 생성하기 전에 정의된 변수가 존재하는지 확인
            coeffs = []
            terms = []
            # Ns[i_idx] * server.get(resource, 0) 만큼의 자원 제공
            coeffs.append(-server_type.get(resource, 0))
            terms.append(svr_name[i_idx])

            for s_idx in range(num_demand_data):
                if (s_idx, i_idx) in Alloc:
                    service = demand_data[s_idx]
                    req_resource = service.get(f'req_{resource}', 0)
                    coeffs.append(req_resource)
                    terms.append(Alloc[s_idx, i_idx])

            # 모든 계수가 0이 아닌 경우에만 제약을 추가 (자원 요구사항이 없는 경우 불필요)
            if any(c != 0 for c in coeffs):
                constraint_expr = solver.Sum(terms[j] * coeffs[j] for j in range(len(terms)))
                constraint_name = f'req_{resource}_{server_type.get("id", i_idx)}'
                # sum(X_si[s,i] * req_res[s]) <= Ns[i] * server_res[i] 형태로 표현 가능
                # 즉, sum(X_si[s,i] * req_res[s]) - Ns[i] * server_res[i] <= 0
                constraint = solver.Add(constraint_expr <= 0, constraint_name)
                logger.solve(f"{constraint.name()}: {constraint_expr} <= 0")

    # 5. 각 서비스의 최대 수요(유닛) 제약 (선택 사항, X_si 변수 상한으로 이미 반영됨)
    # sum over i (X_si[s,i]) <= demands_data[s]['max_units'] (또는 == nếu 정확히 수요 충족)
    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        max_units_s = service.get('max_units')
        if max_units_s is not None and max_units_s != infinity:
            # 서비스 s에 대해 모든 서버에서 제공되는 총 유닛 수는 max_units_s를 넘을 수 없음
            constraint_demand_s = solver.Constraint(0, max_units_s, f'demand_service_{s_idx}')
            for i_idx in range(num_server_data):
                if (s_idx, i_idx) in Alloc:
                    constraint_demand_s.SetCoefficient(Alloc[s_idx, i_idx], 1)
            # logger.solve(f"service_{service.get('id', s_idx)}: sum(X_si[{service.get('id', s_idx)},i]) <= {max_units_s}")
            logger.solve(f"service_{service.get('id', s_idx)}: sum(Dm[{s_idx},i]) <= {max_units_s}")
    # --- 목표 함수 ---
    # 총 이익 = (각 서비스 유닛 수익 합계) - (총 서버 구매 비용)
    objective = solver.Objective()
    # 서버 구매 비용 (음수)
    for i in range(num_server_data):
        objective.SetCoefficient(svr_name[i], -server_data[i].get('cost', 0))

    # 서비스 수익 (양수)
    for s_idx in range(num_demand_data):
        service = demand_data[s_idx]
        for i_idx in range(num_server_data):
            if (s_idx, i_idx) in Alloc:
                objective.SetCoefficient(Alloc[s_idx, i_idx], service.get('revenue_per_unit', 0))

    objective.SetMaximization()
    logger.solve(f"\n**목표 함수:** 총 이익 극대화 (서비스 수익 - 서버 구매 비용)")
    logger.solve(f"  목표: Maximize sum(X_si * revenue_per_unit) - sum(Ns * cost)")

    # --- 문제 해결 ---
    logger.info("Solving Data Center Capacity model...")
    solve_start_time = datetime.datetime.now()
    status = solver.Solve()
    solve_end_time = datetime.datetime.now()
    processing_time_ms = (solve_end_time - solve_start_time).total_seconds() * 1000
    logger.info(f"Solver status: {status}, Time: {processing_time_ms:.2f} ms")

    # --- 결과 추출 ---
    results = {
        'purchased_servers': [],
        'service_allocations': [],
        'total_profit': 0,
        'total_server_cost': 0,
        'total_service_revenue': 0,
        'total_power_used': 0,
        'total_space_used': 0,
    }
    error_msg = None

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        if status == pywraplp.Solver.FEASIBLE:
            logger.warning("Feasible solution found for data center plan, but it might not be optimal.")
            # error_msg 설정은 선택사항

        results['total_profit'] = round(solver.Objective().Value(), 2)

        current_total_server_cost = 0
        current_total_power = 0
        current_total_space = 0
        for i in range(num_server_data):
            num_purchased = svr_name[i].solution_value()
            if abs(num_purchased) < 1e-6: num_purchased = 0  # 부동소수점 정리
            num_purchased = int(round(num_purchased))  # 정수 변수이므로 반올림

            if num_purchased > 0:
                server_type = server_data[i]
                results['purchased_servers'].append({
                    'type_id': server_type.get('id', f'Type{i}'),
                    'count': num_purchased,
                    'unit_cost': server_type.get('cost', 0),
                    'total_cost_for_type': round(num_purchased * server_type.get('cost', 0), 2)
                })
                current_total_server_cost += num_purchased * server_type.get('cost', 0)
                current_total_power += num_purchased * server_type.get('power_kva', 0)
                current_total_space += num_purchased * server_type.get('space_sqm', 0)

        results['total_server_cost'] = round(current_total_server_cost, 2)
        results['total_power_used'] = round(current_total_power, 2)
        results['total_space_used'] = round(current_total_space, 2)

        current_total_service_revenue = 0
        service_details = []
        for s_idx in range(num_demand_data):
            service = demand_data[s_idx]
            total_units_for_service_s = 0
            allocation_details_s = []
            for i_idx in range(num_server_data):
                if (s_idx, i_idx) in Alloc:
                    units_on_server_i = Alloc[s_idx, i_idx].solution_value()
                    if abs(units_on_server_i) < 1e-6: units_on_server_i = 0
                    units_on_server_i = int(round(units_on_server_i))

                    if units_on_server_i > 0:
                        total_units_for_service_s += units_on_server_i
                        allocation_details_s.append({
                            'server_type_id': server_data[i_idx].get('id', f'Type{i_idx}'),
                            'units_allocated': units_on_server_i
                        })

            if total_units_for_service_s > 0:
                service_revenue_s = total_units_for_service_s * service.get('revenue_per_unit', 0)
                current_total_service_revenue += service_revenue_s
                service_details.append({
                    'service_id': service.get('id', f'Service{s_idx}'),
                    'total_units_provided': total_units_for_service_s,
                    'revenue_from_service': round(service_revenue_s, 2),
                    'allocations': allocation_details_s
                })
        results['service_allocations'] = service_details
        results['total_service_revenue'] = round(current_total_service_revenue, 2)

        # 최종 이익 확인 (솔버 목표값과 수동 계산 일치 여부)
        manual_profit = results['total_service_revenue'] - results['total_server_cost']
        logger.info(f"Solver Objective (Profit): {results['total_profit']}, Manual Calc Profit: {manual_profit:.2f}")


    else:  # OPTIMAL 또는 FEASIBLE이 아닌 경우
        solver_status_map = {
            pywraplp.Solver.INFEASIBLE: "실행 불가능한 문제입니다. 제약 조건(예산, 전력, 공간, 자원 요구량 등)을 확인하세요.",
            pywraplp.Solver.UNBOUNDED: "목표 함수가 무한합니다. 수익이 비용보다 과도하게 높거나 제약이 누락되었을 수 있습니다.",
            pywraplp.Solver.ABNORMAL: "솔버가 비정상적으로 종료되었습니다.",
            pywraplp.Solver.MODEL_INVALID: "모델이 유효하지 않습니다.",
            pywraplp.Solver.NOT_SOLVED: "솔버가 문제를 풀지 못했습니다."
        }
        error_msg = solver_status_map.get(status, f"최적해를 찾지 못했습니다. (솔버 상태 코드: {status})")
        logger.error(f"Data center capacity solver failed. Status: {status}. Message: {error_msg}")

    return results, error_msg, processing_time_ms

matched_pairs, total_yield, error_msg = run_optimization(global_constraints, server_data, demands_data)

