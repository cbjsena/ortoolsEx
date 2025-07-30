from gurobipy import Model, GRB, quicksum
from ortools.sat.python import cp_model
import datetime
import json
from common_utils.common_run_opt import export_cp_model
from logging_config import setup_logger
import logging
import settings

setup_logger()
logger = logging.getLogger(__name__)

ORTOOLS_TIME_LIMIT =30
GUROBI_TIME_LIMIT = 30
test_type = 2


def run_sports_scheduling_optimizer_ortools1(input_data):
    """
    3가지 다른 목표를 지원하는 Sports Scheduling 최적화 함수.
    """
    schedule_type = input_data.get('schedule_type')
    objective_choice = input_data.get('objective_choice')
    teams = input_data.get('teams', [])
    distance_matrix = input_data.get('distance_matrix')
    max_consecutive = input_data.get('max_consecutive')
    num_teams_original = len(teams)

    num_teams_original = len(teams)
    logger.info(
        f"Running {schedule_type.upper()} Scheduler with OR-Tools-1. Objective: {objective_choice}, Teams: {num_teams_original}")

    if num_teams_original < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    has_bye = False
    if num_teams_original % 2 != 0 and schedule_type == 'single':
        teams.append('BYE')
        has_bye = True
        logger.info("Odd number of teams for single round-robin. Added a BYE team.")

    num_teams = len(teams)

    # 리그 방식에 따라 슬롯(주차) 수 결정
    if schedule_type == 'single':
        num_slots = num_teams - 1
    else:  # double
        num_slots = 2 * (num_teams - 1)

    model = cp_model.CpModel()

    # --- 1. 결정 변수 ---
    # plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1
    plays = {}
    for s in range(num_slots):
        for h in range(num_teams):
            for a in range(num_teams):
                if h != a:
                    var = model.NewBoolVar(f'plays_{s + 1}_{teams[h]}_{teams[a]}')
                    plays[(s, h, a)] = var
                    logger.solve(f"Var: {var.Name()}")

    # CpModel의 Add()는 name 인자를 지원하지 않으므로, 별도로 dict에 저장
    if not hasattr(model, 'named_constraints'):
        model.named_constraints = {}
    # --- 2. 제약 조건 ---
    # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만 (홈 또는 원정)
    for s in range(num_slots):
        for t in range(num_teams):
            home_games = [plays.get((s, t, a), 0) for a in range(num_teams) if t != a]
            away_games = [plays.get((s, h, t), 0) for h in range(num_teams) if t != h]
            constraint = model.AddExactlyOne(home_games + away_games)

            constraint_name = f"OnceSlot_{s + 1}_{teams[t]}"
            model.named_constraints[constraint_name] = constraint
            constraint_expr = " + ".join(var.Name() for var in home_games + away_games) + " == 1"
            logger.solve(f"Eq: {constraint_name}: {constraint_expr}")

    # 제약 2: 리그 방식에 따른 경기 수 제약
    if schedule_type == 'single':
        for h in range(num_teams):
            for a in range(h + 1, num_teams):
                matchups = [plays.get((s, h, a)) for s in range(num_slots)] + [plays.get((s, a, h)) for s in
                                                                               range(num_slots)]
                constraint = model.Add(sum(matchups) == 1)
                constraint_name = f"SinglePair_{teams[h]}_{teams[a]}"
                model.named_constraints[constraint_name] = constraint
                logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in matchups)} == 1")
    else:  # double
        for h in range(num_teams):
            for a in range(num_teams):
                if h != a:
                    home_matchups = [plays.get((s, h, a)) for s in range(num_slots)]
                    constraint = model.Add(sum(home_matchups) == 1)
                    constraint_name = f"DoublePair_{teams[h]}_{teams[a]}"
                    model.named_constraints[constraint_name] = constraint
                    logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in home_matchups)} == 1")

    # 제약 3: 최대 연속 홈/원정 경기 수 제한
    for t_idx in range(num_teams_original):
        for s in range(num_slots - max_consecutive):
            away_games_in_window = [plays.get((i, h, t_idx)) for i in range(s, s + max_consecutive + 1) for h in
                                    range(num_teams) if t_idx != h]
            constraint = model.Add(sum(away_games_in_window) <= max_consecutive)
            constraint_name = f"MaxAway_{teams[t_idx]}_s{s + 1}"
            model.named_constraints[constraint_name] = constraint
            logger.solve(
                f"{constraint_name}: {' + '.join(v.Name() for v in away_games_in_window)} <= {max_consecutive}")

            home_games_in_window = [plays.get((i, t_idx, a)) for i in range(s, s + max_consecutive + 1) for a in
                                    range(num_teams) if t_idx != a]
            constraint = model.Add(sum(home_games_in_window) <= max_consecutive)
            constraint_name = f"MaxHome_{teams[t_idx]}_s{s + 1}"
            model.named_constraints[constraint_name] = constraint
            logger.solve(
                f"{constraint_name}: {' + '.join(v.Name() for v in home_games_in_window)} <= {max_consecutive}")

    # 제약 4: 같은 팀과 연속으로 경기하지 않음
    # for h in range(num_teams):
    #     for a in range(num_teams):
    #         if h != a:
    #             for s in range(num_slots - 1):
    #                 # s주차와 s+1주차에 연속으로 같은 대진이 없도록 함
    #                 match_s = plays.get((s, h, a), 0) + plays.get((s, a, h), 0)
    #                 match_s_plus_1 = plays.get((s + 1, h, a), 0) + plays.get((s + 1, a, h), 0)
    #                 model.Add(match_s + match_s_plus_1 <= 1)

    # 제약 5: 팀별 이동 거리 계산을 위한 제약
    team_travel_vars = [model.NewIntVar(0, 10000000, f'travel_{i}') for i in range(num_teams_original)]
    for t_idx in range(num_teams_original):
        travel_dist_terms = []
        for s in range(num_slots):
            for opponent_idx in range(num_teams_original):
                if t_idx != opponent_idx:
                    # 단순화된 거리 계산: 원정 경기 시, 자신의 홈에서 상대방 홈으로 이동
                    travel_dist_terms.append(
                        distance_matrix[t_idx][opponent_idx] * plays.get((s, opponent_idx, t_idx)))
        model.Add(team_travel_vars[t_idx] == sum(travel_dist_terms))

    # --- 3. 목표 함수 설정 ---
    if objective_choice == 'minimize_travel':
        model.Minimize(sum(team_travel_vars))
        logger.debug("Objective set to: Minimize Total Travel Distance.")
    elif objective_choice == 'fairness':
        # --- 연속 홈/원정 경기 break 변수 추가 ---
        breaks = []
        for t_idx in range(num_teams_original):
            for s in range(num_slots - 1):
                is_home_s = model.NewBoolVar(f'is_home_t{t_idx}_s{s}')
                is_home_s_plus_1 = model.NewBoolVar(f'is_home_t{t_idx}_s{s + 1}')
                break_var = model.NewBoolVar(f'break_t{t_idx}_s{s}')

                # Reification: is_home 변수를 실제 경기 여부와 연결
                model.Add(sum(plays.get((s, t_idx, a)) for a in range(num_teams) if t_idx != a) == 1).OnlyEnforceIf(
                    is_home_s)
                model.Add(sum(plays.get((s, t_idx, a)) for a in range(num_teams) if t_idx != a) == 0).OnlyEnforceIf(
                    is_home_s.Not())

                model.Add(sum(plays.get((s + 1, t_idx, a)) for a in range(num_teams) if t_idx != a) == 1).OnlyEnforceIf(
                    is_home_s_plus_1)
                model.Add(sum(plays.get((s + 1, t_idx, a)) for a in range(num_teams) if t_idx != a) == 0).OnlyEnforceIf(
                    is_home_s_plus_1.Not())

                # Reification: break_var는 두 시점의 홈 여부가 같을 때 1이 됨
                model.Add(is_home_s == is_home_s_plus_1).OnlyEnforceIf(break_var)
                model.Add(is_home_s != is_home_s_plus_1).OnlyEnforceIf(break_var.Not())
                breaks.append(break_var)

        model.Minimize(sum(breaks))
        logger.debug("Objective set to: Minimize(sum of all break variables).")
    elif objective_choice == 'distance_gap':
        min_travel = model.NewIntVar(0, 10000000, 'min_travel')
        max_travel = model.NewIntVar(0, 10000000, 'max_travel')
        model.AddMinEquality(min_travel, team_travel_vars)
        model.AddMaxEquality(max_travel, team_travel_vars)
        model.Minimize(max_travel - min_travel)
        logger.debug("Objective set to: Minimize gap in travel distance (Distance Gap).")
    else:
        # 기본 목표: 특별한 목표 없음 (실행 가능한 해 찾기)
        logger.debug("Objective set to: Find a feasible solution.")

    # --- 4. 문제 해결 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.ORTOOLS_TIME_LIMIT
    logger.info("Solving the Sports Scheduling model...")
    status = solver.Solve(model)
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} sec")

    # --- 5. 결과 추출 ---
    results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 'N/A', 'team_distances': []}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # 대진표 파싱
        schedule = []
        for s in range(num_slots):
            weekly_matchups = []
            for h in range(num_teams):
                for a in range(num_teams):
                    if h != a and (s, h, a) in plays and solver.Value(plays.get((s, h, a), 0)) == 1:
                        weekly_matchups.append({'home': teams[h], 'away': teams[a]})
            schedule.append({'week': s + 1, 'matchups': weekly_matchups})
        results['schedule'] = schedule

        # for key, var in plays.items():
        #     if solver.Value(var) == 1:
        #         logger.solve(var.Name())

        # 결과 지표 계산
        total_dist_calc = 0
        team_distances_calc = []
        for i in range(num_teams_original):
            dist_val = solver.Value(team_travel_vars[i])
            team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
            total_dist_calc += dist_val

        results['total_distance'] = round(total_dist_calc)
        results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
        results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist)
        total_breaks_calc = 0
        for t_idx in range(num_teams_original):
            for s in range(num_slots - 1):
                is_home_s = sum(solver.Value(plays.get((s, t_idx, a), 0)) for a in range(num_teams) if t_idx != a)
                is_home_s_plus_1 = sum(
                    solver.Value(plays.get((s + 1, t_idx, a), 0)) for a in range(num_teams) if t_idx != a)
                if is_home_s == is_home_s_plus_1:
                    total_breaks_calc += 1
        results['total_breaks'] = total_breaks_calc
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Sports Scheduling failed: {error_msg}")

    return results, error_msg, solver.WallTime()


def run_sports_scheduling_optimizer_ortools2(input_data):
    #TODO ortools MODEL 파악
    """
    OR-Tools CP-SAT를 사용하여 Sports Scheduling 문제를 해결합니다.
    Gurobi2 모델과 동일한 로직을 구현합니다.
    """
    schedule_type = input_data.get('schedule_type')
    objective_choice = input_data.get('objective_choice')
    teams = list(input_data.get('teams'))
    team_names_original = list(teams)
    distance_matrix = input_data.get('distance_matrix')
    max_consecutive = input_data.get('max_consecutive')
    num_teams_original = len(teams)

    logger.info(
        f"Running {schedule_type.upper()} Scheduler with OR-Tools-2. Objective: {objective_choice}, Teams: {num_teams_original}")

    if num_teams_original < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    has_bye = False
    if num_teams_original % 2 != 0:
        teams.append('BYE')
        has_bye = True

    num_teams = len(teams)
    num_slots = 2 * (num_teams_original - 1) if schedule_type == 'double' else num_teams - 1
    num_cities = len(distance_matrix)

    team_to_idx = {name: i for i, name in enumerate(teams)}
    original_team_to_idx = {name: i for i, name in enumerate(team_names_original)}
    home_city_of_team = {i: original_team_to_idx.get(teams[i], -1) for i in range(num_teams)}

    try:
        model = cp_model.CpModel()

        # --- 1. 결정 변수 ---
        # Gurobi: plays = model.addVars(num_slots, num_teams, num_teams, vtype=GRB.BINARY, name="plays")
        plays = {}
        for s in range(num_slots):
            for t in range(num_teams):
                for a in range(num_teams):
                    if t != a:
                        var = model.NewBoolVar(f'plays_{s + 1}_{teams[t]}_{teams[a]}')
                        plays[(s, t, a)] = var
                        logger.solve(f"Var: {var.Name()}")
        # Gurobi: is_at_loc = model.addVars(num_teams, num_slots, num_cities, vtype=GRB.BINARY, name="is_at_loc")
        is_at_loc = {}
        for t in range(num_teams_original):
            for s in range(num_slots):
                for l in range(num_cities):
                    var =model.NewBoolVar(f"is_at_loc_{teams[t]}_{s+1}_{l}")
                    is_at_loc[t, s, l] = var
                    logger.solve(f"Var: {var.Name()}")

        # CpModel의 Add()는 name 인자를 지원하지 않으므로, 별도로 dict에 저장
        if not hasattr(model, 'named_constraints'):
            model.named_constraints = {}
        # --- 2. 제약 조건 ---
        # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만 (홈 또는 원정)
        for s in range(num_slots):
            for t in range(num_teams):
                home_games = [plays.get((s, t, a), 0) for a in range(num_teams) if t != a]
                away_games = [plays.get((s, h, t), 0) for h in range(num_teams) if t != h]
                constraint = model.AddExactlyOne(home_games + away_games)

                constraint_name = f"OnceSlot_{s + 1}_{teams[t]}"
                model.named_constraints[constraint_name] = constraint
                constraint_expr = " + ".join(var.Name() for var in home_games + away_games) + " == 1"
                logger.solve(f"Eq: {constraint_name}: {constraint_expr}")

        # 제약 2: 리그 방식에 따른 경기 수 제약
        if schedule_type == 'single':
            for h in range(num_teams):
                for a in range(h + 1, num_teams):
                    matchups = [plays.get((s, h, a)) for s in range(num_slots)] + [plays.get((s, a, h)) for s in
                                                                                   range(num_slots)]
                    constraint = model.Add(sum(matchups) == 1)
                    constraint_name = f"SinglePair_{teams[h]}_{teams[a]}"
                    model.named_constraints[constraint_name] = constraint
                    logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in matchups)} == 1")
        else:  # double
            for h in range(num_teams):
                for a in range(num_teams):
                    if h != a:
                        home_matchups = [plays.get((s, h, a)) for s in range(num_slots)]
                        constraint = model.Add(sum(home_matchups) == 1)
                        constraint_name = f"DoublePair_{teams[h]}_{teams[a]}"
                        model.named_constraints[constraint_name] = constraint
                        logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in home_matchups)} == 1")

        # 제약 3: 최대 연속 홈/원정 경기 수 제한
        for t_idx in range(num_teams_original):
            for s in range(num_slots - max_consecutive):
                away_games_in_window = [plays.get((i, h, t_idx)) for i in range(s, s + max_consecutive + 1) for h in
                                        range(num_teams) if t_idx != h]
                constraint = model.Add(sum(away_games_in_window) <= max_consecutive)
                constraint_name = f"MaxAway_{teams[t_idx]}_s{s + 1}"
                model.named_constraints[constraint_name] = constraint
                logger.solve(
                    f"{constraint_name}: {' + '.join(v.Name() for v in away_games_in_window)} <= {max_consecutive}")

                home_games_in_window = [plays.get((i, t_idx, a)) for i in range(s, s + max_consecutive + 1) for a in
                                        range(num_teams) if t_idx != a]
                constraint = model.Add(sum(home_games_in_window) <= max_consecutive)
                constraint_name = f"MaxHome_{teams[t_idx]}_s{s + 1}"
                model.named_constraints[constraint_name] = constraint
                logger.solve(
                    f"{constraint_name}: {' + '.join(v.Name() for v in home_games_in_window)} <= {max_consecutive}")

        # 제약 4: 같은 팀과 연속으로 경기하지 않음
        # for h in range(num_teams):
        #     for a in range(num_teams):
        #         if h != a:
        #             for s in range(num_slots - 1):
        #                 # s주차와 s+1주차에 연속으로 같은 대진이 없도록 함
        #                 match_s = plays.get((s, h, a), 0) + plays.get((s, a, h), 0)
        #                 match_s_plus_1 = plays.get((s + 1, h, a), 0) + plays.get((s + 1, a, h), 0)
        #                 model.Add(match_s + match_s_plus_1 <= 1)

        # 제약 4: 팀 위치 결정
        for s in range(num_slots):
            for t in range(num_teams_original):
                # 홈 경기 시 위치
                is_home = model.NewBoolVar(f"is_home_{t}_{s}")
                model.Add(is_home == sum(plays[s, t, a] for a in range(num_teams) if t != a))
                model.AddImplication(is_home, is_at_loc[t, s, home_city_of_team[t]])

                # 원정 경기 시 위치
                for h in range(num_teams_original):
                    if t != h:
                        model.AddImplication(plays[s, h, t], is_at_loc[t, s, home_city_of_team[h]])

            # 각 팀은 한 슬롯에 한 곳에만 위치
            for t in range(num_teams_original):
                model.Add(sum(is_at_loc[t, s, l] for l in range(num_cities)) == 1)

        # --- 4. 목표 함수 설정 ---
        # Gurobi: team_travel_vars = model.addVars(...)
        max_dist = sum(max(row) for row in distance_matrix) * num_slots
        team_travel_vars = [model.NewIntVar(0, max_dist, f'team_travel_{t}') for t in range(num_teams_original)]

        # 동적 이동 거리 계산
        for t in range(num_teams_original):
            slot_travel_dist = []

            for s in range(num_slots):
                # 현재 슬롯의 위치 변수
                curr_loc_vars = [is_at_loc[t, s, l] for l in range(num_cities)]

                # --- 슬롯 0의 이동 거리 계산 ---
                if s == 0:
                    # 첫 경기는 무조건 홈에서 시작하므로, 이전 위치는 홈 도시임
                    home_city_idx = home_city_of_team[t]
                    # 이 경우 (상수 * 변수) 이므로 이미 선형(linear)임
                    dist_for_slot = sum(
                        curr_loc_vars[l2] * distance_matrix[home_city_idx][l2] for l2 in range(num_cities))
                    slot_travel_dist.append(dist_for_slot)

                # --- 슬롯 1 이상의 이동 거리 계산 ---
                else:
                    # 이전 슬롯의 위치 변수
                    prev_loc_vars = [is_at_loc[t, s - 1, l] for l in range(num_cities)]

                    # 이동을 나타내는 변수 선형화 (Z <=> X AND Y)
                    travel_arc_vars_for_slot = {}
                    dist_expressions = []

                    for l1 in range(num_cities):
                        for l2 in range(num_cities):
                            # l1 -> l2 로 이동했으면 1, 아니면 0인 변수
                            z = model.NewBoolVar(f'travel_{t}_{s}_{l1}_{l2}')
                            x = prev_loc_vars[l1]
                            y = curr_loc_vars[l2]

                            # Z <=> (X AND Y) 관계를 강제
                            # 1. Z => X and Z => Y
                            model.AddImplication(z, x)
                            model.AddImplication(z, y)
                            # 2. (X AND Y) => Z
                            model.Add(x + y <= z + 1)

                            # 이동이 발생했다면(z=1), 해당 거리를 더함
                            dist_expressions.append(z * distance_matrix[l1][l2])
                    slot_travel_dist.append(sum(dist_expressions))

            model.Add(team_travel_vars[t] == sum(slot_travel_dist))

        if objective_choice == 'minimize_travel':
            # Gurobi: model.setObjective(quicksum(team_travel_vars), GRB.MINIMIZE)
            total_travel = model.NewIntVar(0, max_dist * num_teams_original, 'total_travel')
            model.Add(total_travel == sum(team_travel_vars))
            model.Minimize(total_travel)
            logger.info("Objective set to: Minimize total travel.")
        elif objective_choice == 'fairness':
            break_vars = []  # break가 발생하면 1이 되는 변수들을 담을 리스트

            for t in range(num_teams_original):
                for s in range(num_slots - 1):
                    # 슬롯 s와 s+1에서의 홈 경기 여부 (합계로 계산하면 0 또는 1이 됨)
                    is_home_s = sum(plays[s, t, a] for a in range(num_teams) if t != a)
                    is_home_s_plus_1 = sum(plays[s + 1, t, a] for a in range(num_teams) if t != a)

                    # 두 상태의 차이 (-1, 0, 1)
                    diff = model.NewIntVar(-1, 1, f'diff_{t}_{s}')
                    model.Add(diff == is_home_s - is_home_s_plus_1)

                    # 차이의 절댓값. 상태가 바뀌면 1, 유지되면 0. (즉, 'no break' 변수)
                    abs_diff = model.NewBoolVar(f'abs_diff_{t}_{s}')
                    model.AddAbsEquality(abs_diff, diff)

                    # [수정된 핵심 로직] break_var를 'break 발생'과 동일하게 정의
                    # break는 abs_diff가 0일 때 발생하므로, break_var = 1 - abs_diff
                    break_var = model.NewBoolVar(f'break_{team_names_original[t]}_{s+1}')
                    model.Add(break_var != abs_diff)  # (break_var = 1 - abs_diff 와 동일)

                    break_vars.append(break_var)
            total_breaks = sum(break_vars)
            # break 발생 횟수의 총합을 최소화
            model.Add(total_breaks<=5)
            model.Minimize(total_breaks)
            logger.debug("Objective set to: Minimize total number of breaks.")

        elif objective_choice == 'distance_gap':
            min_travel = model.NewIntVar(0, 1000000, "min_travel")
            max_travel = model.NewIntVar(0, 1000000, "max_travel")
            # Gurobi: model.addGenConstrMin(min_travel, team_travel_vars)
            model.AddMinEquality(min_travel, team_travel_vars)
            # Gurobi: model.addGenConstrMax(max_travel, team_travel_vars)
            model.AddMaxEquality(max_travel, team_travel_vars)

            # Gurobi: model.setObjective(max_travel - min_travel, GRB.MINIMIZE)
            model.Minimize(max_travel - min_travel)
            logger.debug("Objective set to: Minimize distance gap.")
        problem_type =input_data.get('problem_type')
        export_cp_model(model, f'{problem_type}.mps')
        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        # Gurobi: model.setParam('TimeLimit', ...)
        solver.parameters.max_time_in_seconds = settings.ORTOOLS_TIME_LIMIT

        solve_start_time = datetime.datetime.now()
        status = solver.Solve(model)
        solve_end_time = datetime.datetime.now()
        processing_time_ms = (solve_end_time - solve_start_time).total_seconds()

        # --- 6. 결과 추출 ---
        results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 0, 'team_distances': [], 'total_breaks': 0,
                   'distance_gap': 0}
        error_msg = None

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = []
            for s in range(num_slots):
                weekly_matchups = []
                for h in range(num_teams):
                    for a in range(num_teams):
                        if h != a and (s, h, a) in plays and solver.Value(plays.get((s, h, a), 0)) == 1:
                            weekly_matchups.append({'home': teams[h], 'away': teams[a]})
                schedule.append({'week': s + 1, 'matchups': weekly_matchups})
            results['schedule'] = schedule

            for key, var in plays.items():
                if solver.Value(var) == 1:
                    logger.solve(var.Name())
            if objective_choice == 'fairness':
                for var in break_vars:
                    if solver.Value(var) == 1:
                        logger.solve(var.Name())

            # 결과 지표 계산
            total_dist_calc = 0
            team_distances_calc = []
            for i in range(num_teams):
                dist_val = solver.Value(team_travel_vars[i])
                team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
                total_dist_calc += dist_val

            results['total_distance'] = round(total_dist_calc)
            results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
            results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
            dist = [item['distance'] for item in results['team_distances']]
            results['distance_gap'] = max(dist) - min(dist)
            total_breaks_calc = 0
            for t_idx in range(num_teams_original):
                for s in range(num_slots - 1):
                    is_home_s = sum(solver.Value(plays.get((s, t_idx, a), 0)) for a in range(num_teams) if t_idx != a)
                    is_home_s_plus_1 = sum(
                        solver.Value(plays.get((s + 1, t_idx, a), 0)) for a in range(num_teams) if t_idx != a)
                    if is_home_s == is_home_s_plus_1:
                        total_breaks_calc += 1
            results['total_breaks'] = total_breaks_calc

            if status == cp_model.FEASIBLE:
                results['time_limit'] = f"Solver is limited {settings.ORTOOLS_TIME_LIMIT} sec. (sub-optimal solution)"

        else:
            error_msg = f"OR-Tools 솔버가 해를 찾지 못했습니다. (상태: {solver.StatusName(status)})"

    except Exception as e:
        logger.error(f"Error using OR-Tools: {e}", exc_info=True)
        error_msg = "OR-Tools 솔버 사용 중 오류가 발생했습니다."
        processing_time_ms = 0

    return results, error_msg, processing_time_ms


def run_sports_scheduling_optimizer_gurobi1(input_data):
    """
    Gurobi를 사용하여 Sports Scheduling 문제를 해결합니다.
    """
    schedule_type = input_data.get('schedule_type', 'double')
    objective_choice = input_data.get('objective_choice', 'fairness')
    teams = list(input_data.get('teams', []))
    distance_matrix_km = input_data.get('distance_matrix', [])
    max_consecutive = input_data.get('max_consecutive', 3)
    num_teams_original = len(teams)

    logger.info(
        f"Running {schedule_type.upper()} Scheduler with Gurobi-1. Objective: {objective_choice}, Teams: {num_teams_original}")

    if num_teams_original < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    has_bye = False
    if num_teams_original % 2 != 0:
        teams.append('BYE')
        has_bye = True

    num_teams = len(teams)
    num_slots = 2 * (num_teams_original - 1) if schedule_type == 'double' else num_teams - 1

    try:
        # --- 1. Gurobi 모델 생성 ---
        model = Model("SportsScheduling")

        # --- 2. 결정 변수 생성 ---
        # plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1
        plays = model.addVars(num_slots, num_teams, num_teams, vtype=GRB.BINARY, name="plays")

        # --- 3. 제약 조건 설정 ---
        # 존재하지 않는 경기 변수 제거 (같은 팀 간의 경기)
        for s in range(num_slots):
            for t in range(num_teams):
                plays[s, t, t].UB = 0

        # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만
        for s in range(num_slots):
            for t in range(num_teams):
                model.addConstr(
                    quicksum(plays[s, t, a] for a in range(num_teams) if t != a) +
                    quicksum(plays[s, h, t] for h in range(num_teams) if t != h) == 1,
                    name=f"PlayOnce_s{s + 1}_t{t + 1}"
                )

        # 제약 2: 리그 방식에 따른 경기 수 제약
        if schedule_type == 'single':
            for h in range(num_teams):
                for a in range(h + 1, num_teams):
                    model.addConstr(quicksum(plays[s, h, a] + plays[s, a, h] for s in range(num_slots)) == 1)
        else:  # double
            for h in range(num_teams):
                for a in range(num_teams):
                    if h != a:
                        model.addConstr(quicksum(plays[s, h, a] for s in range(num_slots)) == 1)

        # 제약 3: 최대 연속 홈/원정 경기 수 제한
        for t in range(num_teams_original):
            for s in range(num_slots - max_consecutive):
                model.addConstr(quicksum(
                    plays[i, h, t] for i in range(s, s + max_consecutive + 1) for h in range(num_teams) if
                    t != h) <= max_consecutive)
                model.addConstr(quicksum(
                    plays[i, t, a] for i in range(s, s + max_consecutive + 1) for a in range(num_teams) if
                    t != a) <= max_consecutive)

        # --- 4. 목표 함수 설정 ---
        team_travel_vars = model.addVars(num_teams_original, vtype=GRB.INTEGER, name="team_travel")

        for t_idx in range(num_teams_original):
            model.addConstr(
                team_travel_vars[t_idx] == quicksum(
                    distance_matrix_km[t_idx][opp_idx] * plays.get((s, opp_idx, t_idx), 0)
                    for s in range(num_slots) for opp_idx in range(num_teams_original) if t_idx != opp_idx
                )
            )

        if objective_choice == 'minimize_travel':
            model.setObjective(quicksum(team_travel_vars[t] for t in range(num_teams_original)), GRB.MINIMIZE)

        elif objective_choice == 'fairness':  # 연속 경기 최소화
            breaks = model.addVars(num_teams_original, num_slots - 1, vtype=GRB.BINARY, name="breaks")
            for t in range(num_teams_original):
                for s in range(num_slots - 1):
                    is_home_s = quicksum(plays[s, t, a] for a in range(num_teams) if t != a)
                    is_home_s_plus_1 = quicksum(plays[s + 1, t, a] for a in range(num_teams) if t != a)
                    # Gurobi에서는 is_home_s == is_home_s_plus_1 같은 직접 비교가 제약에 사용됨
                    # break_var = 1 if is_home_s == is_home_s_plus_1
                    model.addGenConstrIndicator(breaks[t, s], True, is_home_s - is_home_s_plus_1, GRB.EQUAL, 0.0)
            model.setObjective(quicksum(breaks), GRB.MINIMIZE)

        elif objective_choice == 'distance_gap':
            min_travel = model.addVar(vtype=GRB.INTEGER, name="min_travel")
            max_travel = model.addVar(vtype=GRB.INTEGER, name="max_travel")
            model.addGenConstrMin(min_travel, team_travel_vars)
            model.addGenConstrMax(max_travel, team_travel_vars)
            model.setObjective(max_travel - min_travel, GRB.MINIMIZE)

        # --- 5. 문제 해결 ---
        model.setParam('TimeLimit', settings.GUROBI_TIME_LIMIT)  # 시간 제한 10초
        solve_start_time = datetime.datetime.now()
        model.optimize()
        solve_end_time = datetime.datetime.now()
        processing_time_ms = (solve_end_time - solve_start_time).total_seconds()

        # --- 6. 결과 추출 ---
        results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 0, 'team_distances': [], 'total_breaks': 0,
                   'distance_gap': 0}
        error_msg = None

        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            # 대진표 파싱
            schedule = []
            for s in range(num_slots):
                weekly_matchups = []
                for h in range(num_teams):
                    for a in range(num_teams):
                        if h != a and (s, h, a) in plays and plays[s, h, a].X > 0.5:
                            weekly_matchups.append({'home': teams[h], 'away': teams[a]})
                schedule.append({'week': s + 1, 'matchups': weekly_matchups})
            results['schedule'] = schedule

            # for key, var in plays.items():
            #     if var.X > 0.5:
            #         logger.solve(var.VarName)

            # 결과 지표 계산
            total_dist_calc = 0
            team_distances_calc = []
            for i in range(num_teams):
                dist_val = team_travel_vars[i].X
                team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
                total_dist_calc += dist_val

            results['total_distance'] = round(total_dist_calc)
            results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
            results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
            dist = [item['distance'] for item in results['team_distances']]
            results['distance_gap'] = max(dist) - min(dist)
            total_breaks_calc = 0
            for t_idx in range(num_teams_original):
                for s in range(num_slots - 1):
                    is_home_s = sum(plays.get((s, t_idx, a), 0).X for a in range(num_teams) if t_idx != a)
                    is_home_s_plus_1 = sum(
                        plays.get((s + 1, t_idx, a), 0).X for a in range(num_teams) if t_idx != a)
                    if is_home_s == is_home_s_plus_1:
                        total_breaks_calc += 1
            results['total_breaks'] = total_breaks_calc
            if model.status == GRB.TIME_LIMIT:
                timeLimit = model.getParamInfo('TimeLimit')[2]
                results['time_limit'] =f"Solver is limited {timeLimit} sec."

        else:
            error_msg = "Gurobi 솔버가 해를 찾지 못했습니다."

    except Exception as e:
        logger.error(f"Error using Gurobi: {e}", exc_info=True)
        error_msg = "Gurobi 솔버 사용 중 오류가 발생했습니다. 라이선스 또는 설치를 확인하세요."
        processing_time_ms = 0

    return results, error_msg, processing_time_ms


def run_sports_scheduling_optimizer_gurobi2(input_data):
    # TODO gurobi MODEL 파악
    """
    Gurobi를 사용하여 Sports Scheduling 문제를 해결합니다.
    """
    schedule_type = input_data.get('schedule_type', 'double')
    objective_choice = input_data.get('objective_choice', 'fairness')
    teams = list(input_data.get('teams', []))
    team_names_original = list(teams)   # BYE 추가 전 원본 팀 이름 저장
    distance_matrix = input_data.get('distance_matrix', [])
    max_consecutive = input_data.get('max_consecutive', 3)
    num_teams_original = len(teams)

    logger.info(
        f"Running {schedule_type.upper()} Scheduler with Gurobi-2. Objective: {objective_choice}, Teams: {num_teams_original}")

    if num_teams_original < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    has_bye = False
    if num_teams_original % 2 != 0:
        teams.append('BYE')
        has_bye = True

    num_teams = len(teams)
    num_slots = 2 * (num_teams_original - 1) if schedule_type == 'double' else num_teams - 1
    num_cities = len(distance_matrix)

    # 팀 이름 -> 인덱스, 홈 도시 인덱스 매핑
    team_to_idx = {name: i for i, name in enumerate(teams)}
    original_team_to_idx = {name: i for i, name in enumerate(team_names_original)}
    home_city_of_team = {i: original_team_to_idx.get(teams[i], -1) for i in range(num_teams)}  # -1 for BYE team

    try:
        model = Model("AdvancedSportsScheduling")
        # model.setParam('OutputFlag', 0)  # Gurobi 로그 숨기기

        # --- 2. 결정 변수 ---
        plays = model.addVars(num_slots, num_teams, num_teams, vtype=GRB.BINARY, name="plays")
        is_at_loc = model.addVars(num_teams, num_slots, num_cities, vtype=GRB.BINARY, name="is_at_loc")

        # --- 3. 제약 조건 ---
        model.addConstrs((plays[s, t, t] == 0 for s in range(num_slots) for t in range(num_teams)), "no_self_play")

        # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만
        model.addConstrs((quicksum(plays[s, t, a] for a in range(num_teams) if t != a) +
                          quicksum(plays[s, h, t] for h in range(num_teams) if t != h) == 1
                          for s in range(num_slots) for t in range(num_teams)), "play_once_per_slot")

        # 제약 2: 리그 방식
        if schedule_type == 'single':
            model.addConstrs((quicksum(plays[s, h, a] + plays[s, a, h] for s in range(num_slots)) == 1
                              for h in range(num_teams) for a in range(h + 1, num_teams)), "single_round_robin")
        else:  # double
            model.addConstrs((quicksum(plays[s, h, a] for s in range(num_slots)) == 1
                              for h in range(num_teams) for a in range(num_teams) if h != a), "double_round_robin")

        # 제약 3: 최대 연속 경기
        for t in range(num_teams_original):
            for s in range(num_slots - max_consecutive):
                model.addConstr(quicksum(
                    plays[i, h, t] for i in range(s, s + max_consecutive + 1) for h in range(num_teams_original) if
                    t != h) <= max_consecutive)
                model.addConstr(quicksum(
                    plays[i, t, a] for i in range(s, s + max_consecutive + 1) for a in range(num_teams_original) if
                    t != a) <= max_consecutive)

        # [NEW] 제약 4: 팀 위치 결정
        for s in range(num_slots):
            for t in range(num_teams_original):
                # 팀 t가 홈 경기 시, 위치는 자신의 홈
                is_home = quicksum(plays[s, t, a] for a in range(num_teams) if t != a)
                model.addConstr(is_at_loc[t, s, home_city_of_team[t]] >= is_home)
                # 팀 t가 원정 경기 시, 위치는 상대팀 h의 홈
                for h in range(num_teams_original):
                    if t != h:
                        model.addConstr(is_at_loc[t, s, home_city_of_team[h]] >= plays[s, h, t])
            # 각 팀은 한 슬롯에 한 곳에만 위치
            for t in range(num_teams_original):
                model.addConstr(quicksum(is_at_loc[t, s, l] for l in range(num_cities)) == 1)

        # --- 4. 목표 함수 설정 ---
        team_travel_vars = model.addVars(num_teams_original, vtype=GRB.INTEGER, name="team_travel")

        # [NEW] 동적 이동 거리 계산
        for t in range(num_teams_original):
            slot_travel_dist = []
            # [NEW] s=0: introduce binary variables for initial location
            initial_loc_vars = model.addVars(num_cities, vtype=GRB.BINARY, name=f"init_loc_{t}")
            for l in range(num_cities):
                if l == home_city_of_team[t]:
                    model.addConstr(initial_loc_vars[l] == 1)
                else:
                    model.addConstr(initial_loc_vars[l] == 0)
            for s in range(num_slots):
                if s == 0:
                    prev_loc_vars = [initial_loc_vars[l] for l in range(num_cities)]
                else:
                    prev_loc_vars = [is_at_loc[t, s - 1, l] for l in range(num_cities)]
                curr_loc_vars = [is_at_loc[t, s, l] for l in range(num_cities)]

                # dist_in_slot[s] = sum over l1,l2 ( prev_loc_vars[l1] * curr_loc_vars[l2] * distance[l1][l2] )
                # 위 식은 변수 간의 곱이므로 선형이 아님. 선형화 필요.
                # Gurobi에서는 and_ 제약으로 선형화 가능
                travel_arc_vars = model.addVars(num_cities, num_cities, vtype=GRB.BINARY, name=f"travel_{t}_{s}")
                for l1 in range(num_cities):
                    for l2 in range(num_cities):
                        # Z = X AND Y
                        model.addGenConstrAnd(travel_arc_vars[l1, l2], [prev_loc_vars[l1], curr_loc_vars[l2]])

                dist_for_slot = quicksum(
                    travel_arc_vars[l1, l2] * distance_matrix[l1][l2] for l1 in range(num_cities) for l2 in
                    range(num_cities))
                slot_travel_dist.append(dist_for_slot)

            model.addConstr(team_travel_vars[t] == quicksum(slot_travel_dist))

        if objective_choice == 'minimize_travel':
            model.setObjective(quicksum(team_travel_vars[t] for t in range(num_teams_original)), GRB.MINIMIZE)
            logger.debug("Objective set to: Minimize total travel.")
        elif objective_choice == 'fairness':  # 연속 경기 최소화
            breaks = model.addVars(num_teams_original, num_slots - 1, vtype=GRB.BINARY, name="breaks")
            for t in range(num_teams_original):
                for s in range(num_slots - 1):
                    # is_home_s: 팀 t가 시간 s에 홈이면 1
                    is_home_s = quicksum(plays[s, t, a] for a in range(num_teams) if t != a)
                    # is_home_s_plus_1: 팀 t가 시간 s+1에 홈이면 1
                    is_home_s_plus_1 = quicksum(plays[s + 1, t, a] for a in range(num_teams) if t != a)

                    # diff = is_home_s - is_home_s_plus_1 (값은 -1, 0, 1)
                    diff = model.addVar(lb=-1, ub=1, vtype=GRB.INTEGER, name=f"diff_{t}_{s}")
                    model.addConstr(diff == is_home_s - is_home_s_plus_1)

                    # abs_diff = |diff|. abs_diff는 0 또는 1 (break가 없으면 1, 있으면 0)
                    abs_diff = model.addVar(vtype=GRB.BINARY, name=f"abs_diff_{t}_{s}")
                    model.addGenConstrAbs(abs_diff, diff)

                    # breaks[t,s]는 abs_diff의 반대. breaks = 1 - abs_diff
                    model.addConstr(breaks[t, s] == 1 - abs_diff)

            model.setObjective(quicksum(breaks.values()), GRB.MINIMIZE)
            logger.debug("Objective set to: Minimize total number of breaks.")

        elif objective_choice == 'distance_gap':
            min_travel = model.addVar(vtype=GRB.INTEGER, name="min_travel")
            max_travel = model.addVar(vtype=GRB.INTEGER, name="max_travel")
            model.addGenConstrMin(min_travel, team_travel_vars)
            model.addGenConstrMax(max_travel, team_travel_vars)
            model.setObjective(max_travel - min_travel, GRB.MINIMIZE)
            logger.debug("Objective set to: Minimize distance gap.")
        # --- 5. 문제 해결 ---
        model.setParam('TimeLimit', settings.GUROBI_TIME_LIMIT)
        solve_start_time = datetime.datetime.now()
        model.optimize()
        solve_end_time = datetime.datetime.now()
        processing_time_ms = (solve_end_time - solve_start_time).total_seconds()

        # --- 6. 결과 추출 ---
        results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 0, 'team_distances': [], 'total_breaks': 0,
                   'distance_gap': 0}
        error_msg = None

        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            # 대진표 파싱
            schedule = []
            for s in range(num_slots):
                weekly_matchups = []
                for h in range(num_teams):
                    for a in range(num_teams):
                        if h != a and (s, h, a) in plays and plays[s, h, a].X > 0.5:
                            weekly_matchups.append({'home': teams[h], 'away': teams[a]})
                schedule.append({'week': s + 1, 'matchups': weekly_matchups})
            results['schedule'] = schedule

            for key, var in plays.items():
                if var.X > 0.5:
                    logger.solve(f'plays_{key[0]+1}_{team_names_original[key[1]]}_{team_names_original[key[2]]}')
            for key, var in breaks.items():
                if var.X > 0.5:
                    logger.solve(f'breaks_{key[0]+1}_{team_names_original[key[1]]}')

            # 결과 지표 계산
            total_dist_calc = 0
            team_distances_calc = []
            for i in range(num_teams):
                dist_val = team_travel_vars[i].X
                team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
                total_dist_calc += dist_val

            results['total_distance'] = round(total_dist_calc)
            results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
            results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
            dist = [item['distance'] for item in results['team_distances']]
            results['distance_gap'] = max(dist) - min(dist)
            total_breaks_calc = 0
            for t_idx in range(num_teams_original):
                for s in range(num_slots - 1):
                    is_home_s = sum(plays.get((s, t_idx, a), 0).X for a in range(num_teams) if t_idx != a)
                    is_home_s_plus_1 = sum(
                        plays.get((s + 1, t_idx, a), 0).X for a in range(num_teams) if t_idx != a)
                    if is_home_s == is_home_s_plus_1:
                        total_breaks_calc += 1
            results['total_breaks'] = total_breaks_calc
            if model.status == GRB.TIME_LIMIT:
                timeLimit = model.getParamInfo('TimeLimit')[2]
                results['time_limit'] = f"Solver is limited {timeLimit} sec."

        else:
            error_msg = "Gurobi 솔버가 해를 찾지 못했습니다."

    except Exception as e:
        logger.error(f"Error using Gurobi: {e}", exc_info=True)
        error_msg = "Gurobi 솔버 사용 중 오류가 발생했습니다. 라이선스 또는 설치를 확인하세요."
        processing_time_ms = 0

    return results, error_msg, processing_time_ms


if test_type ==1:
    with open('../test_data/puzzles_sports_scheduling_data/minimize_travel_single_team4.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    results_data, error_msg_opt, processing_time_ms = run_sports_scheduling_optimizer_ortools1(input_data)
    logger.info(f"total_distance:{results_data['total_distance']} "
                f"distance_gap: {results_data['distance_gap']}, "
                f"total_breaks: {results_data['total_breaks']}")
elif test_type == 2:
    with open('../test_data/puzzles_sports_scheduling_data/test.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f) #fairness_single_team4 minimize_travel_single_team4
    results_data, error_msg_opt, processing_time_ms = run_sports_scheduling_optimizer_ortools2(input_data)
    logger.info(f"total_distance:{results_data['total_distance']} "
                f"distance_gap: {results_data['distance_gap']}, "
                f"total_breaks: {results_data['total_breaks']}")
elif test_type == 3:
    with open('../test_data/puzzles_sports_scheduling_data/fairness_single_team4.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f) #fairness_single_team4 minimize_travel_single_team4
    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_gurobi1(input_data)
    logger.info(f"Running Time:{processing_time} "
                f"total_distance:{results_data['total_distance']} "
                f"distance_gap: {results_data['distance_gap']}, "
                f"total_breaks: {results_data['total_breaks']}")
elif test_type == 4:
    with open('../test_data/puzzles_sports_scheduling_data/fairness_single_team4.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f) #fairness_single_team4 minimize_travel_single_team4
    results_data, error_msg_opt, processing_time = run_sports_scheduling_optimizer_gurobi2(input_data)
    logger.info(f"Running Time:{processing_time} "
                f"total_distance:{results_data['total_distance']} "
                f"distance_gap: {results_data['distance_gap']}, "
                f"total_breaks: {results_data['total_breaks']}")