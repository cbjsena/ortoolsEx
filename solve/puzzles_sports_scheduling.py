from ortools.sat.python import cp_model
import json
from logging_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

def run_sports_scheduling_optimizer1(input_data):
    logger.info(f"Running Sports Scheduling Optimizer. Objective: {input_data.get('objective_choice')}")

    teams = input_data.get('teams', [])
    distance_matrix = input_data.get('distance_matrix', [])
    objective_choice = input_data.get('objective_choice', 'fairness')  # 기본 목표는 공정성
    num_teams = len(teams)

    if num_teams < 2:
        return None, "오류: 최소 2개 팀이 필요합니다.", 0.0

    # 팀 수가 홀수이면 'BYE' 추가 (이동 거리 계산에서는 제외)
    has_bye = False
    if num_teams % 2 != 0:
        teams.append('BYE')
        # 거리 행렬에도 BYE 팀을 위한 행/열 추가 (거리는 0)
        new_size = num_teams + 1
        new_dist_matrix = [[0] * new_size for _ in range(new_size)]
        for i in range(num_teams):
            for j in range(num_teams):
                new_dist_matrix[i][j] = distance_matrix[i][j]
        distance_matrix = new_dist_matrix
        num_teams += 1
        has_bye = True
        logger.info("Odd number of teams. Added a BYE team.")

    num_slots = num_teams - 1

    model = cp_model.CpModel()

    # --- 1. 결정 변수 ---
    # plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1
    plays = {}
    for s in range(num_slots):
        for h in range(num_teams):
            for a in range(num_teams):
                if h != a:
                    var = model.NewBoolVar(f'plays_{s+1}_{teams[h]}_{teams[a]}')
                    plays[(s, h, a)] = var
                    logger.solve(f"Created variable: plays[{s+1}, {teams[h]}, {teams[a]}]")

    # --- 2. 제약 조건 ---
    # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만 (홈 또는 원정)
    for s in range(num_slots):
        for t in range(num_teams):
            home_games = [plays[(s, t, a)] for a in range(num_teams) if t != a]
            away_games = [plays[(s, h, t)] for h in range(num_teams) if t != h]
            # 제약 이름 생성
            constraint_name = f"Once_per_Slot_{s}_{teams[t]}"
            # 제약 추가 및 변수에 저장
            constraint = model.Add(sum(home_games) + sum(away_games) == 1)
            # CpModel의 Add()는 name 인자를 지원하지 않으므로, 별도로 dict에 저장
            if not hasattr(model, 'named_constraints'):
                model.named_constraints = {}
            model.named_constraints[constraint_name] = constraint
            home_vars = [f"plays[{s + 1}, {teams[t]}, {teams[a]}]" for a in range(num_teams) if t != a]
            away_vars = [f"plays[{s + 1}, {teams[h]}, {teams[t]}]" for h in range(num_teams) if t != h]
            constraint_expr = " + ".join(
                [f"plays[{s + 1}, {teams[s]}, {teams[t]}] + plays[{s + 1}, {teams[t]}, {teams[s]}]" for s in
                 range(num_slots)]
            ) + " == 1"
            logger.solve(f"{constraint_name}: {constraint_expr}")

    # 제약 2: 각 팀 쌍은 시즌 전체에 걸쳐 정확히 한 번만 만남 (홈/원정 구분 없이)
    for h in range(num_teams):
        for a in range(h + 1, num_teams):

            # 제약 이름 생성
            constraint_name = f"Round_Robin_{teams[h]}_{teams[a]}"
            # 제약 추가 및 변수에 저장
            constraint = model.Add(sum(plays[(s, h, a)] + plays[(s, a, h)] for s in range(num_slots)) == 1)
            model.named_constraints[constraint_name] = constraint
            constraint_expr = " + ".join(
                [f"plays[{s + 1}, {teams[h]}, {teams[a]}] + plays[{s + 1}, {teams[a]}, {teams[h]}]" for s in
                 range(num_slots)]
            ) + " == 1"
            logger.solve(f"{constraint_name}: {constraint_expr}")


    # --- 3. 목표 함수 설정 ---
    if objective_choice == 'minimize_travel':
        # 총 이동 거리 최소화
        total_distance = model.NewIntVar(0, 10000000, 'total_distance')  # 충분히 큰 값

        # last_loc[t, s]: 시간 s-1에 팀 t의 위치 (현재 위치)
        # current_loc[t, s]: 시간 s에 팀 t의 위치 (다음 위치)
        # 이 모델에서는 각 팀의 이전 위치를 추적해야 함.
        # 더 간단한 접근: 각 원정 경기에 대한 이동 거리의 합을 최소화
        distance_terms = []
        for s in range(num_slots):
            for h in range(num_teams):
                for a in range(num_teams):
                    if h != a and teams[a] != 'BYE' and teams[h] != 'BYE':  # BYE 팀은 이동 거리 0
                        # 원정팀(a)이 홈팀(h)으로 이동하는 거리
                        distance_terms.append(distance_matrix[a][h] * plays[(s, h, a)])

        model.Add(total_distance == sum(distance_terms))
        logger.solve(f"Distance terms variables: {[str(term) for term in distance_terms]}")
        logger.solve(f"Total distance variable: {total_distance}")
        model.Minimize(total_distance)
        logger.debug("Objective set to: Minimize Total Travel Distance.")
    else:
        # 기본 목표: 특별한 목표 없음 (실행 가능한 해 찾기)
        logger.debug("Objective set to: Find a feasible solution (fairness).")

    # --- 4. 문제 해결 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    logger.info("Solving the Sports Scheduling model...")
    status = solver.Solve(model)
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} sec")

    # --- 5. 결과 추출 ---
    results = {'schedule': [], 'has_bye': has_bye}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = []
        for s in range(num_slots):
            weekly_matchups = []
            for t1 in range(num_teams):
                for t2 in range(num_teams):
                     if (s, t1, t2) in plays and solver.Value(plays[(s, t1, t2)]) == 1:
                        # BYE 팀이 포함된 경기는 '휴식'으로 표시
                        if teams[t1] == 'BYE':
                            weekly_matchups.append((teams[t2], 'BYE'))
                        elif teams[t2] == 'BYE':
                            weekly_matchups.append((teams[t1], 'BYE'))
                        else:
                            weekly_matchups.append((teams[t1], teams[t2]))
            schedule.append({'week': s + 1, 'matchups': weekly_matchups})
        results['schedule'] = schedule

        for key, var in plays.items():
            if solver.Value(var) == 1:
                logger.solve(var.Name())

        if objective_choice == 'minimize_travel':
            results['total_distance'] = round(solver.ObjectiveValue())  # km 단위로 환산
            logger.solve(f"Total distance: {results['total_distance']} km")
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Sports Scheduling failed: {error_msg}")

    return results, error_msg, solver.WallTime()


def run_sports_scheduling_optimizer2(input_data):
    schedule_type = input_data.get('schedule_type', 'double')
    objective_choice = input_data.get('objective_choice', 'fairness')
    teams = input_data.get('teams', [])
    distance_matrix = input_data.get('distance_matrix', [])
    num_teams_original = len(teams)

    logger.info(
        f"Running {schedule_type.upper()} Round-Robin Scheduler. Objective: {objective_choice}, Teams: {num_teams_original}")

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
                    plays[(s, h, a)] = model.NewBoolVar(f'plays_s{s}_h{h}_a{a}')

    # --- 2. 제약 조건 ---
    # 제약 1: 각 팀은 각 슬롯에서 정확히 한 경기만
    for s in range(num_slots):
        for t in range(num_teams):
            home_games = [plays.get((s, t, a), 0) for a in range(num_teams) if t != a]
            away_games = [plays.get((s, h, t), 0) for h in range(num_teams) if t != h]
            model.AddExactlyOne(home_games + away_games)

    # 제약 2: 리그 방식에 따른 경기 수 제약
    for h in range(num_teams):
        for a in range(h + 1, num_teams):
            if schedule_type == 'single':
                # 싱글 라운드 로빈: 시즌 전체에 걸쳐 두 팀은 정확히 한 번 만남 (홈/원정 무관)
                model.Add(sum(plays.get((s, h, a), 0) + plays.get((s, a, h), 0) for s in range(num_slots)) == 1)
            else:  # double
                # 더블 라운드 로빈: 각 팀의 홈에서 정확히 한 번씩 만남
                model.Add(sum(plays.get((s, h, a), 0) for s in range(num_slots)) == 1)
                model.Add(sum(plays.get((s, a, h), 0) for s in range(num_slots)) == 1)

    # (선택적 제약) 제약 3: 같은 팀과 연속으로 경기하지 않음
    for h in range(num_teams):
        for a in range(num_teams):
            if h != a:
                for s in range(num_slots - 1):
                    # s주차와 s+1주차에 연속으로 같은 대진이 없도록 함
                    match_s = plays.get((s, h, a), 0) + plays.get((s, a, h), 0)
                    match_s_plus_1 = plays.get((s + 1, h, a), 0) + plays.get((s + 1, a, h), 0)
                    model.Add(match_s + match_s_plus_1 <= 1)

    # 제약 3: 팀별 이동 거리 변수 생성
    team_travel_vars = [model.NewIntVar(0, 10000000, f'travel_{i}') for i in range(num_teams)]

    for t_idx in range(num_teams):
        travel_dist = []
        for s in range(num_slots):
            for opponent_idx in range(num_teams):
                if t_idx == opponent_idx:
                    continue
                # t_idx팀이 원정(away)일 때의 이동 거리: opponent_idx -> t_idx (이전 위치를 고려해야 더 정확함)
                # 단순화된 모델: 원정 경기 시, 자신의 홈에서 상대방 홈으로 이동한다고 가정
                # t_idx가 원정팀(a)으로 opponent_idx(h)와 경기할 때의 이동 거리
                travel_dist.append(distance_matrix[t_idx][opponent_idx] * plays[(s, opponent_idx, t_idx)])
        model.Add(team_travel_vars[t_idx] == sum(travel_dist))

    # --- 3. 목표 함수 설정 ---
    if objective_choice == 'minimize_travel':
        logger.debug("Objective set to: Minimize Total Travel Distance.")
        model.Minimize(sum(team_travel_vars))
        logger.debug("Objective set to: Minimize Total Travel Distance.")
    elif objective_choice == 'fairness':
        min_travel = model.NewIntVar(0, 10000000, 'min_travel')
        max_travel = model.NewIntVar(0, 10000000, 'max_travel')
        model.AddMinEquality(min_travel, team_travel_vars)
        model.AddMaxEquality(max_travel, team_travel_vars)
        model.Minimize(max_travel - min_travel)
        logger.debug("Objective set to: Minimize difference in travel distance (Fairness).")
    else:
        # 기본 목표: 특별한 목표 없음 (실행 가능한 해 찾기)
        logger.debug("Objective set to: Find a feasible solution (fairness).")

    # --- 4. 문제 해결 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    logger.info("Solving the Sports Scheduling model...")
    status = solver.Solve(model)
    logger.info(f"Solver status: {status}, Time: {solver.WallTime():.2f} sec")

    # --- 5. 결과 추출 ---
    results = {'schedule': [], 'has_bye': has_bye, 'total_distance': 'N/A', 'team_distances': []}
    error_msg = None

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = []
        total_dist_calc = 0
        for s in range(num_slots):
            weekly_matchups = []
            for t1 in range(num_teams):
                for t2 in range(num_teams):
                    if (s, t1, t2) in plays and solver.Value(plays[(s, t1, t2)]) == 1:
                        # BYE 팀이 포함된 경기는 '휴식'으로 표시
                        if teams[t1] == 'BYE':
                            weekly_matchups.append((teams[t2], 'BYE'))
                        elif teams[t2] == 'BYE':
                            weekly_matchups.append((teams[t1], 'BYE'))
                        else:
                            weekly_matchups.append((teams[t1], teams[t2]))
            schedule.append({'week': s + 1, 'matchups': weekly_matchups})
        results['schedule'] = schedule
        for key, var in plays.items():
            if solver.Value(var) == 1:
                logger.solve(var.Name())
        team_distances_calc = []
        for i in range(num_teams):
            dist_val = solver.Value(team_travel_vars[i])
            team_distances_calc.append({'name': input_data['teams'][i], 'distance': round(dist_val)})
            total_dist_calc += dist_val

        results['total_distance'] = round(total_dist_calc)
        results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
        results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
        dist = [item['distance'] for item in results['team_distances'] ]
        results['max_diff'] = max(dist) - min(dist)
    else:
        error_msg = f"최적 스케줄을 찾지 못했습니다. (솔버 상태: {solver.StatusName(status)})"
        logger.error(f"Sports Scheduling failed: {error_msg}")

    return results, error_msg, solver.WallTime()

test_type = 2
if test_type ==1:
    with open('../test_data/puzzles_sports_scheduling_data/test1.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    results_data, error_msg_opt, processing_time_ms = run_sports_scheduling_optimizer1(input_data)
elif test_type == 2:
    with open('../test_data/puzzles_sports_scheduling_data/test2.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    results_data, error_msg_opt, processing_time_ms = run_sports_scheduling_optimizer2(input_data)
logger.info(results_data)