import json
import logging
from gurobipy import Model, GRB, quicksum  # Gurobi 사용을 위해 import


from common_utils.ortools_solvers import BaseOrtoolsCpSolver

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. 팩토리 클래스 (Factory)
# --------------------------------------------------------------------------
class SportsSolverFactory:
    """
    입력 데이터에 따라 적절한 스포츠 스케줄링 솔버 인스턴스를 생성하고 반환합니다.
    """

    def __init__(self, input_data):
        solver_type = input_data.get('solver_type', 'ORTOOLS')
        num_teams = len(input_data.get('teams', []))

        if solver_type == 'GUROBI':
            logger.info("Using Gurobi solver for sports scheduling.")
            # if num_teams <= 5:
            #     self.solver = GurobiComplexSolver(input_data)
            # else:
            #     self.solver = GurobiSimpleSolver(input_data)
        else:  # 기본값은 OR-Tools
            if num_teams <= 5:
                self.solver = OrtoolsComplexSolver(input_data)
            else:
                self.solver = OrtoolsSimpleSolver(input_data)

        logger.info(f"Solver selected: {self.solver.__class__.__name__} for {num_teams} teams.")

    def solve(self):
        """선택된 솔버의 solve 메서드를 호출합니다."""
        return self.solver.solve()


# --------------------------------------------------------------------------
# 2. 기본 클래스 계층 구조
# --------------------------------------------------------------------------

class BaseSportsSchedulingSolver:
    """모든 스포츠 스케줄링 문제의 공통 데이터만 처리."""

    def __init__(self, input_data, **kwargs):
        # super() 체인을 위해 kwargs를 사용합니다.
        super().__init__(input_data, **kwargs)

        self.input_data = input_data
        self.schedule_type = input_data.get('schedule_type')
        self.objective_choice = input_data.get('objective_choice')
        self.teams_original = list(input_data.get('teams', []))
        self.distance_matrix = input_data.get('distance_matrix', [])
        self.max_consecutive = input_data.get('max_consecutive', 3)

        self.num_teams_original = len(self.teams_original)
        self.teams = list(self.teams_original)
        self.has_bye = False

        if self.num_teams_original % 2 != 0:
            self.teams.append('BYE')
            self.has_bye = True
            logger.info("Odd number of teams for single round-robin. Added a BYE team.")

        self.num_teams = len(self.teams)

        if self.schedule_type == 'single':
            self.num_slots = self.num_teams - 1
        else:
            self.num_slots = 2 * (self.num_teams_original - 1)

        self.num_cities = len(self.distance_matrix)


class BaseOrtoolsSportsSolver(BaseSportsSchedulingSolver, BaseOrtoolsCpSolver):
    """OR-Tools CP-SAT 솔버를 위한 공통 로직을 포함하는 기본 클래스."""

    def __init__(self, input_data):
        # 부모 클래스들의 생성자를 모두 호출
        super().__init__(input_data)
        self.plays = {}
        self.team_travel_vars = []


    def _create_variables_plays(self):
        """plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1 """
        for s in range(self.num_slots):
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a:
                        self.plays[(s, h, a)] = self.model.NewBoolVar(f'plays_s{s}_h{h}_a{a}')

    def _add_one_game_per_slot_constraint(self):
        """제약: 각 팀은 각 슬롯에서 정확히 한 경기만 수행합니다."""
        for s in range(self.num_slots):
            for t in range(self.num_teams):
                home_games = [self.plays.get((s, t, a), 0) for a in range(self.num_teams) if t != a]
                away_games = [self.plays.get((s, h, t), 0) for h in range(self.num_teams) if t != h]
                self.model.AddExactlyOne(home_games + away_games)

    def _add_league_format_constraint(self):
        """제약: 리그 방식(single/double)에 따른 총 경기 수를 설정합니다."""
        if self.schedule_type == 'single':
            for h in range(self.num_teams):
                for a in range(h + 1, self.num_teams):
                    matchups = [self.plays.get((s, h, a), 0) for s in range(self.num_slots)] + \
                               [self.plays.get((s, a, h), 0) for s in range(self.num_slots)]
                    self.model.Add(sum(matchups) == 1)
        else:  # double
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a:
                        self.model.Add(sum(self.plays.get((s, h, a), 0) for s in range(self.num_slots)) == 1)

    def _add_max_consecutive_games_constraint(self):
        """제약: 최대 연속 홈/원정 경기 수를 제한합니다."""
        for t in range(self.num_teams_original):
            for s in range(self.num_slots - self.max_consecutive):
                home_window = [self.plays.get((i, t, a), 0) for i in range(s, s + self.max_consecutive + 1) for a in range(self.num_teams) if t != a]
                away_window = [self.plays.get((i, h, t), 0) for i in range(s, s + self.max_consecutive + 1) for h in range(self.num_teams) if t != h]
                self.model.Add(sum(home_window) <= self.max_consecutive)
                self.model.Add(sum(away_window) <= self.max_consecutive)


    def _add_common_constraints(self):
        """ 모든 공통 제약 함수를 호출하는 메인 함수입니다."""
        self._add_one_game_per_slot_constraint()
        self._add_league_format_constraint()
        self._add_max_consecutive_games_constraint()



class BaseGurobiSportsSolver(BaseSportsSchedulingSolver):
    """Gurobi 솔버를 위한 공통 로직을 포함하는 기본 클래스."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = Model("SportsScheduling")
        self.plays = {}

    def _create_variables(self):
        self.plays = self.model.addVars(self.num_slots, self.num_teams, self.num_teams, vtype=GRB.BINARY, name="plays")
        for s in range(self.num_slots):
            for t in range(self.num_teams):
                self.plays[s, t, t].UB = 0

    def _add_common_constraints(self):
        # 각 팀은 각 슬롯에서 정확히 한 경기만
        for s in range(self.num_slots):
            for t in range(self.num_teams):
                self.model.addConstr(
                    quicksum(self.plays[s, t, a] for a in range(self.num_teams) if t != a) +
                    quicksum(self.plays[s, h, t] for h in range(self.num_teams) if t != h) == 1
                )
        # 리그 방식
        if self.schedule_type == 'single':
            for h in range(self.num_teams):
                for a in range(h + 1, self.num_teams):
                    self.model.addConstr(
                        quicksum(self.plays[s, h, a] + self.plays[s, a, h] for s in range(self.num_slots)) == 1)
        else:  # double
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a:
                        self.model.addConstr(quicksum(self.plays[s, h, a] for s in range(self.num_slots)) == 1)
        # 최대 연속 경기
        for t in range(self.num_teams_original):
            for s in range(self.num_slots - self.max_consecutive):
                self.model.addConstr(quicksum(
                    self.plays[i, h, t] for i in range(s, s + self.max_consecutive + 1) for h in range(self.num_teams)
                    if t != h) <= self.max_consecutive)
                self.model.addConstr(quicksum(
                    self.plays[i, t, a] for i in range(s, s + self.max_consecutive + 1) for a in range(self.num_teams)
                    if t != a) <= self.max_consecutive)


# --------------------------------------------------------------------------
# 3. 구체적인 솔버 클래스 구현
# --------------------------------------------------------------------------

class OrtoolsSimpleSolver(BaseOrtoolsSportsSolver):
    """6개 팀 이상을 위한 간단한 OR-Tools 모델."""
    def __init__(self, input_data):
        super().__init__(input_data)


    def _create_variables(self):
        super()._create_variables_plays()

    def _add_constraints(self):
        super()._add_common_constraints()

    def _set_objective_function(self):
        self.team_travel_vars = [self.model.NewIntVar(0, 10000000, f'travel_{idx}') for idx in range(self.num_teams_original)]
        for t_idx in range(self.num_teams_original):
            travel_dist_terms = []
            for s in range(self.num_slots):
                for opponent_idx in range(self.num_teams_original):
                    if t_idx != opponent_idx:
                        # 단순화된 거리 계산: 원정 경기 시, 자신의 홈에서 상대방 홈으로 이동
                        travel_dist_terms.append(
                            self.distance_matrix[t_idx][opponent_idx] * self.plays.get((s, opponent_idx, t_idx)))
            self.model.Add(self.team_travel_vars[t_idx] == sum(travel_dist_terms))

        # --- 3. 목표 함수 설정 ---
        if self.objective_choice == 'minimize_travel':
            self.model.Minimize(sum(self.team_travel_vars))
            logger.debug("Objective set to: Minimize Total Travel Distance.")
        elif self.objective_choice == 'fairness':
            # --- 연속 홈/원정 경기 break 변수 추가 ---
            breaks = []
            for t_idx in range(self.num_teams_original):
                for s in range(self.num_slots - 1):
                    is_home_s = self.model.NewBoolVar(f'is_home_t{t_idx}_s{s}')
                    is_home_s_plus_1 = self.model.NewBoolVar(f'is_home_t{t_idx}_s{s + 1}')
                    break_var = self.model.NewBoolVar(f'break_t{t_idx}_s{s}')

                    # Reification: is_home 변수를 실제 경기 여부와 연결
                    self.model.Add(sum(self.plays.get((s, t_idx, a)) for a in range(self.num_teams) if t_idx != a) == 1).OnlyEnforceIf(
                        is_home_s)
                    self.model.Add(sum(self.plays.get((s, t_idx, a)) for a in range(self.num_teams) if t_idx != a) == 0).OnlyEnforceIf(
                        is_home_s.Not())

                    self.model.Add(
                        sum(self.plays.get((s + 1, t_idx, a)) for a in range(self.num_teams) if t_idx != a) == 1).OnlyEnforceIf(
                        is_home_s_plus_1)
                    self.model.Add(
                        sum(self.plays.get((s + 1, t_idx, a)) for a in range(self.num_teams) if t_idx != a) == 0).OnlyEnforceIf(
                        is_home_s_plus_1.Not())

                    # Reification: break_var는 두 시점의 홈 여부가 같을 때 1이 됨
                    self.model.Add(is_home_s == is_home_s_plus_1).OnlyEnforceIf(break_var)
                    self.model.Add(is_home_s != is_home_s_plus_1).OnlyEnforceIf(break_var.Not())
                    breaks.append(break_var)

            self.model.Minimize(sum(breaks))
            logger.debug("Objective set to: Minimize(sum of all break variables).")
        elif self.objective_choice == 'distance_gap':
            min_travel = self.model.NewIntVar(0, 10000000, 'min_travel')
            max_travel = self.model.NewIntVar(0, 10000000, 'max_travel')
            self.model.AddMinEquality(min_travel, self.team_travel_vars)
            self.model.AddMaxEquality(max_travel, self.team_travel_vars)
            self.model.Minimize(max_travel - min_travel)
            logger.debug("Objective set to: Minimize gap in travel distance (Distance Gap).")
        else:
            # 기본 목표: 특별한 목표 없음 (실행 가능한 해 찾기)
            logger.debug("Objective set to: Find a feasible solution.")

    def _extract_results(self, solver):
        results = {'schedule': [], 'has_bye': self.has_bye, 'total_distance': 'N/A', 'team_distances': []}

        schedule = []
        for s in range(self.num_slots):
            weekly_matchups = []
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a and (s, h, a) in self.plays and solver.Value(self.plays.get((s, h, a), 0)) == 1:
                        weekly_matchups.append({'home': self.teams[h], 'away': self.teams[a]})
            schedule.append({'week': s + 1, 'matchups': weekly_matchups})
        results['schedule'] = schedule

        # 결과 지표 계산
        total_dist_calc = 0
        team_distances_calc = []
        for idx in range(self.num_teams_original):
            dist_val = solver.Value(self.team_travel_vars[idx])
            team_distances_calc.append({'name': self.input_data['teams'][idx], 'distance': round(dist_val)})
            total_dist_calc += dist_val

        results['total_distance'] = round(total_dist_calc)
        results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
        results['team_distances'].sort(key=lambda x: x['name'])  # 팀 이름순 정렬
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist)
        total_breaks_calc = 0
        for t_idx in range(self.num_teams_original):
            for s in range(self.num_slots - 1):
                is_home_s = sum(solver.Value(self.plays.get((s, t_idx, a), 0)) for a in range(self.num_teams) if t_idx != a)
                is_home_s_plus_1 = sum(
                    solver.Value(self.plays.get((s + 1, t_idx, a), 0)) for a in range(self.num_teams) if t_idx != a)
                if is_home_s == is_home_s_plus_1:
                    total_breaks_calc += 1
        results['total_breaks'] = total_breaks_calc

        return results

class OrtoolsComplexSolver(BaseOrtoolsSportsSolver):
    """BaseOrtoolsSportsSolver를 상속받아, 복잡한 모델에만 필요한 변수와 제약 조건을 추가"""
    def __init__(self, input_data):
        super().__init__(input_data)
        self.is_at_loc = {}
        self.break_vars = []

        self.original_team_to_idx = {name: idx for idx, name in enumerate(self.teams_original)}
        self.home_city_of_team = {idx: self.original_team_to_idx.get(self.teams[idx], -1) for idx in range(self.num_teams)}

    def _create_variables_is_at_loc(self):
        for t in range(self.num_teams_original):
            for s in range(self.num_slots):
                for l in range(self.num_cities):
                    var = self.model.NewBoolVar(f"is_at_loc_{self.teams[t]}_{s + 1}_{l}")
                    self.is_at_loc[t, s, l] = var
                    # logger.solve(f"Var: {var.Name()}")

    def _add_constraints_location(self):
        # 제약 4: 팀 위치 결정
        for s in range(self.num_slots):
            for t in range(self.num_teams_original):
                # 홈 경기 시 위치
                is_home = self.model.NewBoolVar(f"is_home_{t}_{s}")
                self.model.Add(is_home == sum(self.plays[s, t, a] for a in range(self.num_teams) if t != a))
                self.model.AddImplication(is_home, self.is_at_loc[t, s, self.home_city_of_team[t]])

                # 원정 경기 시 위치
                for h in range(self.num_teams_original):
                    if t != h:
                        self.model.AddImplication(self.plays[s, h, t], self.is_at_loc[t, s, self.home_city_of_team[h]])

            # 각 팀은 한 슬롯에 한 곳에만 위치
            for t in range(self.num_teams_original):
                self.model.Add(sum(self.is_at_loc[t, s, l] for l in range(self.num_cities)) == 1)


    def _create_variables(self):
        super()._create_variables_plays()
        self._create_variables_is_at_loc()


    def _add_constraints(self):
        super()._add_common_constraints()
        self._add_constraints_location()


    def _set_objective_function(self):
        """목표 함수 (총비용 최소화)를 설정합니다."""
        max_dist = sum(max(row) for row in self.distance_matrix) * self.num_slots
        team_travel_vars = [self.model.NewIntVar(0, max_dist, f'team_travel_{t}') for t in range(self.num_teams_original)]

        # 동적 이동 거리 계산
        for t in range(self.num_teams_original):
            slot_travel_dist = []

            for s in range(self.num_slots):
                # 현재 슬롯의 위치 변수
                curr_loc_vars = [self.is_at_loc[t, s, l] for l in range(self.num_cities)]

                # --- 슬롯 0의 이동 거리 계산 ---
                if s == 0:
                    # 첫 경기는 무조건 홈에서 시작하므로, 이전 위치는 홈 도시임
                    home_city_idx = self.home_city_of_team[t]
                    # 이 경우 (상수 * 변수) 이므로 이미 선형(linear)임
                    dist_for_slot = sum(
                        curr_loc_vars[l2] * self.distance_matrix[home_city_idx][l2] for l2 in range(self.num_cities))
                    slot_travel_dist.append(dist_for_slot)

                # --- 슬롯 1 이상의 이동 거리 계산 ---
                else:
                    # 이전 슬롯의 위치 변수
                    prev_loc_vars = [self.is_at_loc[t, s - 1, l] for l in range(self.num_cities)]

                    # 이동을 나타내는 변수 선형화 (Z <=> X AND Y)
                    dist_expressions = []

                    for l1 in range(self.num_cities):
                        for l2 in range(self.num_cities):
                            # l1 -> l2 로 이동했으면 1, 아니면 0인 변수
                            z = self.model.NewBoolVar(f'travel_{t}_{s}_{l1}_{l2}')
                            x = prev_loc_vars[l1]
                            y = curr_loc_vars[l2]

                            # Z <=> (X AND Y) 관계를 강제
                            # 1. Z => X and Z => Y
                            self.model.AddImplication(z, x)
                            self.model.AddImplication(z, y)
                            # 2. (X AND Y) => Z
                            self.model.Add(x + y <= z + 1)

                            # 이동이 발생했다면(z=1), 해당 거리를 더함
                            dist_expressions.append(z * self.distance_matrix[l1][l2])
                    slot_travel_dist.append(sum(dist_expressions))

            self.model.Add(team_travel_vars[t] == sum(slot_travel_dist))

        if self.objective_choice == 'minimize_travel':
            # Gurobi: model.setObjective(quicksum(team_travel_vars), GRB.MINIMIZE)
            total_travel = self.model.NewIntVar(0, max_dist * self.num_teams_original, 'total_travel')
            self.model.Add(total_travel == sum(team_travel_vars))
            self.model.Minimize(total_travel)
            logger.info("Objective set to: Minimize total travel.")
        elif self.objective_choice == 'fairness':
            break_vars = []  # break가 발생하면 1이 되는 변수들을 담을 리스트

            for t in range(self.num_teams_original):
                for s in range(self.num_slots - 1):
                    # 슬롯 s와 s+1에서의 홈 경기 여부 (합계로 계산하면 0 또는 1이 됨)
                    is_home_s = sum(self.plays[s, t, a] for a in range(self.num_teams) if t != a)
                    is_home_s_plus_1 = sum(self.plays[s + 1, t, a] for a in range(self.num_teams) if t != a)

                    # 두 상태의 차이 (-1, 0, 1)
                    diff = self.model.NewIntVar(-1, 1, f'diff_{t}_{s}')
                    self.model.Add(diff == is_home_s - is_home_s_plus_1)

                    # 차이의 절댓값. 상태가 바뀌면 1, 유지되면 0. (즉, 'no break' 변수)
                    abs_diff = self.model.NewBoolVar(f'abs_diff_{t}_{s}')
                    self.model.AddAbsEquality(abs_diff, diff)

                    # [수정된 핵심 로직] break_var를 'break 발생'과 동일하게 정의
                    # break는 abs_diff가 0일 때 발생하므로, break_var = 1 - abs_diff
                    break_var = self.model.NewBoolVar(f'break_{self.teams_original[t]}_{s+1}')
                    self.model.Add(break_var != abs_diff)  # (break_var = 1 - abs_diff 와 동일)

                    break_vars.append(break_var)
            total_breaks = sum(break_vars)
            # break 발생 횟수의 총합을 최소화
            self.model.Add(total_breaks<=5)
            self.model.Minimize(total_breaks)
            logger.debug("Objective set to: Minimize total number of breaks.")

        elif self.objective_choice == 'distance_gap':
            min_travel = self.model.NewIntVar(0, 1000000, "min_travel")
            max_travel = self.model.NewIntVar(0, 1000000, "max_travel")
            # Gurobi: model.addGenConstrMin(min_travel, team_travel_vars)
            self.model.AddMinEquality(min_travel, team_travel_vars)
            # Gurobi: model.addGenConstrMax(max_travel, team_travel_vars)
            self.model.AddMaxEquality(max_travel, team_travel_vars)

            # Gurobi: model.setObjective(max_travel - min_travel, GRB.MINIMIZE)
            self.model.Minimize(max_travel - min_travel)
            logger.debug("Objective set to: Minimize distance gap.")

    def _extract_results(self, solver):
        results = {'schedule': [], 'has_bye': self.has_bye, 'total_distance': 0, 'team_distances': [], 'total_breaks': 0,
                   'distance_gap': 0}

        schedule = []
        for s in range(self.num_slots):
            matchups = []
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a and self.plays.get((s, h, a)) is not None and solver.Value(self.plays[(s, h, a)]) > 0.5:
                        matchups.append({'home': self.teams[h], 'away': self.teams[a]})
            schedule.append({'week': s + 1, 'matchups': matchups})
        results['schedule'] = schedule

        # for key, var in self.plays.items():
        #     if solver.Value(var) == 1:
        #         logger.solve(var.Name())
        # if self.objective_choice == 'fairness':
        #     for var in self.break_vars:
        #         if solver.Value(var) == 1:
        #             logger.solve(var.Name())

        # 결과 지표 계산
        team_distances = []
        for idx in range(self.num_teams_original):
            dist_val = solver.Value(self.team_travel_vars[idx]) if self.team_travel_vars else 0
            team_distances.append({'name': self.teams_original[idx], 'distance': round(dist_val)})

        results['team_distances'] = sorted(team_distances, key=lambda x: x['name'])
        results['total_distance'] = round(sum(team_distances))
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist)
        total_breaks_calc = 0
        for t_idx in range(self.num_teams_original):
            for s in range(self.num_slots - 1):
                is_home_s = sum(
                    solver.Value(self.plays.get((s, t_idx, a), 0)) for a in range(self.num_teams) if t_idx != a)
                is_home_s_plus_1 = sum(
                    solver.Value(self.plays.get((s + 1, t_idx, a), 0)) for a in range(self.num_teams) if t_idx != a)
                if is_home_s == is_home_s_plus_1:
                    total_breaks_calc += 1
        results['total_breaks'] = total_breaks_calc

        return results

with open('../test_data/puzzles_sports_scheduling_data/test.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

solver_instance = SportsSolverFactory(input_data)
results_data, error_msg_opt, processing_time = solver_instance.solve()