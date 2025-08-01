import json
import logging
from gurobipy import Model, GRB, quicksum  # Gurobi 사용을 위해 import

from common_utils.gurobi_solvers import BaseGurobiSolver
from common_utils.ortools_solvers import BaseOrtoolsCpSolver
from core.decorators import log_solver_make

from logging_config import setup_logger
import logging


setup_logger()
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
            if num_teams <= 5:
                self.solver = GurobiComplexSolver(input_data)
            else:
                self.solver = GurobiSimpleSolver(input_data)
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
        self.schedule_type = input_data.get('schedule_type')
        self.objective_choice = input_data.get('objective_choice')
        self.teams_original = list(input_data.get('teams', []))
        self.distance_matrix = input_data.get('distance_matrix', [])
        self.max_consecutive = input_data.get('max_consecutive', 3)
        self.num_teams_original = len(self.teams_original)
        self.teams = list(self.teams_original)
        self.city_list = input_data.get('city_list', [])
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
        self.break_penalty_weight = 1000
        self.original_team_to_idx = {name: idx for idx, name in enumerate(self.teams_original)}
        self.home_city_of_team = {idx: self.original_team_to_idx.get(self.teams[idx], -1) for idx in range(self.num_teams)}

class BaseOrtoolsSportsSolver(BaseSportsSchedulingSolver, BaseOrtoolsCpSolver):
    """OR-Tools CP-SAT 솔버를 위한 공통 로직을 포함하는 기본 클래스."""

    def __init__(self, input_data):
        # 부모 클래스들의 생성자를 모두 호출
        super().__init__(input_data)
        self.plays = {}
        self.team_travel_vars = []
        self.break_vars = []
        self.max_dist = sum(self.distance_matrix[0]) * self.num_slots  # 최대 이동 거리 계산

    @log_solver_make
    def _create_plays_variables(self):
        """plays[s, h, a]: 시간 s에 홈팀 h가 원정팀 a와 경기하면 1 """
        for s in range(self.num_slots):
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a:
                        var = self.model.NewBoolVar(f'plays_{s + 1}_{self.teams[h]}_{self.teams[a]}')
                        self.plays[(s, h, a)] = var
                        logger.solve(f"Var: {var.Name()}")

    def _add_one_game_per_slot_constraint(self):
        """제약: 각 팀은 각 슬롯에서 정확히 한 경기만 수행합니다."""
        for s in range(self.num_slots):
            for t in range(self.num_teams):
                home_games = [self.plays.get((s, t, a), 0) for a in range(self.num_teams) if t != a]
                away_games = [self.plays.get((s, h, t), 0) for h in range(self.num_teams) if t != h]
                constraint = self.model.AddExactlyOne(home_games + away_games)

                constraint_name = f"OnceSlot_{s + 1}_{self.teams[t]}"
                self.model.named_constraints[constraint_name] = constraint
                constraint_expr = " + ".join(var.Name() for var in home_games + away_games) + " == 1"
                logger.solve(f"Eq: {constraint_name}: {constraint_expr}")

    def _add_league_format_constraint(self):
        """제약: 리그 방식(single/double)에 따른 총 경기 수를 설정합니다."""
        if self.schedule_type == 'single':
            for h in range(self.num_teams):
                for a in range(h + 1, self.num_teams):
                    matchups = [self.plays.get((s, h, a), 0) for s in range(self.num_slots)] + \
                               [self.plays.get((s, a, h), 0) for s in range(self.num_slots)]
                    constraint = self.model.Add(sum(matchups) == 1)
                    constraint_name = f"SinglePair_{self.teams[h]}_{self.teams[a]}"
                    self.model.named_constraints[constraint_name] = constraint
                    logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in matchups)} == 1")
        else:  # double
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a:
                        home_matchups = [self.plays.get((s, h, a), 0) for s in range(self.num_slots)]
                        constraint = self.model.Add(sum(home_matchups) == 1)
                        constraint_name = f"DoublePair_{self.teams[h]}_{self.teams[a]}"
                        self.model.named_constraints[constraint_name] = constraint
                        logger.solve(f"{constraint_name}: {' + '.join(v.Name() for v in home_matchups)} == 1")

    def _add_max_consecutive_games_constraint(self):
        """제약: 최대 연속 홈/원정 경기 수를 제한합니다."""
        for t in range(self.num_teams_original):
            for s in range(self.num_slots - self.max_consecutive):
                away_window = [self.plays.get((idx, h, t),0)
                               for idx in range(s, s + self.max_consecutive + 1)
                               for h in range(self.num_teams) if t != h]
                constraint = self.model.Add(sum(away_window) <= self.max_consecutive)
                constraint_name = f"MaxAway_{self.teams[t]}_s{s + 1}"
                self.model.named_constraints[constraint_name] = constraint
                logger.solve(
                    f"{constraint_name}: {' + '.join(v.Name() for v in away_window)} <= {self.max_consecutive}")

                home_window = [self.plays.get((idx, t, a),0)
                               for idx in range(s, s + self.max_consecutive + 1)
                               for a in range(self.num_teams) if t != a]
                constraint = self.model.Add(sum(home_window) <= self.max_consecutive)
                constraint_name = f"MaxHome_{self.teams[t]}_s{s + 1}"
                self.model.named_constraints[constraint_name] = constraint
                logger.solve(
                    f"{constraint_name}: {' + '.join(v.Name() for v in home_window)} <= {self.max_consecutive}")

    def _add_common_constraints(self):
        """ 모든 공통 제약 함수를 호출하는 메인 함수입니다."""
        self._add_one_game_per_slot_constraint()
        self._add_league_format_constraint()
        self._add_max_consecutive_games_constraint()

    def _define_simple_total_distance(self):
        # 총 이동 거리 (Total Distance)
        self.team_travel_vars = [self.model.NewIntVar(0, self.max_dist, f'travel_{t}') for t in
                                 range(self.num_teams_original)]
        for t in range(self.num_teams_original):
            terms = [self.distance_matrix[t][opp] * self.plays.get((s, opp, t), 0)
                     for s in range(self.num_slots) for opp in range(self.num_teams_original) if t != opp]
            self.model.Add(self.team_travel_vars[t] == sum(terms))
        return sum(self.team_travel_vars)

    def _define_simple_total_breaks(self):
        # 총 연속 경기 수 (Total Breaks)
        for t in range(self.num_teams_original):
            for s in range(self.num_slots - 1):
                is_home_s = sum(self.plays.get((s, t, a), 0) for a in range(self.num_teams) if t != a)
                is_home_s_plus_1 = sum(self.plays.get((s + 1, t, a), 0) for a in range(self.num_teams) if t != a)

                # is_home_s == is_home_s_plus_1 이면 break_var = 1
                break_var = self.model.NewBoolVar(f'break_{t}_{s}')
                self.model.Add(is_home_s == is_home_s_plus_1).OnlyEnforceIf(break_var)
                self.model.Add(is_home_s != is_home_s_plus_1).OnlyEnforceIf(break_var.Not())
                self.break_vars.append(break_var)
        return sum(self.break_vars)

    def _define_simple_distance_gap(self):
        # 이동 거리 격차 (Distance Gap)
        min_travel = self.model.NewIntVar(0, self.max_dist, 'min_travel')
        max_travel = self.model.NewIntVar(0, self.max_dist, 'max_travel')
        self.model.AddMinEquality(min_travel, self.team_travel_vars)
        self.model.AddMaxEquality(max_travel, self.team_travel_vars)
        return max_travel - min_travel

class OrtoolsSimpleSolver(BaseOrtoolsSportsSolver):
    """6개 팀 이상을 위한 간단한 OR-Tools 모델."""

    def _create_variables(self):
        super()._create_plays_variables()

    def _add_constraints(self):
        super()._add_common_constraints()

    def _set_objective_function(self):
        # --- 3. 목표 함수 설정 ---
        # --- 가중치를 이용한 다중 목표 함수 설정 ---
        # 가중치: 하위 목표의 최대값이 상위 목표의 1단위를 넘지 않도록 설정
        total_travel = self._define_simple_total_distance()
        total_breaks = self._define_simple_total_breaks()
        distance_gap = self._define_simple_distance_gap()
        PRIMARY_WEIGHT = 10000

        if self.objective_choice == 'minimize_travel':
            self.model.Minimize(total_travel + total_breaks)
        elif self.objective_choice == 'fairness':
            self.model.Minimize(total_breaks * PRIMARY_WEIGHT + total_travel)
        elif self.objective_choice == 'distance_gap':
            self.model.Minimize(distance_gap * PRIMARY_WEIGHT + total_travel)

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
        results['objective_choice'] = self.objective_choice

        return results


class OrtoolsComplexSolver(BaseOrtoolsSportsSolver):
    """BaseOrtoolsSportsSolver를 상속받아, 복잡한 모델에만 필요한 변수와 제약 조건을 추가"""
    def __init__(self, input_data):
        super().__init__(input_data)
        self.is_at_loc = {}

    def _create_location_variables(self):
        for t in range(self.num_teams_original):
            for s in range(self.num_slots):
                for l in range(self.num_cities):
                    var = self.model.NewBoolVar(f"is_at_loc_{self.teams[t]}_{s + 1}_{l}")
                    self.is_at_loc[t, s, l] = var
                    logger.solve(f"Var: {var.Name()}")

    def _create_variables(self):
        super()._create_plays_variables()
        self._create_location_variables()

    def _add_location_constraints(self):
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


    def _add_constraints(self):
        super()._add_common_constraints()
        self._add_location_constraints()


    def _set_objective_function(self):
        """목표 함수 (총비용 최소화)를 설정합니다."""
        max_dist = sum(max(row) for row in self.distance_matrix) * self.num_slots
        self.team_travel_vars = [self.model.NewIntVar(0, max_dist, f'team_travel_{t}') for t in range(self.num_teams_original)]

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

            self.model.Add(self.team_travel_vars[t] == sum(slot_travel_dist))

        if self.objective_choice == 'minimize_travel':
            # Gurobi: model.setObjective(quicksum(team_travel_vars), GRB.MINIMIZE)
            total_travel = self.model.NewIntVar(0, max_dist * self.num_teams_original, 'total_travel')
            self.model.Add(total_travel == sum(self.team_travel_vars))
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
            total_travel = sum(self.team_travel_vars)
            self.model.Minimize(total_breaks * self.break_penalty_weight + total_travel)
            logger.solve("Objective set to: Minimize total number of breaks.")

        elif self.objective_choice == 'distance_gap':
            min_travel = self.model.NewIntVar(0, 1000000, "min_travel")
            max_travel = self.model.NewIntVar(0, 1000000, "max_travel")
            # Gurobi: model.addGenConstrMin(min_travel, team_travel_vars)
            self.model.AddMinEquality(min_travel, self.team_travel_vars)
            # Gurobi: model.addGenConstrMax(max_travel, team_travel_vars)
            self.model.AddMaxEquality(max_travel, self.team_travel_vars)

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
                    if h != a and (s, h, a) in self.plays and solver.Value(self.plays.get((s, h, a), 0)) == 1:
                        matchups.append({'home': self.teams[h], 'away': self.teams[a]})
            schedule.append({'week': s + 1, 'matchups': matchups})
        results['schedule'] = schedule

        for key, var in self.plays.items():
            if solver.Value(var) == 1:
                logger.solve(var.Name())
        if self.objective_choice == 'fairness':
            for var in self.break_vars:
                if solver.Value(var) == 1:
                    logger.solve(var.Name())

        # 결과 지표 계산
        total_dist_calc = 0
        team_distances_calc = []
        for i in range(self.num_teams_original):
            dist_val = solver.Value(self.team_travel_vars[i])
            team_distances_calc.append({'name': self.teams[i], 'distance': round(dist_val)})
            total_dist_calc += dist_val

        results['total_distance'] = round(total_dist_calc)
        results['team_distances'] = sorted(team_distances_calc, key=lambda x: x['name'])
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
        results['objective_choice'] = self.objective_choice

        return results


class BaseGurobiSportsSolver(BaseSportsSchedulingSolver, BaseGurobiSolver):
    """Gurobi 솔버를 위한 공통 로직을 포함하는 기본 클래스."""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.model = Model("SportsScheduling")
        self.plays = {}
        self.breaks = {}
        self.team_travel_vars = []

    def _create_plays_variables(self):
        try:
            for s in range(self.num_slots):
                for h in range(self.num_teams):
                    for a in range(self.num_teams):
                        if h == a:
                            continue  # 같은 팀 간의 경기는 변수를 생성하지 않음
                        var_name = f"Plays_{s + 1}_{self.teams[h]}_{self.teams[a]}"
                        self.plays[s, h, a] = self.model.addVar(vtype=GRB.BINARY, name=var_name)

            if self.analysis_mode:
                self.model.update() # Gurobi 모델의 내부 상태를 동기화하여 'VarName' 속성 접근 오류를 방지
                for (s, h, a), var in self.plays.items():
                    if h == a: continue
                    related_info = {
                        'slot': s + 1,
                        'home_team': self.teams[h],
                        'away_team': self.teams[a]
                    }
                    self.analyzer.add_variable(var, 'Plays', **related_info)
        except Exception as e:
            logger.error(f"Error creating plays variables: {e}")
            raise

    def _add_one_game_per_slot_constraint(self):
        """제약: 각 팀은 각 슬롯에서 정확히 한 경기만 수행합니다."""
        try:
            for s in range(self.num_slots):
                for t in range(self.num_teams):
                    eq_name = f"PlayOnce_{s + 1}_{self.teams[t]}"
                    eq = self.model.addConstr(
                        quicksum(self.plays[s, t, a] for a in range(self.num_teams) if t != a) +
                        quicksum(self.plays[s, h, t] for h in range(self.num_teams) if t != h) == 1,
                        name=eq_name
                    )

                    if self.analysis_mode:
                        self.analyzer.add_constraint(self.model, eq, 'PlayOnce', slot=s + 1, team=self.teams[t])
        except Exception as e:
            logger.error(f"Error adding one game per slot constraint: {e}")
            raise

    def _add_league_format_constraint(self):
        """제약: 리그 방식(single/double)에 따른 총 경기 수를 설정합니다."""
        try:
            if self.schedule_type == 'single':
                for h in range(self.num_teams):
                    for a in range(h + 1, self.num_teams):
                        eq_name = f"Single_{self.teams[h]}_{self.teams[a]}"
                        eq = self.model.addConstr(
                            quicksum(self.plays[s, h, a] + self.plays[s, a, h] for s in range(self.num_slots)) == 1,
                            name=eq_name)
                        if self.analysis_mode:
                            self.analyzer.add_constraint(self.model, eq, 'Single', home_team=self.teams[h],
                                                         away_team=self.teams[a])
            else:  # double
                for h in range(self.num_teams):
                    for a in range(self.num_teams):
                        if h != a:
                            eq_name = f"Double_{self.teams[h]}_{self.teams[a]}"
                            eq = self.model.addConstr(quicksum(self.plays[s, h, a] for s in range(self.num_slots)) == 1,
                                                      name=eq_name)
                            if self.analysis_mode:
                                self.analyzer.add_constraint(self.model, eq, 'Double', home_team=self.teams[h],
                                                             away_team=self.teams[a])
        except Exception as e:
            logger.error(f"Error adding league format constraint: {e}")
            raise

    def _add_max_consecutive_games_constraint(self):
        """제약: 최대 연속 홈/원정 경기 수를 제한합니다."""
        try:
            for t in range(self.num_teams_original):
                for s in range(self.num_slots - self.max_consecutive):
                    eq_name = f"Consecutive_home_{self.teams[t]}_{s + 1}"
                    eq = self.model.addConstr(quicksum(
                        self.plays[i, t, a] for i in range(s, s + self.max_consecutive + 1) for a in range(self.num_teams)
                        if t != a) <= self.max_consecutive, name =eq_name)
                    if self.analysis_mode:
                        self.analyzer.add_constraint(self.model, eq, 'Consecutive_home', team=self.teams_original[t],start_slot=s + 1)

                    eq_name = f"Consecutive_away_{self.teams[t]}_{s + 1}"
                    eq = self.model.addConstr(quicksum(
                        self.plays[i, h, t] for i in range(s, s + self.max_consecutive + 1) for h in range(self.num_teams)
                        if t != h) <= self.max_consecutive, name = eq_name)
                    if self.analysis_mode:
                        self.analyzer.add_constraint(self.model, eq, 'Consecutive_away', team=self.teams_original[t], start_slot=s + 1)
        except Exception as e:
            logger.error(f"Error adding max consecutive games constraint: {e}")
            raise

    def _add_common_constraints(self):
        self._add_one_game_per_slot_constraint()
        self._add_league_format_constraint()
        self._add_max_consecutive_games_constraint()


class GurobiSimpleSolver(BaseGurobiSportsSolver):
    """6개 팀 이상을 위한 간단한 Gurobi 모델."""

    def _create_variables(self):
        super()._create_plays_variables()

    def _add_constraints(self):
        super()._add_common_constraints()

    def _set_objective_function(self):
        logger.solve("--- 3. Setting Objective for Gurobi Simple Solver ---")
        self.team_travel_vars = self.model.addVars(self.num_teams_original, vtype=GRB.INTEGER, name="team_travel")
        for t_idx in range(self.num_teams_original):
            self.model.addConstr(
                self.team_travel_vars[t_idx] == quicksum(
                    self.distance_matrix[t_idx][opp_idx] * self.plays.get((s, opp_idx, t_idx), 0)
                    for s in range(self.num_slots) for opp_idx in range(self.num_teams_original) if t_idx != opp_idx
                )
            )

        if self.objective_choice == 'minimize_travel':
            self.model.setObjective(quicksum(self.team_travel_vars), GRB.MINIMIZE)

        elif self.objective_choice == 'fairness':
            self.breaks = self.model.addVars(self.num_teams_original, self.num_slots - 1, vtype=GRB.BINARY, name="breaks")
            for t in range(self.num_teams_original):
                for s in range(self.num_slots - 1):
                    is_home_s = quicksum(self.plays[s, t, a] for a in range(self.num_teams) if t != a)
                    is_home_s_plus_1 = quicksum(self.plays[s + 1, t, a] for a in range(self.num_teams) if t != a)
                    self.model.addGenConstrIndicator(self.breaks[t, s], True, is_home_s - is_home_s_plus_1, GRB.EQUAL, 0.0)

            total_breaks = quicksum(self.breaks.values())
            total_travel = quicksum(self.team_travel_vars)
            self.model.setObjective(total_breaks * self.break_penalty_weight + total_travel, GRB.MINIMIZE)

        elif self.objective_choice == 'distance_gap':
            min_travel = self.model.addVar(vtype=GRB.INTEGER, name="min_travel")
            max_travel = self.model.addVar(vtype=GRB.INTEGER, name="max_travel")
            self.model.addGenConstrMin(min_travel, self.team_travel_vars)
            self.model.addGenConstrMax(max_travel, self.team_travel_vars)
            self.model.setObjective(max_travel - min_travel, GRB.MINIMIZE)

    def _extract_results(self):
        logger.info("Extracting results for Gurobi Simple Solver...")
        results = {'schedule': [], 'has_bye': self.has_bye, 'total_distance': 0, 'team_distances': [],
                   'total_breaks': 0, 'distance_gap': 0}

        # 대진표 파싱
        schedule = []
        for s in range(self.num_slots):
            matchups = []
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a and self.plays[s, h, a].X > 0.5:
                        matchups.append({'home': self.teams[h], 'away': self.teams[a]})
            schedule.append({'week': s + 1, 'matchups': matchups})
        results['schedule'] = schedule

        # 결과 지표 계산
        team_distances = []
        for idx in range(self.num_teams_original):
            dist_val = self.team_travel_vars[idx].X
            team_distances.append({'name': self.teams_original[idx], 'distance': round(dist_val)})

        results['team_distances'] = sorted(team_distances, key=lambda x: x['name'])
        results['total_distance'] = round(sum(item['distance'] for item in team_distances))
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist) if dist else 0

        total_breaks = 0
        for t_idx in range(self.num_teams_original):
            for s in range(self.num_slots - 1):
                is_home_s = sum(self.plays[s, t_idx, a].X for a in range(self.num_teams) if t_idx != a)
                is_home_s_plus_1 = sum(self.plays[s + 1, t_idx, a].X for a in range(self.num_teams) if t_idx != a)
                if abs(is_home_s - is_home_s_plus_1) < 0.5:  # 연속
                    total_breaks += 1
        results['total_breaks'] = total_breaks
        results['objective_choice'] = self.objective_choice
        return results

class GurobiComplexSolver(BaseGurobiSportsSolver):
    """5개 팀 이하를 위한 복잡한 Gurobi 모델."""
    def __init__(self, input_data):
        super().__init__(input_data)
        self.is_at_loc = {}

    def _create_location_variables(self):
        for t in range(self.num_teams_original):
            for s in range(self.num_slots):
                for c in range(self.num_teams_original):
                    var_name = f"Atloc_{s + 1}_{self.teams[t]}_{self.city_list[c]}"
                    self.is_at_loc[t, s, c] = self.model.addVar(vtype=GRB.BINARY, name=var_name)

        if self.analysis_mode:
            self.model.update()
            for (t, s, c), var in self.is_at_loc.items():
                self.analyzer.add_variable(
                    var, 'Atloc',
                    team=self.teams_original[t],
                    slot=s + 1,
                    city=self.city_list[c]  # 도시 이름으로 기록
                )

    def _create_variables(self):
        super()._create_plays_variables()
        self._create_location_variables()

    def _add_location_constraints(self):
        # 위치 제약 추가
        for s in range(self.num_slots):
            for t in range(self.num_teams_original):
                # 팀 t가 홈 경기 시, 위치는 자신의 홈
                is_home = quicksum(self.plays[s, t, a] for a in range(self.num_teams) if t != a)
                self.model.addConstr(self.is_at_loc[t, s, self.home_city_of_team[t]] >= is_home)
                # 팀 t가 원정 경기 시, 위치는 상대팀 h의 홈
                for h in range(self.num_teams_original):
                    if t != h:
                        self.model.addConstr(self.is_at_loc[t, s, self.home_city_of_team[h]] >= self.plays[s, h, t])
            for t in range(self.num_teams_original):
                self.model.addConstr(quicksum(self.is_at_loc[t, s, l] for l in range(self.num_cities)) == 1)

    def _add_constraints(self):
        super()._add_common_constraints()
        self._add_location_constraints()

    def _set_objective_function(self):
        logger.solve("--- 3. Setting Objective for Gurobi Complex Solver ---")
        self.team_travel_vars = self.model.addVars(self.num_teams_original, vtype=GRB.INTEGER, name="team_travel")

        # 동적 이동 거리 계산
        for t in range(self.num_teams_original):
            slot_travel_dist = []
            # [NEW] s=0: introduce binary variables for initial location
            initial_loc_vars = self.model.addVars(self.num_cities, vtype=GRB.BINARY, name=f"init_loc_{t}")
            for l in range(self.num_cities):
                if l == self.home_city_of_team[t]:
                    self.model.addConstr(initial_loc_vars[l] == 1)
                else:
                    self.model.addConstr(initial_loc_vars[l] == 0)
            for s in range(self.num_slots):
                if s == 0:
                    prev_loc_vars = [initial_loc_vars[l] for l in range(self.num_cities)]
                else:
                    prev_loc_vars = [self.is_at_loc[t, s - 1, l] for l in range(self.num_cities)]
                curr_loc_vars = [self.is_at_loc[t, s, l] for l in range(self.num_cities)]

                # dist_in_slot[s] = sum over l1,l2 ( prev_loc_vars[l1] * curr_loc_vars[l2] * distance[l1][l2] )
                # 위 식은 변수 간의 곱이므로 선형이 아님. 선형화 필요.
                # Gurobi에서는 and_ 제약으로 선형화 가능
                travel_arc_vars = self.model.addVars(self.num_cities, self.num_cities, vtype=GRB.BINARY,
                                                     name=f"travel_{t}_{s}")
                for l1 in range(self.num_cities):
                    for l2 in range(self.num_cities):
                        # Z = X AND Y
                        self.model.addGenConstrAnd(travel_arc_vars[l1, l2], [prev_loc_vars[l1], curr_loc_vars[l2]])

                dist_for_slot = quicksum(
                    travel_arc_vars[l1, l2] * self.distance_matrix[l1][l2] for l1 in range(self.num_cities) for l2 in
                    range(self.num_cities))
                slot_travel_dist.append(dist_for_slot)

            self.model.addConstr(self.team_travel_vars[t] == quicksum(slot_travel_dist))

        # 목표 함수 선택
        if self.objective_choice == 'minimize_travel':
            self.model.setObjective(quicksum(self.team_travel_vars), GRB.MINIMIZE)
        elif self.objective_choice == 'fairness':
            self.breaks = self.model.addVars(self.num_teams_original, self.num_slots - 1, vtype=GRB.BINARY, name="breaks")
            for t in range(self.num_teams_original):
                for s in range(self.num_slots - 1):
                    # is_home_s: 팀 t가 시간 s에 홈이면 1
                    is_home_s = quicksum(self.plays[s, t, a] for a in range(self.num_teams) if t != a)
                    # is_home_s_plus_1: 팀 t가 시간 s+1에 홈이면 1
                    is_home_s_plus_1 = quicksum(self.plays[s + 1, t, a] for a in range(self.num_teams) if t != a)

                    # diff = is_home_s - is_home_s_plus_1 (값은 -1, 0, 1)
                    diff = self.model.addVar(lb=-1, ub=1, vtype=GRB.INTEGER, name=f"diff_{t}_{s}")
                    self.model.addConstr(diff == is_home_s - is_home_s_plus_1)

                    # abs_diff = |diff|. abs_diff는 0 또는 1 (break가 없으면 1, 있으면 0)
                    abs_diff = self.model.addVar(vtype=GRB.BINARY)
                    self.model.addGenConstrAbs(abs_diff, diff)

                    # breaks[t,s]는 abs_diff의 반대. breaks = 1 - abs_diff
                    self.model.addConstr(self.breaks[t, s] == 1 - abs_diff)

            total_breaks = quicksum(self.breaks.values())
            total_travel = quicksum(self.team_travel_vars)
            self.model.setObjective(total_breaks * self.break_penalty_weight + total_travel, GRB.MINIMIZE)
        elif self.objective_choice == 'distance_gap':
            min_travel = self.model.addVar(vtype=GRB.INTEGER, name="min_travel")
            max_travel = self.model.addVar(vtype=GRB.INTEGER, name="max_travel")
            self.model.addGenConstrMin(min_travel, self.team_travel_vars)
            self.model.addGenConstrMax(max_travel, self.team_travel_vars)
            self.model.setObjective(max_travel - min_travel, GRB.MINIMIZE)

    def _extract_results(self):
        logger.info("Extracting results for Gurobi Complex Solver...")
        results = {'schedule': [], 'has_bye': self.has_bye, 'total_distance': 0, 'team_distances': [], 'total_breaks': 0,
                   'distance_gap': 0}

        # 대진표 파싱
        schedule = []
        for s in range(self.num_slots):
            matchups = []
            for h in range(self.num_teams):
                for a in range(self.num_teams):
                    if h != a and self.plays[s, h, a].X > 0.5:
                        matchups.append({'home': self.teams[h], 'away': self.teams[a]})
            schedule.append({'week': s + 1, 'matchups': matchups})
        results['schedule'] = schedule

        # 결과 지표 계산
        team_distances = []
        for idx in range(self.num_teams_original):
            dist_val = self.team_travel_vars[idx].X
            team_distances.append({'name': self.teams_original[idx], 'distance': round(dist_val)})

        results['team_distances'] = sorted(team_distances, key=lambda x: x['name'])
        results['total_distance'] = round(sum(item['distance'] for item in team_distances))
        dist = [item['distance'] for item in results['team_distances']]
        results['distance_gap'] = max(dist) - min(dist) if dist else 0

        total_breaks = 0
        if self.objective_choice == 'fairness':
            total_breaks = sum(var.X for var in self.breaks.values())
        else:  # 수동 계산
            for t_idx in range(self.num_teams_original):
                for s in range(self.num_slots - 1):
                    is_home_s = sum(self.plays[s, t_idx, a].X for a in range(self.num_teams) if t_idx != a)
                    is_home_s_plus_1 = sum(self.plays[s + 1, t_idx, a].X for a in range(self.num_teams) if t_idx != a)
                    if abs(is_home_s - is_home_s_plus_1) < 0.5:
                        total_breaks += 1
        results['total_breaks'] = round(total_breaks)
        results['objective_choice'] = self.objective_choice
        return results



with open('../test_data/puzzles_sports_scheduling_data/minimize_travel_single_team4.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

solver_instance = SportsSolverFactory(input_data)
results_data, error_msg_opt, processing_time = solver_instance.solve()
for key, value in results_data.items():
    logger.solve(f'{key}:{value}')