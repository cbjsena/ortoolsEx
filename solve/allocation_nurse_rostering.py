from gurobipy import Model, GRB, quicksum
from ortools.sat.python import cp_model
import datetime
import json
from logging_config import setup_logger
import logging
import settings

setup_logger()
logger = logging.getLogger(__name__)

SHIFT_NIGHT ='Ngt'
def run_nurse_roster_advanced_optimizer(input_data):
    """
    숙련도, 휴가, 강화된 공정성 등 고급 제약이 포함된 스케줄링 문제를 해결합니다.
    """

    # --- 입력 데이터 파싱 ---
    nurses_data = input_data['nurses_data']
    num_nurses = len(nurses_data)
    num_days = input_data['num_days']
    shifts = input_data['shifts']

    # 숙련도별 필요 인원
    skill_requirements = input_data['skill_requirements']
    all_skills = list(skill_requirements[shifts[0]].keys())  # ['상', '중', '하']

    # 간호사별 휴가 요청
    vacation_requests = input_data['vacation_requests']  # {nurse_id: [day1, day2], ...}

    # 선택적으로 적용할 공정성 제약
    enabled_fairness = input_data.get('enabled_fairness', [])

    weekend_days = input_data['weekend_days']
    num_shifts_per_day = len(shifts)

    # 간호사 ID와 인덱스, 스킬 매핑
    nurse_ids = [n['id'] for n in nurses_data]
    nurses_by_skill = {skill: [n['id'] for n in nurses_data if n['skill'] == skill] for skill in all_skills}
    logger.info(
        f"Running Nurse Rostering Advanced Optimizer. - Num nurses: {num_nurses},  Num days: {num_days}, Shifts: {shifts}")

    try:
        model = cp_model.CpModel()
        varList = {}
        eqList= {}
        # --- 1. 결정 변수 생성 ---
        assigns = {}
        for n_id in nurse_ids:
            for d in range(num_days):
                for s in range(num_shifts_per_day):
                    varName = f"assigns_{nurses_data[n_id].get('name')}_{d+1}_{shifts[s]}"
                    logger.solve(f'BoolVar: {varName}')
                    assigns[(n_id, d, s)] = model.NewBoolVar(varName)

        # --- 2. 강성 제약 조건 (Hard Constraints) ---

        # 제약 1: 각 간호사는 하루 최대 1개 시프트 근무
        for n_id in nurse_ids:
            for d in range(num_days):
                model.AddAtMostOne(assigns[(n_id, d, s)] for s in range(num_shifts_per_day))

        # 제약 2: 숙련도별 필요 인원 충족
        for d in range(num_days):
            for s_idx, s_name in enumerate(shifts):
                for skill, required_count in skill_requirements[s_name].items():
                    nurses_with_that_skill = nurses_by_skill[skill]
                    model.Add(sum(assigns[(n_id, d, s_idx)] for n_id in nurses_with_that_skill) >= required_count)

        # 제약 3: 휴가 요청 반영
        for n_id, off_days in vacation_requests.items():
            for d in off_days:
                model.Add(sum(assigns[(n_id, d, s)] for s in range(num_shifts_per_day)) == 0)

        # --- 3. 연성 제약 조건 (Soft Constraints) 및 목표 함수 ---

        # 목표 1: [신규] 공평한 야간 근무 분배
        if 'fair_nights' in enabled_fairness:
            night_shift_idx = shifts.index(SHIFT_NIGHT)
            night_shifts_worked = [sum(assigns[(n_id, d, night_shift_idx)] for d in range(num_days)) for n_id in
                                   nurse_ids]
            min_nights = model.NewIntVar(0, num_days, 'min_nights')
            max_nights = model.NewIntVar(0, num_days, 'max_nights')
            model.AddMinEquality(min_nights, night_shifts_worked)
            model.AddMaxEquality(max_nights, night_shifts_worked)
            night_gap = max_nights - min_nights
        else:
            night_gap = 0
        #
        # # 목표 2: [신규] 공평한 휴무일 분배
        # if 'fair_offs' in enabled_fairness:
        #     total_shifts_worked = [
        #         sum(assigns[(n_id, d, s)] for d in range(num_days) for s in range(num_shifts_per_day)) for n_id in
        #         nurse_ids]
        #     off_days_per_nurse = [num_days - s for s in total_shifts_worked]
        #     min_offs = model.NewIntVar(0, num_days, 'min_offs')
        #     max_offs = model.NewIntVar(0, num_days, 'max_offs')
        #     model.AddMinEquality(min_offs, off_days_per_nurse)
        #     model.AddMaxEquality(max_offs, off_days_per_nurse)
        #     off_gap = max_offs - min_offs
        # else:
        #     off_gap = 0
        #
        # # 목표 3: [기존] 공평한 주말 근무 분배
        if 'fair_weekends' in enabled_fairness:
            weekend_shifts_worked = [sum(assigns[(n_id, d, s)] for d in weekend_days for s in range(num_shifts_per_day))
                                     for n_id in nurse_ids]
            min_weekend_shifts = model.NewIntVar(0, len(weekend_days), 'min_weekend')
            max_weekend_shifts = model.NewIntVar(0, len(weekend_days), 'max_weekend')
            model.AddMinEquality(min_weekend_shifts, weekend_shifts_worked)
            model.AddMaxEquality(max_weekend_shifts, weekend_shifts_worked)
            weekend_gap = max_weekend_shifts - min_weekend_shifts
        else:
            weekend_gap = 0
            weekend_shifts_worked = [0] * num_nurses  # 결과 표시를 위한 기본값

        # --- 4. 목표 함수 설정 ---
        # 각 공정성 목표의 격차(gap) 합을 최소화
        model.Minimize(night_gap * 2 + weekend_gap)  # + off_gap  * 3야간, 주말에 가중치 부여

        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        processing_time = solver.WallTime()
        logger.info(f"Solver status: {status_name}, Time: {processing_time} ms")

        # --- 6. 결과 추출 ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = {}
            for d in range(num_days):
                schedule[d] = {}
                for s_idx, s_name in enumerate(shifts):
                    schedule[d][s_idx] = [nurses_data[n_id].get('name') for n_id in nurse_ids if solver.Value(assigns[(n_id, d, s_idx)]) == 1]

            # 각 간호사별 통계 계산
            total_shifts = [
                sum(solver.Value(assigns[(n_id, d, s)]) for d in range(num_days) for s in range(num_shifts_per_day)) for
                n_id in nurse_ids]
            if 'fair_nights' in enabled_fairness and SHIFT_NIGHT in shifts:
                night_shift_idx = shifts.index(SHIFT_NIGHT)
                total_nights = [sum(solver.Value(assigns[(n_id, d, night_shift_idx)]) for d in range(num_days)) for n_id
                                in nurse_ids]
            else:
                total_nights = [0] * num_nurses
            total_weekends = [solver.Value(w) for w in
                              weekend_shifts_worked] if 'fair_weekends' in enabled_fairness else [0] * num_nurses
            total_offs = [num_days - ts for ts in total_shifts]

            results_data = {
                'schedule': schedule,
                'nurse_stats': {
                    n_id: {
                        'total': total_shifts[i], 'nights': total_nights[i],
                        'weekends': total_weekends[i], 'offs': total_offs[i]
                    } for i, n_id in enumerate(nurse_ids)
                },
                'total_penalty': solver.ObjectiveValue()
            }
            logger.info(f'results_data:{results_data}')
            return results_data, None, processing_time
        else:
            return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", round(processing_time, 4)

    except Exception as e:
        return None, f"오류 발생: {str(e)}", None

with open('../test_data/allocation_nurse_data/test1.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_nurse_roster_advanced_optimizer(input_data)
logger.info(results_data)