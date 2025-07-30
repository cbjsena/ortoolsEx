from ortools.sat.python import cp_model

from utils.data_utils_allocation import *
from utils.common_run_opt import *
import json
from logging_config import setup_logger
import logging


setup_logger()
logger = logging.getLogger(__name__)

SHIFT_NIGHT ='Ngt'
nurses_data=[]
num_nurses = 0
num_days = 0
shifts = []
min_shifts_per_nurse = 5
max_shifts_per_nurse = 8
weekend_days = [] # 주말에 해당하는 날짜 인덱스
all_nurses = range(num_nurses)
all_days = range(num_days)
all_shifts = range(len(shifts))
num_shifts_per_day = len(shifts)
nurse_ids=[]
nurses_by_skill={}
vacation_requests=[]
enabled_fairness=[]
skill_requirements={}
def parse_data(input_data):
    SHIFT_NIGHT = preset_nurse_rostering_shifts[2]
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
    logger.solve(
        f"Running Nurse Rostering Advanced Optimizer. - Num nurses: {num_nurses},  Num days: {num_days}, Shifts: {shifts}")

def run_nurse_roster_advanced_optimizer1(input_data):
    """
    숙련도, 휴가, 강화된 공정성 등 고급 제약이 포함된 스케줄링 문제를 해결합니다.
    """
    problem_type ='nurse rostering'

    try:
        model = cp_model.CpModel()

        # --- 1. 결정 변수 생성 ---
        assigns = setVarAssign(model, nurse_ids)
        # for n_id in nurse_ids:
        #     for d in range(num_days):
        #         for s in range(num_shifts_per_day):
        #             varName = f"assigns_{nurses_data[n_id].get('name')}_{d + 1}_{shifts[s]}"
        #             logger.solve(f'BoolVar: {varName}')
        #             assigns[(n_id, d, s)] = model.NewBoolVar(varName)

        # --- 2. 강성 제약 조건 (Hard Constraints) ---

        # 제약 1: 각 간호사는 하루 최대 1개 시프트 근무
        for n_id in nurse_ids:
            for d in range(num_days):
                model.AddAtMostOne(assigns[(n_id, d, s)] for s in range(num_shifts_per_day))

        # 제약 2: [수정] 숙련도별 필요 인원 충족
        for d in range(num_days):
            for s_idx, s_name in enumerate(shifts):
                for skill, required_count in skill_requirements[s_name].items():
                    nurses_with_that_skill = nurses_by_skill[skill]
                    model.Add(sum(assigns[(n_id, d, s_idx)] for n_id in nurses_with_that_skill) >= required_count)

        # for d in range(num_days):
        #     for s_idx, s_name in enumerate(shifts):
        #         for shift, requirements in skill_requirements[s_name].items():
        #             total_sum = sum(requirements.values())+3
        #             model.Add(sum(assigns[n_id, d, s_idx] for n_id in nurse_ids) <= total_sum)

        # 제약 3: [신규] 휴가 요청 반영
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

        # 목표 2: [신규] 공평한 휴무일 분배
        if 'fair_offs' in enabled_fairness:
            total_shifts_worked = [
                sum(assigns[(n_id, d, s)] for d in range(num_days) for s in range(num_shifts_per_day)) for n_id in
                nurse_ids]
            off_days_per_nurse = [num_days - s for s in total_shifts_worked]
            min_offs = model.NewIntVar(0, num_days, 'min_offs')
            max_offs = model.NewIntVar(0, num_days, 'max_offs')
            model.AddMinEquality(min_offs, off_days_per_nurse)
            model.AddMaxEquality(max_offs, off_days_per_nurse)
            off_gap = max_offs - min_offs
        else:
            off_gap = 0

        # 목표 3: [기존] 공평한 주말 근무 분배
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
        model.Minimize(night_gap * 2 + off_gap + weekend_gap * 3)  # 야간, 주말에 가중치 부여

        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status, processing_time = solving_log(solver, problem_type, model)

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
            # logger.info(f'results_data:{results_data}')
            return results_data, None, processing_time
        else:
            return None, "해를 찾을 수 없었습니다. 제약 조건이 너무 엄격하거나, 필요 인원이 간호사 수에 비해 너무 많을 수 있습니다.", round(processing_time, 4)

    except Exception as e:
        return None, f"오류 발생: {str(e)}", None


def run_nurse_roster_advanced_optimizer2(input_data):
    """
    [수정] 숙련도별 차등 임금, 주말/야간 추가 수당을 고려하여
    총 급여를 최소화하는 스케줄링 문제를 해결합니다.
    """
    # --- 입력 데이터 파싱 (기존과 동일) ---
    nurses_data = input_data['nurses_data']
    num_nurses = len(nurses_data)
    num_days = input_data['num_days']
    shifts = input_data['shifts']
    skill_requirements = input_data['skill_requirements']
    all_skills = list(skill_requirements[shifts[0]].keys())
    vacation_requests = input_data['vacation_requests']
    weekend_days = input_data['weekend_days']
    num_shifts_per_day = len(shifts)

    # 간호사 ID 및 스킬 매핑
    nurse_ids = [n['id'] for n in nurses_data]
    nurses_by_skill = {skill: [n['id'] for n in nurses_data if n['skill'] == skill] for skill in all_skills}
    nurse_skill_map = {n['id']: n['skill'] for n in nurses_data}

    # --- [신규] 급여 관련 파라미터 정의 ---
    # 숙련도별 일일 기본급 (예시)
    base_wage_by_skill = {'H': 150, 'M': 120, 'L': 100}

    # 주말 및 야간 근무 수당 (기본급의 2배이므로, 추가분은 1배)
    weekend_premium_multiplier = 1.0
    night_premium_multiplier = 1.0

    try:
        model = cp_model.CpModel()

        # --- 1. 결정 변수 생성 (기존과 동일) ---
        assigns = {}
        for n_id in nurse_ids:
            for d in range(num_days):
                for s in range(len(shifts)):
                    assigns[(n_id, d, s)] = model.NewBoolVar(f'assigns_n{n_id}_d{d}_s{s}')

        # --- 2. 강성 제약 조건 (기존과 동일) ---
        # 제약 1: 하루 최대 1개 시프트
        for n_id in nurse_ids:
            for d in range(num_days):
                model.AddAtMostOne(assigns[(n_id, d, s)] for s in range(len(shifts)))

        # 제약 2: 시프트별, 숙련도별 최소 필요 인원 충족
        for d in range(num_days):
            for s_idx, s_name in enumerate(shifts):
                for skill, required_count in skill_requirements[s_name].items():
                    model.Add(sum(assigns[(n_id, d, s_idx)] for n_id in nurses_by_skill[skill]) >= required_count)

        # [신규] 제약 3: 시프트별 총 인원 상한 제약 (과도한 배정 방지)
        # 최소 필요 인원에서 1명 이상 초과 배정하지 않도록 강제
        # for d in range(num_days):
        #     for s_idx, s_name in enumerate(shifts):
        #         total_required = sum(skill_requirements[s_name].values())
        #         model.Add(sum(assigns[(n_id, d, s_idx)] for n_id in nurse_ids) <= total_required + 3)

        # 제약 4: 휴가 요청 반영 (기존과 동일)
        for n_id, off_days in vacation_requests.items():
            for d in off_days:
                model.Add(sum(assigns[(n_id, d, s)] for s in range(len(shifts))) == 0)

        # --- 3. 총 급여 계산 및 목표 함수 설정 ---
        total_wage = model.NewIntVar(0, 1000000, 'total_wage')  # 충분히 큰 상한값
        wage_components = []

        night_shift_idx = shifts.index('Ngt') if 'Ngt' in shifts else -1

        # 평균 주말 근무일 및 야간 근무일 계산 (페널티 기준선)
        avg_weekend_shifts = (sum(sum(req.values()) for req in skill_requirements.values()) * len(
            weekend_days)) / num_nurses
        avg_night_shifts = (sum(
            skill_requirements['Ngt'].values()) * num_days) / num_nurses if night_shift_idx != -1 else 0

        for n_id in nurse_ids:
            nurse_skill = nurse_skill_map[n_id]
            base_wage = base_wage_by_skill[nurse_skill]

            # 기본급 계산
            shifts_worked = [assigns[(n_id, d, s)] for d in range(num_days) for s in range(len(shifts))]
            wage_components.extend([s * base_wage for s in shifts_worked])

            # 주말 근무 수당 계산
            weekend_shifts = [assigns[(n_id, d, s)] for d in weekend_days for s in range(len(shifts))]
            num_weekend_shifts = sum(weekend_shifts)

            # 주말 근무가 평균보다 많을 경우에 대한 페널티(추가 수당)
            is_over_weekend = model.NewBoolVar(f'is_over_weekend_n{n_id}')
            model.Add(num_weekend_shifts > int(avg_weekend_shifts)).OnlyEnforceIf(is_over_weekend)
            model.Add(num_weekend_shifts <= int(avg_weekend_shifts)).OnlyEnforceIf(is_over_weekend.Not())
            wage_components.extend(
                [s * base_wage * weekend_premium_multiplier * is_over_weekend for s in weekend_shifts])

            # 야간 근무 수당 계산
            if night_shift_idx != -1:
                night_shifts = [assigns[(n_id, d, night_shift_idx)] for d in range(num_days)]
                num_night_shifts = sum(night_shifts)
                is_over_night = model.NewBoolVar(f'is_over_night_n{n_id}')
                model.Add(num_night_shifts > int(avg_night_shifts)).OnlyEnforceIf(is_over_night)
                model.Add(num_night_shifts <= int(avg_night_shifts)).OnlyEnforceIf(is_over_night.Not())
                wage_components.extend([s * base_wage * night_premium_multiplier * is_over_night for s in night_shifts])

        model.Add(total_wage == sum(wage_components))

        # --- 4. 목표 함수 설정 ---
        # 총 임금 최소화
        model.Minimize(total_wage)

        # --- 5. 문제 해결 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        processing_time = solver.WallTime()
        logger.info(f"Solver status: {status}, Time: {processing_time} ms")

        # --- 6. 결과 추출 ---
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # 최종 스케줄 생성
            schedule = {}
            for d in range(num_days):
                schedule[d] = {}
                for s_idx, s_name in enumerate(shifts):
                    schedule[d][s_idx] = [nurses_data[n_id].get('name') for n_id in nurse_ids if
                                          solver.Value(assigns[(n_id, d, s_idx)]) == 1]

            # --- [수정] 비용 및 근무 통계 지표 계산 ---
            nurse_stats = {}
            night_shift_idx = shifts.index('Ngt') if 'Ngt' in shifts else -1

            # 평균 주말/야간 근무일수 (수당 계산 기준)
            avg_weekend_shifts = (sum(sum(req.values()) for req in skill_requirements.values()) * len(
                weekend_days)) / num_nurses
            avg_night_shifts = (sum(
                skill_requirements['Ngt'].values()) * num_days) / num_nurses if night_shift_idx != -1 else 0

            for n_id in nurse_ids:
                nurse_info = nurses_data[n_id]
                skill = nurse_info['skill']
                base_wage = base_wage_by_skill[skill]

                # 근무일수 계산
                total_shifts = sum(solver.Value(assigns[(n_id, d, s)]) for d in range(num_days) for s in range(num_shifts_per_day))
                num_weekends = sum(solver.Value(assigns[(n_id, d, s)]) for d in weekend_days for s in range(num_shifts_per_day))
                num_nights = sum(solver.Value(assigns[(n_id, d, night_shift_idx)]) for d in
                                 range(num_days)) if night_shift_idx != -1 else 0

                # 급여 계산
                base_wage_total = total_shifts * base_wage
                weekend_premium = num_weekends * base_wage * weekend_premium_multiplier if num_weekends > avg_weekend_shifts else 0
                night_premium = num_nights * base_wage * night_premium_multiplier if num_nights > avg_night_shifts else 0
                total_individual_wage = base_wage_total + weekend_premium + night_premium

                nurse_stats[n_id] = {
                    'name': nurse_info['name'],
                    'skill': skill,
                    'total_shifts': total_shifts,
                    'night_shifts': num_nights,
                    'weekend_shifts': num_weekends,
                    'off_days': num_days - total_shifts,
                    'base_wage': base_wage_total,
                    'premium_pay': weekend_premium + night_premium,
                    'total_wage': total_individual_wage
                }

            results_data = {
                'schedule': schedule,
                'nurse_stats': nurse_stats,
                'total_wage': solver.ObjectiveValue(),
                'status_name': solver.StatusName(status)
            }
            return results_data, None, processing_time
        else:
            error_message = f"해를 찾을 수 없었습니다. (상태: {solver.StatusName(status)})"
            return None, error_message, processing_time
    except Exception as e:
        return None, f"오류 발생: {str(e)}", None

def setVarAssign(model, nurse_ids):
    assigns={}
    for n_id in nurse_ids:
        for d in range(num_days):
            for s in range(num_shifts_per_day):
                varName = f"assigns_{nurses_data[n_id].get('name')}_{d + 1}_{shifts[s]}"
                logger.solve(f'BoolVar: {varName}')
                assigns[(n_id, d, s)] = model.NewBoolVar(varName)
    return assigns

with open('../test_data/allocation_nurse_data/test.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

results_data, error_msg_opt, processing_time_ms = run_nurse_roster_advanced_optimizer1(input_data)
logger.info(results_data)