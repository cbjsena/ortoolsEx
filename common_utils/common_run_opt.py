import datetime
import logging
import os
import traceback

from google.protobuf import text_format
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)

status_map = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}

def start_log(problem_type:str):
    logger.info(f"Running Optimizer {problem_type}")


def solving_log(solver, problem_type:str, model=None):
    logger.info(f"Solving the {problem_type} model")
    if solver.__class__.__name__ == "CpSolver":
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
    else:
        status = solver.Solve()
        status_name = status_map.get(status, "UNKNOWN")
    processing_time = get_solving_time_sec(solver)
    logger.info(f"Solver finished. Status: {status_name}, Time: {processing_time} sec")

    return status, processing_time


def gurobi_solving_log(model, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    model.optimize()
    status = model.status
    processing_time = get_time(model.Runtime)
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return status, processing_time


def ortools_routing_solving_log(routing, search_parameters, problem_type:str):
    logger.info(f"Solving the {problem_type} model")
    solve_start_time = datetime.datetime.now()
    solution = routing.SolveWithParameters(search_parameters)
    status = routing.status()
    solve_end_time = datetime.datetime.now()
    processing_time = get_time((solve_end_time - solve_start_time).total_seconds())
    logger.info(f"Solver finished. Status: {status}, Time: {processing_time} sec")

    return solution, status, processing_time


def get_solving_time_sec(solver):
    """
       OR-Tools solver의 WallTime을 초 단위 문자열로 반환합니다.
       - CP-SAT solver는 초(sec)
       - Linear solver는 밀리초(ms)
       """
    try:
        time_raw = solver.WallTime()
    except AttributeError:
        return "N/A"

    # CP-SAT solver는 일반적으로 `CpSolver` 클래스의 인스턴스
    is_cp_sat = solver.__class__.__name__ == "CpSolver"

    processing_time = time_raw if is_cp_sat else time_raw / 1000  # 초 단위로 통일
    return get_time(processing_time)


def get_time(processing_time):
    return f"{processing_time:.3f}" if processing_time is not None else "N/A"


def get_mps_dir(filename: str):
    # 현재 파일 기준 상위 폴더의 mps 디렉토리 경로 구하기
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    mps_dir = os.path.abspath(os.path.join(base_dir, "..", "mps"))

    # 디렉토리가 없다면 생성
    os.makedirs(mps_dir, exist_ok=True)

    # 전체 경로 설정
    file_path = os.path.join(mps_dir, filename)

    return file_path


def export_cp_model(model: cp_model.CpModel, filename: str):
    """
    CP-SAT 모델을 MPS 파일로 저장합니다.
    :param model:
    :param filename: 저장할 파일 이름 (확장자 포함)
    :return:
    """
    file_path = get_mps_dir(filename)

    # CP-SAT 모델을 MPS 파일로 저장
    proto = model.Proto()
    with open(file_path, "w") as f:
        f.write(text_format.MessageToString(proto))


def export_ortools_solver(solver: pywraplp.Solver, filename: str):
    """
    OR-Tools 솔버 모델을 MPS 파일로 저장합니다.
    :param solver:
    :param filename: 저장할 파일 이름 (확장자 포함)
    :return:
    """
    file_path = get_mps_dir(filename)


    solver.WriteModelToMpsFile(get_mps_dir(file_path), True, False)
    logger.info(f"OR-Tools model exported to {file_path}")


def export_gurobi_model(model, filename: str):
    """
    Gurobi 모델을 MPS 파일로 저장합니다.
    :param model: Gurobi 모델 객체
    :param filename: 저장할 파일 이름 (확장자 포함)
    """
    file_path = get_mps_dir(filename)

    # Gurobi 모델을 MPS 파일로 저장
    model.write(file_path)
    logger.info(f"Gurobi model exported to {file_path}")


def parse_pb_file(filename: str) -> tuple[list[str], dict[int, dict[str, any]]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    mps_dir = os.path.abspath(os.path.join(base_dir, "..", "mps"))
    file_path = os.path.join(mps_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    var_names: list[str] = []
    constraints:dict[int, dict[str, any]] = {}

    current_block = None
    in_linear = False
    in_lin_max = False
    in_exprs = False
    in_target = False
    in_bool_or = False
    in_bool_and = False
    in_at_most_one = False

    current_line_no = -1
    offset =0
    current_vars: list[int] = []
    current_coeffs: list[int] = []
    current_exprs: list[dict[str, list[int]]] = []
    current_target: list[int] = []
    current_literals: list[int] = []
    try:
        for i, line in enumerate(lines):
            line = line.strip()

            # 변수 이름 추출
            if line.startswith("variables {"):
                current_block = "variable"
            elif line.startswith("constraints {"):
                current_block = "constraint"
                # 제약 초기화
                current_vars = []
                current_coeffs = []
                current_exprs = []
                current_target = []
                current_literals = []
                current_line_no = i+1
            elif current_block == "variable" and line.startswith("name:"):
                name = line.split("name:")[1].strip().strip('"')
                var_names.append(name)

            # constraints: linear 안 파싱
            elif line == "linear {":
                in_linear = True
            elif in_linear and line.startswith("vars:"):
                current_vars.append(int(line.split("vars:")[1].strip()))
            elif in_linear and line.startswith("coeffs:"):
                current_coeffs.append(int(line.split("coeffs:")[1].strip()))
            elif in_linear and line.startswith("offset:"):
                offset = int(line.split("offset:")[1].strip())
            elif in_linear and line == "}":
                in_linear = False
                constraints[current_line_no] = {
                    "type": "linear",
                    "vars": current_vars,
                    "coeffs": current_coeffs,
                    "offset": offset
                }

                # --- lin_max 제약 ---
            elif line == "lin_max {":
                in_lin_max = True
            elif in_lin_max and line == "exprs {":
                in_exprs = True
                current_vars = []
                current_coeffs = []
            elif in_exprs and line.startswith("vars:"):
                current_vars.append(int(line.split("vars:")[1].strip()))
            elif in_exprs and line.startswith("coeffs:"):
                current_coeffs.append(int(line.split("coeffs:")[1].strip()))
            elif in_exprs and line.startswith("offset:"):
                offset = int(line.split("offset:")[1].strip())
            elif in_exprs and line == "}":
                current_exprs.append({
                    "vars": current_vars,
                    "coeffs": current_coeffs,
                    "offset": offset
                })
                in_exprs = False
            elif in_lin_max and line == "target {":
                in_target = True
                current_target = []
            elif in_target and line.startswith("vars:"):
                current_target.append(int(line.split("vars:")[1].strip()))
            elif in_target and line == "}":
                in_target = False
            elif in_lin_max and line == "}":
                in_lin_max = False
                constraints[current_line_no] = {
                    "type": "lin_max",
                    "exprs": current_exprs,
                    "target": current_target
                }

                # --- at_most_one ---
            elif line == "at_most_one {":
                in_at_most_one = True
                current_literals = []
            elif in_at_most_one and line.startswith("literals:"):
                current_literals.append(int(line.split("literals:")[1].strip()))
            elif in_at_most_one and line == "}":
                in_at_most_one = False
                constraints[current_line_no] = {
                    "type": "at_most_one",
                    "literals": current_literals
                }

                # --- bool_or ---
            elif line == "bool_or {":
                in_bool_or = True
                current_literals = []
            elif in_bool_or and line.startswith("literals:"):
                current_literals.append(int(line.split("literals:")[1].strip()))
            elif in_bool_or and line == "}":
                in_bool_or = False
                constraints[current_line_no] = {
                    "type": "bool_or",
                    "literals": current_literals
                }

                # --- bool_and ---
            elif line == "bool_and {":
                in_bool_and = True
                current_literals = []
            elif in_bool_and and line.startswith("literals:"):
                current_literals.append(int(line.split("literals:")[1].strip()))
            elif in_bool_and and line == "}":
                in_bool_and = False
                constraints[current_line_no] = {
                    "type": "bool_and",
                    "literals": current_literals
                }
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()

    return var_names, constraints


def desc_model_by_line(line_no:int, var_names: list[str], constraints: list[dict[str, any]]):
    try:
        if line_no not in constraints:
            print(f"❌ 제약 시작 라인 {line_no}에는 제약이 없습니다.")
            return

        c = constraints[line_no]
        ctype = c.get("type", "unknown")

        print(f"\n🧩 Constraint at line {line_no} (type: {ctype}):")

        if ctype == "linear":
            expr = ""
            for i, var_idx in enumerate(c["vars"]):
                name = var_names[var_idx] if var_idx < len(var_names) else f"var#{var_idx}"
                coeff = c["coeffs"][i] if i < len(c["coeffs"]) else "?"
                expr += f"{coeff}*{name} + "
            expr = expr.rstrip(" + ")

            offset = c.get("offset", 0)
            if offset != 0:
                expr += f" + ({offset})"
            print(f"  ▶️ {expr} ≥ 0")

        elif ctype == "lin_max":
            print("  exprs:")
            for expr in c["exprs"]:
                line = "    "
                for i, var_idx in enumerate(expr["vars"]):
                    name = var_names[var_idx] if var_idx < len(var_names) else f"var#{var_idx}"
                    coeff = expr["coeffs"][i] if i < len(expr["coeffs"]) else "?"
                    line += f"{coeff}*{name} + "
                if "offset" in expr:
                    print(f"{line.rstrip(' +')} >= {expr['offset']}")
                else:
                    print(line.rstrip(" +"))

            print("  target:")
            for var_idx in c["target"]:
                name = var_names[var_idx] if var_idx < len(var_names) else f"var#{var_idx}"
                print(f"    {name}")

        elif ctype in ["at_most_one", "bool_or", "bool_and"]:
            print(f"  literals:")
            for var_idx in c["literals"]:
                name = var_names[var_idx] if var_idx < len(var_names) else f"var#{var_idx}"
                print(f"    {name}")

        else:
            print("  (지원하지 않는 제약 형식입니다. 원시 내용):")
            print(c)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()