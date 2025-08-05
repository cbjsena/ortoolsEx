import logging
import gurobipy as gp
from psycopg2.extras import execute_values

from common_utils.base_analyzer import BaseGurobiAnalyzer, BaseOrtoolsCpModelAnalyzer

logger = logging.getLogger(__name__)


class GurobiModelAnalyzer(BaseGurobiAnalyzer):
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        """
        db_config: dict with keys 'host', 'port', 'user', 'password', 'dbname'
        """
        super().__init__(run_id)

    def add_variable(self, var_obj, var_group, **kwargs):
        """
        Gurobi 변수 객체(gp.Var)를 받아 DB에 저장합니다.
        kwargs를 통해 slot, home_team 등 추가 정보를 받습니다.
        """
        try:
            var_type_map = {
                gp.GRB.CONTINUOUS: 'CONTINUOUS',
                gp.GRB.BINARY: 'BINARY',
                gp.GRB.INTEGER: 'INTEGER'
            }
            sql = """
                        INSERT INTO analysis_variable (
                            run_id, var_name, var_group, var_type, lower_bound, upper_bound,
                            slot, home_team, away_team, team, city
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id, var_name) DO NOTHING
                    """
            self.cur.execute(sql, (
                self.run_id,
                var_obj.VarName,
                var_group,
                var_type_map.get(var_obj.VType, 'UNKNOWN'),
                var_obj.LB,
                var_obj.UB,
                kwargs.get('slot'),
                kwargs.get('home_team'),
                kwargs.get('away_team'),
                kwargs.get('team'),
                kwargs.get('city'),
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"analyzer.add_variable 에서 에러 발생: {e},{e.__traceback__}", exc_info=True)

            raise

    def add_constraint(self, model_obj, constr_obj, eq_group, **kwargs):
        """Gurobi 제약 객체(gp.Constr)를 받아 DB에 저장하고, matrix 항목도 함께 생성합니다."""

        try:
            model_obj.update()
            eq_name = constr_obj.ConstrName

            # 일반 제약(Linear Constraint) 처리
            if isinstance(constr_obj, gp.Constr):
                sign_map = {'<': '<=', '>': '>=', '=': '=='}
                sign = sign_map.get(constr_obj.Sense, '?')
                rhs = constr_obj.RHS
                eq_type = 'Linear'

                eq_sql = """
                                INSERT INTO analysis_equation (
                                    run_id, eq_name, eq_group, eq_type, sign, rhs,
                                    slot, home_team, away_team, team, city
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """
                self.cur.execute(eq_sql, (
                    self.run_id, eq_name, eq_group, eq_type, sign, rhs,
                    kwargs.get('slot'), kwargs.get('home_team'), kwargs.get('away_team'),
                    kwargs.get('team'), kwargs.get('city'),
                ))
                equation_id = self.cur.fetchone()[0]

                row = model_obj.getRow(constr_obj)
                matrix_data = []
                for i in range(row.size()):
                    var = row.getVar(i)
                    coeff = row.getCoeff(i)

                    self.cur.execute(
                        "SELECT id FROM analysis_variable WHERE run_id = %s AND var_name = %s",
                        (self.run_id, var.VarName)
                    )
                    var_result = self.cur.fetchone()
                    if var_result is None:
                        logger.warning(f"변수 {var.VarName}가 DB에 없음 → matrix 항목 건너뜀")
                        continue

                    variable_id = var_result[0]
                    matrix_data.append((self.run_id, variable_id, equation_id, coeff))

                matrix_sql = """
                               INSERT INTO analysis_matrixentry (run_id, variable_id, equation_id, coefficient)
                               VALUES %s
                           """
                execute_values(self.cur, matrix_sql, matrix_data)

            # 일반 제약(General Constraint) 처리 (예: AND, ABS 등)
            elif isinstance(constr_obj, gp.GenConstr):
                eq_type = f"General_{gp.GENCONSTR_NAMES.get(constr_obj.Type, 'UNKNOWN')}"
                eq_sql = """
                                INSERT INTO analysis_equation (
                                    run_id, eq_name, eq_group, eq_type, sign, rhs,
                                    slot, team
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (run_id, eq_name) DO NOTHING
                            """
                self.cur.execute(eq_sql, (
                    self.run_id, eq_name, eq_group, eq_type, "", 0.0,
                    kwargs.get('slot'), kwargs.get('team')
                ))

            self.conn.commit()
        except Exception as e:
            logger.error(f"analyzer.add_constraint에서 에러 발생: {e}", exc_info=True)
            raise

    def update_variable_results(self, model_obj):
        """최적화가 끝난 모델 객체를 받아 모든 변수의 결과 값을 DB에 업데이트합니다."""
        update_data = [
            (var.X, self.run_id, var.VarName) for var in model_obj.getVars()
        ]

        sql = """
                    UPDATE analysis_variable
                    SET result_value = %s
                    WHERE run_id = %s AND var_name = %s
                """
        self.cur.executemany(sql, update_data)
        self.conn.commit()
        logger.info(f"Updated results for {len(update_data)} variables.")

    def close(self):
        self.cur.close()
        self.conn.close()


class OrtoolsCpModelAnalyzer(BaseOrtoolsCpModelAnalyzer):
    """
    OR-Tools CP-SAT 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """
    def __init__(self, run_id=None):
        super().__init__(run_id)

    def add_variable(self, var_obj, var_group, **kwargs):
        """
        OR-Tools 변수 객체를 받아 DB에 저장합니다.
        상한/하한 등은 kwargs를 통해 직접 전달받아야 합니다.
        """
        logger.solve(f"analysis_mode:add_variable-{var_group}")
        var_type = 'UNKNOWN'
        # 변수 타입 추정 (BoolVar, IntVar)
        try:
            if 'Bool' in str(type(var_obj)):
                var_type = 'BINARY'
            elif 'Int' in str(type(var_obj)):
                var_type = 'INTEGER'
            sql = """
                INSERT INTO analysis_variable (
                    run_id, var_name, var_group, var_type, lower_bound, upper_bound,
                    slot, home_team, away_team, team, city
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, var_name) DO NOTHING
            """
            self.cur.execute(sql, (
                self.run_id,
                var_obj.Name(),
                var_group,
                var_type,
                kwargs.get('lower_bound'),
                kwargs.get('upper_bound'),
                kwargs.get('slot'),
                kwargs.get('home_team'),
                kwargs.get('away_team'),
                kwargs.get('team'),
                kwargs.get('city'),
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding variable {var_obj.Name()}: {e}", exc_info=True)
            raise

    def add_constraint(self, eq_name, eq_type, sign, rhs, **kwargs):
        """
        OR-Tools CP-SAT 제약 조건 정보를 DB에 저장합니다.
        kwargs를 통해 slot, home_team 등 추가 정보를 받습니다.
        """
        try:
            eq_sql = """
                INSERT INTO analysis_equation (
                    run_id, eq_name, eq_group, eq_type, sign, rhs,
                    slot, home_team, away_team, team, city
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cur.execute(eq_sql, (
                self.run_id,
                eq_name,
                kwargs.get('eq_group', 'default'),
                eq_type,
                sign,
                rhs,
                kwargs.get('slot'),
                kwargs.get('home_team'),
                kwargs.get('away_team'),
                kwargs.get('team'),
                kwargs.get('city'),
            ))
            equation_id = self.cur.fetchone()[0]
            self.conn.commit()
            return equation_id
        except Exception as e:
            logger.error(f"analyzer.add_constraint {eq_name} 에서 에러 발생: {e}", exc_info=True)
            raise

    def add_matrix_entry(self, variable_id, equation_id, coefficient):
        """
        OR-Tools CP-SAT 행렬 항목 정보를 DB에 저장합니다.
        """
        try:
            matrix_sql = """
                INSERT INTO analysis_matrixentry (run_id, variable_id, equation_id, coefficient)
                VALUES (%s, %s, %s, %s)
            """
            self.cur.execute(matrix_sql, (self.run_id, variable_id, equation_id, coefficient))
            self.conn.commit()
        except Exception as e:
            logger.error(f"analyzer.add_matrix_entry 에서 에러 발생: {e}", exc_info=True)
            raise

    def update_variable_results(self, solver_obj, variables_to_log):
        """
        OR-Tools CP-SAT 솔버의 최적화 결과를 DB에 업데이트합니다.
        solver_obj: 해결이 완료된 cp_model.CpSolver() 객체
        variables_to_log: 값을 기록할 OR-Tools 변수 객체들의 리스트
        """
        try:
            # executemany에 사용될 (결과값, run_id, 변수이름) 튜플 리스트를 생성합니다.
            update_data = [
                (solver_obj.Value(var), self.run_id, var.Name())
                for var in variables_to_log if var is not None
            ]

            if not update_data:
                logger.warning("No variable values to update.")
                return

            sql = """
                UPDATE analysis_variable
                SET result_value = %s
                WHERE run_id = %s AND var_name = %s
            """
            # with 문을 사용하여 커밋과 커서 관리를 자동화합니다.
            with self.conn, self.conn.cursor() as cur:
                for row_data in update_data:
                    cur.execute(sql, row_data)

            logger.info(f"Updated results for {len(update_data)} variables.")
        except Exception as e:
            logger.error(f"Error in OR-Tools update_variable_results: {e}", exc_info=True)
            self.conn.rollback()  # 오류 발생 시 롤백
            raise