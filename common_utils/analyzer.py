
import logging
import uuid
import gurobipy as gp
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'optdemo_user',
    'password': 'optdemo_password',
    'dbname': 'optdemo_dev'
}

class GurobiModelAnalyzer:
    """
    Gurobi 최적화 모델의 구성 요소를 데이터베이스에 기록하는 클래스.
    """

    def __init__(self, run_id=None):
        """
        db_config: dict with keys 'host', 'port', 'user', 'password', 'dbname'
        """
        self.run_id = '1f21c0a0-d161-4c60-b54b-462738f841b3' #run_id if run_id else str(uuid.uuid4())
        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = False  # manual commit
        self.cur = self.conn.cursor()

        logger.info(f"Initializing GurobiModelAnalyzer with run_id: {self.run_id}")
        self._delete_existing()

    def _delete_existing(self):
        self.cur.execute("DELETE FROM analysis_matrixentry WHERE run_id = %s", (self.run_id,))
        self.cur.execute("DELETE FROM analysis_variable WHERE run_id = %s", (self.run_id,))
        self.cur.execute("DELETE FROM analysis_equation WHERE run_id = %s", (self.run_id,))
        self.conn.commit()

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
        except gp.GurobiError:
            logger.error("analyzer.add_constraint 에서 Gurobi 모델 업데이트 실패", exc_info=True)

        try:
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
            logger.error(f"analyzer.add_constraint {eq_name} 에서 에러 발생: {e}", exc_info=True)
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