import pandas as pd
from collections import defaultdict, deque
import random
import datetime
import time
import copy
import math
import sys
from tqdm import tqdm # 引入tqdm库用于显示进度条

# --- 安全时间计算函数 ---
def safe_time_diff_seconds(time1, time2, max_seconds=86400*365*10):
    """安全计算两个时间的差值（秒），避免溢出"""
    try:
        if pd.isna(time1) or pd.isna(time2):
            return float('inf')

        min_date = datetime.datetime(1900, 1, 1)
        max_date = datetime.datetime(2100, 1, 1)

        if time1 < min_date or time1 > max_date or time2 < min_date or time2 > max_date:
            return float('inf')

        diff = time2 - time1
        seconds = diff.total_seconds()

        if abs(seconds) > max_seconds:
            return float('inf') if seconds > 0 else -float('inf')

        return seconds
    except (OverflowError, ValueError, TypeError):
        return float('inf')

# --- 全局参数与新约束常量 ---
PROACTIVE_POSITIONING_START_DATE = datetime.datetime(2025, 5, 26, 0, 0)
OPTIMIZATION_START_DATE = datetime.datetime(2025, 5, 29, 0, 0)
OPTIMIZATION_END_DATE = datetime.datetime(2025, 6, 4, 23, 59, 59)
MAX_FLIGHTS_PER_FDP = 4
MAX_TOTAL_TASKS_PER_FDP = 6
MAX_FDP_FLY_TIME_MINS = 8 * 60
MAX_FDP_DUTY_TIME_HOURS = 12
CONN_TIME_DIFF_AIRCRAFT_HOURS = 3
CONN_TIME_BUS_HOURS = 2
MIN_REST_TIME_HOURS = 12

# --- 目标函数权重与罚分项 ---
SCORE_WEIGHT_FLIGHT_TIME_OPT = 5
SCORE_PENALTY_UNCOVERED_FLIGHT_OPT = 1000
SCORE_PENALTY_INVALID_HISTORICAL_START_OPT = 500
SCORE_PENALTY_TELEPORT_OPT = 2000

# --- 最终官方评分标准 ---
FINAL_SCORE_BASE_MULTIPLIER = 1000
FINAL_SCORE_PENALTY_UNCOVERED = 5
FINAL_SCORE_PENALTY_NEW_LAYOVER = 10
FINAL_SCORE_PENALTY_FOREIGN_OVERNIGHT = 0.5
FINAL_SCORE_PENALTY_POSITIONING = 0.5
FINAL_SCORE_PENALTY_VIOLATION = 10

# --- 抢任务机制参数 (修改) ---
TARGET_COVERAGE_RATE = 0.99
# 将原有的迭代次数分配到两个阶段
TASK_STEALING_ITERATIONS_STAGE_1 = 10000
TASK_STEALING_ITERATIONS_STAGE_2 = 15000


class CrewScheduler:
    """
    管理机组排班优化问题的高效启发式算法
    (两日闲人交集置位 + 快速航班优先构建 + 中间改进 + 两阶段抢任务优化 + ALNS + 模拟退火)
    """
    def __init__(self, match_df, flight_df, bus_df, crew_df, layover_df, ground_duty_df):
        self.match_df = match_df
        self.flight_df = flight_df
        self.bus_df = bus_df
        self.crew_df = crew_df
        self.layover_stations = set(layover_df['airport'])
        self.ground_duty_df = ground_duty_df

        self.segment_details = {}
        self.crew_to_legs = {}
        self.leg_to_crews = defaultdict(list)
        self.crew_stay_stations = pd.Series(crew_df.stayStation.values, index=crew_df.crewId).to_dict()
        self.crew_bases = pd.Series(crew_df.base.values, index=crew_df.crewId).to_dict()
        self.fixed_ground_duties = defaultdict(list)

        self.positioning_options = defaultdict(list)
        self.airport_to_flights = defaultdict(list)
        self.flights_to_airport = defaultdict(list) # 新增：用于反向搜索
        self.crew_qualification_counts = {}

        self.ruin_operators = {
            'random_ruin': self._random_ruin,
            'worst_fdp_ruin': self._worst_fdp_ruin,
            'related_ruin': self._related_ruin,
            'uncovered_flight_ruin': self._uncovered_flight_ruin
        }
        self.operator_weights = {name: 1.0 for name in self.ruin_operators}
        self.operator_scores = {name: 0.0 for name in self.ruin_operators}
        self.operator_uses = {name: 0 for name in self.ruin_operators}

        self.alns_reaction_factor = 0.7
        self.alns_score_global_best = 10
        self.alns_score_better = 5
        self.alns_score_accepted = 2
        self.alns_segment_length = 100

        self.rcl_k = 1

        self.all_violations_details = []
        self.patched_crew_ids = []
        self.proactive_recommendations = {}
        self.proactive_moves = []

        print("1. 初始化和数据预处理 (性能优化中)...")
        self._preprocess_data()

    def _preprocess_data(self):
        """数据预处理"""
        for df, date_cols in [(self.flight_df, ['std', 'sta']), (self.bus_df, ['td', 'ta']), (self.ground_duty_df, ['startTime', 'endTime'])]:
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for row in self.flight_df.to_dict('records'):
            self.segment_details[row['id']] = {
                'type': 'flight', 'depaAirport': row['depaAirport'], 'arriAirport': row['arriAirport'],
                'std': row['std'], 'sta': row['sta'], 'aircraftNo': row['aircraftNo'], 'flyTime': row['flyTime']
            }
        for row in self.bus_df.to_dict('records'):
            self.segment_details[row['id']] = {
                'type': 'bus', 'depaAirport': row['depaAirport'], 'arriAirport': row['arriAirport'],
                'std': row['td'], 'sta': row['ta'], 'aircraftNo': None, 'flyTime': 0
            }
        for row in self.ground_duty_df.to_dict('records'):
            self.segment_details[row['id']] = {
                'type': 'ground', 'depaAirport': row['airport'], 'arriAirport': row['airport'],
                'std': row['startTime'], 'sta': row['endTime'], 'aircraftNo': None, 'flyTime': 0,
                'isDuty': row['isDuty']
            }
            if row['isDuty'] == 1:
                self.fixed_ground_duties[row['crewId']].append(row['id'])

        self.crew_to_legs = self.match_df.groupby('crewId')['legId'].apply(set).to_dict()
        for crew_id, legs in self.crew_to_legs.items():
            for leg_id in legs:
                self.leg_to_crews[leg_id].append(crew_id)

        self.crew_qualification_counts = self.match_df.groupby('crewId')['legId'].count().to_dict()

        all_positioning_tasks = pd.concat([
            self.flight_df[['id', 'depaAirport', 'arriAirport', 'std', 'sta']],
            self.bus_df[['id', 'depaAirport', 'arriAirport', 'td', 'ta']].rename(columns={'td':'std', 'ta':'sta'})
        ])
        all_positioning_tasks['duration'] = all_positioning_tasks['sta'] - all_positioning_tasks['std']
        for row in all_positioning_tasks.to_dict('records'):
            if pd.notna(row['std']) and pd.notna(row['sta']):
                key = (row['depaAirport'], row['arriAirport'])
                self.positioning_options[key].append({'id': row['id'], 'duration': row['duration']})
        for key in self.positioning_options:
            self.positioning_options[key].sort(key=lambda x: x['duration'])

        sorted_flights_df = self.flight_df.sort_values('std')
        for row in sorted_flights_df.to_dict('records'):
            if pd.notna(row['std']):
                self.airport_to_flights[row['depaAirport']].append(row['id'])
            if pd.notna(row['sta']):
                self.flights_to_airport[row['arriAirport']].append(row['id'])

        print("数据预处理完成！")

    # =================================================================================
    # ===== 时间逆转策略模块 (新增 & 修复) =====
    # =================================================================================

    def _calculate_past_opportunities(self, crew_id, before_time, at_airport, uncovered_flights):
        """
        【新增】计算一个机组在特定时间点和机场之前，有多少更早的任务机会。
        """
        if not pd.notna(before_time):
            return 0

        opportunity_count = 0
        look_back_limit = before_time - datetime.timedelta(hours=24)
        crew_legs = self.crew_to_legs.get(crew_id, set())

        potential_flights = self.flights_to_airport.get(at_airport, [])

        for flight_id in potential_flights:
            if flight_id in uncovered_flights and flight_id in crew_legs:
                flight_sta = self.segment_details[flight_id]['sta']
                if pd.notna(flight_sta) and look_back_limit <= flight_sta < before_time:
                    opportunity_count += 1
        return opportunity_count

    def _is_schedule_valid_backward(self, crew_id, backward_schedule):
        """
        【新增】镜像验证函数，用于反向优化。它会调用升级后的_is_schedule_valid，但不检查历史衔接。
        """
        all_fdps = []
        for fdp_bwd in backward_schedule:
            all_fdps.append(list(reversed(fdp_bwd)))

        # 调用核心验证逻辑，但明确告知不要检查与历史stayStation的衔接
        violations = self._is_schedule_valid(crew_id, all_fdps, check_history=False)
        return not violations

    def _build_fdp_from_seed_backward(self, seed_flight_id, crew_id, available_flights):
        """
        【新增】从一个种子航班开始，向过去（更早的时间）贪心构建一个合规的FDP。
        """
        work_block = [(seed_flight_id, 0)]

        while sum(1 for seg_id, is_ddh in work_block if self.segment_details[seg_id]['type'] == 'flight' and is_ddh == 0) < MAX_FLIGHTS_PER_FDP:
            current_start_seg_id = work_block[-1][0]
            current_start_seg = self.segment_details[current_start_seg_id]

            next_flight_candidates = []
            potential_prev_flights = self.flights_to_airport.get(current_start_seg['depaAirport'], [])

            for prev_flight_id in potential_prev_flights:
                if prev_flight_id in available_flights and prev_flight_id in self.crew_to_legs.get(crew_id, set()):
                    temp_fdp = work_block + [(prev_flight_id, 0)]
                    if self._is_fdp_valid(list(reversed(temp_fdp))):
                        next_flight_candidates.append(prev_flight_id)

            if not next_flight_candidates: break

            best_prev_flight = max(next_flight_candidates, key=lambda fid: self.segment_details[fid]['sta'])
            work_block.append((best_prev_flight, 0))
            available_flights.remove(best_prev_flight)

        if self._is_fdp_valid(list(reversed(work_block))):
            return work_block
        else:
            if self._is_fdp_valid([(seed_flight_id, 0)]):
                return [(seed_flight_id, 0)]
            return None

    def _run_backward_optimization(self):
        """
        【新增】执行一次从后往前的、基于多标准贪心策略的排班，以确定理论最优的机组初始布局。
        """
        print("  - [反向优化 2.0] 正在从第7天到第1天，使用高级贪心策略反向模拟...")

        backward_solution = defaultdict(list)
        flights_in_window = {
            fid for fid, det in self.segment_details.items()
            if det['type'] == 'flight'
        }
        uncovered_flights_bwd = flights_in_window.copy()

        sorted_flights_bwd = sorted(
            list(uncovered_flights_bwd),
            key=lambda fid: self.segment_details[fid]['sta'],
            reverse=True
        )

        crew_bwd_status = {
            cid: {'next_available_std': datetime.datetime.max, 'location': None} for cid in self.crew_df['crewId']
        }

        for flight_id in tqdm(sorted_flights_bwd, desc="    反向智能分配", unit="个航班"):
            if flight_id not in uncovered_flights_bwd: continue

            flight_seg = self.segment_details[flight_id]
            candidate_assignments = []

            for crew_id in self.leg_to_crews.get(flight_id, []):
                status = crew_bwd_status[crew_id]
                if flight_seg['sta'] >= status['next_available_std']: continue

                backward_fdp = self._build_fdp_from_seed_backward(flight_id, crew_id, uncovered_flights_bwd.copy())
                if not backward_fdp: continue

                temp_bwd_schedule = backward_solution.get(crew_id, []) + [backward_fdp]
                if self._is_schedule_valid_backward(crew_id, temp_bwd_schedule):
                    true_start_seg = self.segment_details[backward_fdp[-1][0]]
                    qual_count = self.crew_qualification_counts.get(crew_id, 9999)
                    return_bonus = 0 if true_start_seg['depaAirport'] == self.crew_bases.get(crew_id) else 1
                    opportunity_score = self._calculate_past_opportunities(
                        crew_id, true_start_seg['std'], true_start_seg['depaAirport'], uncovered_flights_bwd
                    )

                    candidate_assignments.append({
                        'crew_id': crew_id, 'fdp': backward_fdp, 'qual_count': qual_count,
                        'return_bonus': return_bonus, 'opportunity_score': opportunity_score,
                        'cost': (status['next_available_std'] - flight_seg['sta']).total_seconds()
                    })

            if candidate_assignments:
                # 【已修改】调整贪心策略优先级为“资质优先”
                candidate_assignments.sort(key=lambda x: (
                    x['return_bonus'],      # 1. 是否能返回基地 (0 = 是, 1 = 否)
                    x['qual_count'],        # 2. 资质优先 (越少越优先)
                    -x['opportunity_score'],# 3. 未来机会 (越多越优先)
                    x['cost']               # 4. 成本 (越低越优先)
                ))
                selected_assignment = candidate_assignments[0]

                assigned_crew = selected_assignment['crew_id']
                assigned_fdp = selected_assignment['fdp']

                backward_solution[assigned_crew].append(assigned_fdp)
                for seg_id, is_ddh in assigned_fdp:
                    if is_ddh == 0: uncovered_flights_bwd.discard(seg_id)

                true_start_time = self.segment_details[assigned_fdp[-1][0]]['std']
                crew_bwd_status[assigned_crew]['next_available_std'] = true_start_time - datetime.timedelta(hours=MIN_REST_TIME_HOURS)

        if not flights_in_window:
             final_bwd_coverage = 100.0
        else:
            final_bwd_coverage = (len(flights_in_window) - len(uncovered_flights_bwd)) / len(flights_in_window) * 100
        print(f"  - [反向优化] 模拟完成。理论覆盖率: {final_bwd_coverage:.2f}%")

        ideal_stay_stations = {}
        for crew_id in self.crew_df['crewId']:
            all_tasks_bwd = [task for fdp in backward_solution.get(crew_id, []) for task in fdp]
            if not all_tasks_bwd:
                ideal_stay_stations[crew_id] = self.crew_bases.get(crew_id)
                continue

            earliest_task = min(all_tasks_bwd, key=lambda x: self.segment_details[x[0]]['std'])
            ideal_stay_stations[crew_id] = self.segment_details[earliest_task[0]]['depaAirport']

        return ideal_stay_stations, backward_solution, uncovered_flights_bwd

    def _build_hybrid_initial_solution(self):
        """
        【新增】构建一个融合了“反向最优”和“正向贪心”的混合初始解。
        """
        print("\n[混合初始化策略 4.0] 开始执行...")

        ideal_stay_stations, backward_solution, bwd_uncovered = self._run_backward_optimization()

        hybrid_solution = defaultdict(list)
        flights_to_replan = set()
        crews_to_replan = []

        print("  - [锁定与识别] 正在识别可锁定机组与需重排机组...")
        for crew_id in self.crew_df['crewId']:
            actual_location = self.crew_stay_stations.get(crew_id)
            target_location = ideal_stay_stations.get(crew_id)

            can_be_locked = False

            proposed_full_schedule_fdps = []

            # 获取该机组在反向优化中的完整排班
            crew_bwd_schedule = backward_solution.get(crew_id, [])

            # 情况一：机组已在理想位置
            if actual_location == target_location:
                proposed_full_schedule_fdps = [list(reversed(fdp)) for fdp in crew_bwd_schedule]
                if not self._is_schedule_valid(crew_id, proposed_full_schedule_fdps):
                     can_be_locked = True
            # 情况二：机组可以被成功置位到理想位置
            elif actual_location != target_location and pd.notna(target_location):
                time_window_start = PROACTIVE_POSITIONING_START_DATE
                time_window_end = OPTIMIZATION_START_DATE
                pos_task_id = self._get_best_positioning_task(
                    actual_location, target_location, time_window_start, time_window_end
                )
                if pos_task_id:
                    # 【安全锁定】构建预锁定计划并进行终审
                    proposed_full_schedule_fdps.append([(pos_task_id, 1)])
                    for fdp_bwd in crew_bwd_schedule:
                        proposed_full_schedule_fdps.append(list(reversed(fdp_bwd)))

                    if not self._is_schedule_valid(crew_id, proposed_full_schedule_fdps):
                        can_be_locked = True

            if can_be_locked:
                hybrid_solution[crew_id] = proposed_full_schedule_fdps
            else:
                crews_to_replan.append(crew_id)
                for fdp in crew_bwd_schedule:
                    for seg_id, is_ddh in fdp:
                        if is_ddh == 0:
                            flights_to_replan.add(seg_id)

        final_flights_to_replan = bwd_uncovered.union(flights_to_replan)

        print(f"  - [锁定与识别] 完成。{len(hybrid_solution)}名机组的计划已锁定。")

        if crews_to_replan and final_flights_to_replan:
            print(f"  - [正向修复] 将为剩余的 {len(crews_to_replan)} 名机组，重新规划 {len(final_flights_to_replan)} 个航班。")

            crew_initial_status_replan = {
                cid: {'end': OPTIMIZATION_START_DATE - datetime.timedelta(hours=MIN_REST_TIME_HOURS), 'location': self.crew_stay_stations.get(cid)}
                for cid in crews_to_replan
            }

            hybrid_solution, final_uncovered_flights = self._build_flight_centric_initial_solution(
                crew_initial_status_replan,
                target_crews=crews_to_replan,
                target_flights=final_flights_to_replan,
                existing_solution=hybrid_solution
            )
        else:
            print("  - [正向修复] 无需修复，所有机组均已锁定或无任务。")
            final_uncovered_flights = final_flights_to_replan

        return hybrid_solution, final_uncovered_flights

    # =================================================================================
    # ===== 原有正向优化模块 (微创改造 & 未修改) =====
    # =================================================================================

    def _calculate_objective_for_opt(self, solution, uncovered_flights):
        """计算用于优化过程的内部目标函数值"""
        total_flight_time_minutes = 0
        teleport_penalty = 0
        historical_location_penalty = 0

        for crew_id, fdps in solution.items():
            for fdp in fdps:
                for seg_id, is_ddh in fdp:
                    segment = self.segment_details[seg_id]
                    if segment['type'] == 'flight' and is_ddh == 0:
                        if pd.notna(segment['std']) and pd.notna(segment['sta']) and segment['std'] < OPTIMIZATION_END_DATE and segment['sta'] > OPTIMIZATION_START_DATE:
                            total_flight_time_minutes += segment['flyTime']

            all_duties = []
            for fdp in fdps:
                if not fdp or any(pd.isna(self.segment_details[seg_id]['std']) for seg_id, _ in fdp): continue
                all_duties.append({'type': 'FDP', 'start': self.segment_details[fdp[0][0]]['std'], 'end': self.segment_details[fdp[-1][0]]['sta'], 'obj': fdp})
            for ground_duty_id in self.fixed_ground_duties.get(crew_id, []):
                gd_seg = self.segment_details[ground_duty_id]
                if pd.notna(gd_seg['std']) and pd.notna(gd_seg['sta']):
                    all_duties.append({'type': 'Ground', 'start': gd_seg['std'], 'end': gd_seg['sta'], 'obj': ground_duty_id})

            if not all_duties: continue
            all_duties.sort(key=lambda x: x['start'])

            first_duty_details = self.segment_details[all_duties[0]['obj'][0][0]] if all_duties[0]['type'] == 'FDP' else self.segment_details[all_duties[0]['obj']]
            historical_station = self.crew_stay_stations.get(crew_id)
            if first_duty_details['depaAirport'] != historical_station:
                teleport_penalty += SCORE_PENALTY_TELEPORT_OPT
            elif historical_station not in self.layover_stations:
                historical_location_penalty += SCORE_PENALTY_INVALID_HISTORICAL_START_OPT

            if len(all_duties) > 1:
                for i in range(len(all_duties) - 1):
                    prev_duty_details = self.segment_details[all_duties[i]['obj'][-1][0]] if all_duties[i]['type'] == 'FDP' else self.segment_details[all_duties[i]['obj']]
                    next_duty_details = self.segment_details[all_duties[i+1]['obj'][0][0]] if all_duties[i+1]['type'] == 'FDP' else self.segment_details[all_duties[i+1]['obj']]
                    if prev_duty_details['arriAirport'] != next_duty_details['depaAirport']:
                        teleport_penalty += SCORE_PENALTY_TELEPORT_OPT

        uncovered_penalty = len(uncovered_flights) * SCORE_PENALTY_UNCOVERED_FLIGHT_OPT
        score = (total_flight_time_minutes * SCORE_WEIGHT_FLIGHT_TIME_OPT) - uncovered_penalty - teleport_penalty - historical_location_penalty
        return score

    def _is_fdp_valid(self, fdp, return_violations=False):
        """检查单个飞行值勤日（FDP）是否满足所有规则, 可选择返回详细违规列表"""
        violations = []
        if not fdp:
            return [] if return_violations else True

        for seg_id, _ in fdp:
            if pd.isna(self.segment_details[seg_id]['std']) or pd.isna(self.segment_details[seg_id]['sta']):
                if return_violations: violations.append({'type': 'FDP内部任务时间无效', 'details': f'任务 {seg_id} 缺少时间'})
                else: return False
        if not return_violations and violations: return False

        start_airport = self.segment_details[fdp[0][0]]['depaAirport']
        end_airport = self.segment_details[fdp[-1][0]]['arriAirport']
        if start_airport not in self.layover_stations or end_airport not in self.layover_stations:
            if return_violations: violations.append({'type': 'FDP起降点非法', 'details': f'起于{start_airport}或止于{end_airport}，非可过夜机场'})
            else: return False

        if len(fdp) > MAX_TOTAL_TASKS_PER_FDP:
            if return_violations: violations.append({'type': 'FDP总任务数超限', 'details': f'共{len(fdp)}个任务 (上限{MAX_TOTAL_TASKS_PER_FDP})'})
            else: return False

        flight_tasks_count = sum(1 for seg_id, is_ddh in fdp if self.segment_details[seg_id]['type'] == 'flight' and is_ddh == 0)
        if flight_tasks_count > MAX_FLIGHTS_PER_FDP:
            if return_violations: violations.append({'type': 'FDP飞行任务数超限', 'details': f'共{flight_tasks_count}个飞行任务 (上限{MAX_FLIGHTS_PER_FDP})'})
            else: return False

        has_operating_task = any(is_ddh == 0 for _, is_ddh in fdp)
        if flight_tasks_count == 0 and has_operating_task:
            if return_violations: violations.append({'type': 'FDP无飞行任务', 'details': '一个包含执飞任务的值勤日必须包含飞行任务'})
            else: return False

        total_flight_time = sum(self.segment_details[seg_id]['flyTime'] for seg_id, is_ddh in fdp if self.segment_details[seg_id]['type'] == 'flight' and is_ddh == 0)
        if total_flight_time > MAX_FDP_FLY_TIME_MINS:
            if return_violations: violations.append({'type': 'FDP飞行时间超限', 'details': f'总飞时{total_flight_time}分钟 (上限{MAX_FDP_FLY_TIME_MINS})'})
            else: return False

        first_task_start_time = self.segment_details[fdp[0][0]]['std']
        last_flight_task = next((seg for seg in reversed(fdp) if self.segment_details[seg[0]]['type'] == 'flight' and seg[1] == 0), None)

        last_task_arrival_time = self.segment_details[fdp[-1][0]]['sta'] if not last_flight_task else self.segment_details[last_flight_task[0]]['sta']

        if pd.isna(last_task_arrival_time):
                     if return_violations: violations.append({'type': 'FDP结束时间无效', 'details': '无法确定执勤时间'})
                     else: return False
        elif (last_task_arrival_time - first_task_start_time) > datetime.timedelta(hours=MAX_FDP_DUTY_TIME_HOURS):
            duty_hours = ((last_task_arrival_time - first_task_start_time).total_seconds() / 3600)
            if return_violations: violations.append({'type': 'FDP执勤时间超限', 'details': f'总执勤{duty_hours:.2f}小时 (上限{MAX_FDP_DUTY_TIME_HOURS})'})
            else: return False

        for i in range(len(fdp) - 1):
            seg1_id, _ = fdp[i]
            seg2_id, _ = fdp[i+1]
            seg1 = self.segment_details[seg1_id]
            seg2 = self.segment_details[seg2_id]
            if seg1['sta'] >= seg2['std']:
                if return_violations: violations.append({'type': 'FDP内部任务重叠', 'details': f'任务{seg1_id}结束于{seg1["sta"]}，任务{seg2_id}开始于{seg2["std"]}'})
                else: return False; break

            min_conn_time = datetime.timedelta(minutes=0)
            if seg1['type'] == 'bus' or seg2['type'] == 'bus': min_conn_time = datetime.timedelta(hours=CONN_TIME_BUS_HOURS)
            elif seg1['type'] == 'flight' and seg2['type'] == 'flight' and pd.notna(seg1['aircraftNo']) and pd.notna(seg2['aircraftNo']) and seg1['aircraftNo'] != seg2['aircraftNo']:
                min_conn_time = datetime.timedelta(hours=CONN_TIME_DIFF_AIRCRAFT_HOURS)

            if seg2['std'] - seg1['sta'] < min_conn_time:
                conn_hours = (seg2['std'] - seg1['sta']).total_seconds() / 3600
                if return_violations: violations.append({'type': 'FDP内部衔接时间不足', 'details': f'任务{seg1_id}与{seg2_id}之间衔接{conn_hours:.2f}小时 (要求{min_conn_time.total_seconds()/3600})'})
                else: return False; break

        return violations if return_violations else not violations

    def _is_schedule_valid(self, crew_id, fdps, check_history=True):
        """
        【升级】检查一个机组的完整排班是否满足所有规则，引入条件性休息规则。
        """
        violation_list = []
        all_duties_for_crew = []

        for fdp in fdps:
            fdp_violations = self._is_fdp_valid(fdp, return_violations=True)
            if fdp_violations:
                for v in fdp_violations: v['crewId'] = crew_id
                violation_list.extend(fdp_violations)

            if not fdp or any(pd.isna(self.segment_details[seg_id]['std']) for seg_id, _ in fdp): continue
            all_duties_for_crew.append({
                'start': self.segment_details[fdp[0][0]]['std'], 'end': self.segment_details[fdp[-1][0]]['sta'],
                'start_loc': self.segment_details[fdp[0][0]]['depaAirport'], 'end_loc': self.segment_details[fdp[-1][0]]['arriAirport'],
                'is_flight_period_task': True, 'obj': fdp
            })

        for ground_duty_id in self.fixed_ground_duties.get(crew_id, []):
            gd_seg = self.segment_details[ground_duty_id]
            if pd.notna(gd_seg['std']) and pd.notna(gd_seg['sta']):
                all_duties_for_crew.append({
                    'start': gd_seg['std'], 'end': gd_seg['sta'],
                    'start_loc': gd_seg['depaAirport'], 'end_loc': gd_seg['arriAirport'],
                    'is_flight_period_task': False, 'obj': ground_duty_id
                })

        if not all_duties_for_crew: return violation_list
        all_duties_for_crew.sort(key=lambda x: x['start'])

        if check_history:
            first_duty_start_loc = all_duties_for_crew[0]['start_loc']
            if first_duty_start_loc != self.crew_stay_stations.get(crew_id):
                violation_list.append({'crewId': crew_id, 'type': '地点不衔接(历史)', 'details': f"计划从 {first_duty_start_loc} 开始, 但历史停留点为 {self.crew_stay_stations.get(crew_id)}"})

        for i in range(len(all_duties_for_crew) - 1):
            prev_duty, next_duty = all_duties_for_crew[i], all_duties_for_crew[i+1]
            if prev_duty['end_loc'] != next_duty['start_loc']:
                violation_list.append({'crewId': crew_id, 'type': '地点不衔接(任务间)', 'details': f"任务结束于 {prev_duty['end_loc']}, 但下一任务从 {next_duty['start_loc']} 开始"})

            # 【关键修复】应用条件性休息规则
            if next_duty['is_flight_period_task']:
                if (next_duty['start'] - prev_duty['end']) < datetime.timedelta(hours=MIN_REST_TIME_HOURS):
                    rest_hours = (next_duty['start'] - prev_duty['end']).total_seconds() / 3600
                    violation_list.append({'crewId': crew_id, 'type': '最短休息时间不足(12h)', 'details': f"两次任务间仅休息{rest_hours:.2f}小时"})

        # 【关键修复】“上四休二”规则检查
        duty_blocks = []
        if all_duties_for_crew:
            current_block = [all_duties_for_crew[0]]
            for i in range(len(all_duties_for_crew) - 1):
                prev_duty_end_time = all_duties_for_crew[i]['end']
                next_duty_start_time = all_duties_for_crew[i+1]['start']

                # 1. 确定第一个“完整”的休息日是哪天
                first_complete_rest_day = prev_duty_end_time.date()
                if prev_duty_end_time.time() > datetime.time(0, 0):
                    first_complete_rest_day += datetime.timedelta(days=1)

                # 2. 判断是否满足“休二”
                if next_duty_start_time.date() >= first_complete_rest_day + datetime.timedelta(days=2):
                    duty_blocks.append(current_block)
                    current_block = [all_duties_for_crew[i+1]]
                else:
                    current_block.append(all_duties_for_crew[i+1])
            duty_blocks.append(current_block)

        for i, block in enumerate(duty_blocks):
            is_flight_period = any(d['is_flight_period_task'] for d in block)
            if is_flight_period:
                last_flight_duty_index = -1
                for j in range(len(block) - 1, -1, -1):
                    if block[j]['is_flight_period_task']:
                        last_flight_duty_index = j; break
                if last_flight_duty_index == -1: continue
                core_flight_period = block[:last_flight_duty_index + 1]

                start_date = core_flight_period[0]['start'].date()
                end_date = core_flight_period[-1]['end'].date()
                if (end_date - start_date).days + 1 > 4:
                    violation_list.append({'crewId': crew_id, 'type': '飞行周期跨度超限', 'details': f'周期从 {start_date} 到 {end_date} 跨越 {(end_date - start_date).days + 1} 天 (上限4)'})

        return violation_list

    def _get_best_positioning_task(self, from_airport, to_airport, after_time=None, before_time=None):
        """快速查找最优置位任务"""
        key = (from_airport, to_airport)
        if key not in self.positioning_options:
            return None

        if after_time and before_time and after_time >= before_time:
            return None

        for task_option in self.positioning_options[key]:
            seg = self.segment_details[task_option['id']]
            is_after_valid = (after_time is None) or (pd.notna(seg['std']) and seg['std'] >= after_time)
            is_before_valid = (before_time is None) or (pd.notna(seg['sta']) and seg['sta'] <= before_time)
            if is_after_valid and is_before_valid:
                return task_option['id']
        return None

    def _build_fdp_from_seed(self, seed_flight_id, crew_id, available_flights, max_flights_in_fdp=MAX_FLIGHTS_PER_FDP):
        """贪心构建FDP"""
        work_block = [(seed_flight_id, 0)]
        current_flight_id = seed_flight_id

        while sum(1 for seg_id, is_ddh in work_block if self.segment_details[seg_id]['type'] == 'flight' and is_ddh == 0) < max_flights_in_fdp:
            last_seg = self.segment_details[current_flight_id]
            next_flight_candidates = []
            potential_next_flights = self.airport_to_flights.get(last_seg['arriAirport'], [])

            for next_flight_id in potential_next_flights:
                if next_flight_id in available_flights and next_flight_id in self.crew_to_legs.get(crew_id, set()):
                    temp_fdp = work_block + [(next_flight_id, 0)]
                    if self._is_fdp_valid(temp_fdp):
                        next_flight_candidates.append(next_flight_id)

            if not next_flight_candidates: break

            best_next_flight = next_flight_candidates[0]
            work_block.append((best_next_flight, 0))
            current_flight_id = best_next_flight
            available_flights.remove(best_next_flight)

        if self._is_fdp_valid(work_block):
            return work_block
        else:
            if self._is_fdp_valid([(seed_flight_id, 0)]):
                return [(seed_flight_id, 0)]
            return None

    def _calculate_future_opportunities(self, crew_id, after_time, at_airport, uncovered_flights):
        """计算一个机组在特定时间和地点后的潜在航班数量"""
        if not pd.notna(after_time):
            return 0

        opportunity_count = 0
        next_available_time = after_time + datetime.timedelta(hours=MIN_REST_TIME_HOURS)
        look_ahead_limit = next_available_time + datetime.timedelta(hours=24)

        potential_flights = self.airport_to_flights.get(at_airport, [])
        crew_legs = self.crew_to_legs.get(crew_id, set())

        for flight_id in potential_flights:
            if flight_id in uncovered_flights and flight_id in crew_legs:
                flight_std = self.segment_details[flight_id]['std']
                if pd.notna(flight_std) and next_available_time <= flight_std < look_ahead_limit:
                    opportunity_count += 1

        return opportunity_count

    def _build_flight_centric_initial_solution(self, crew_initial_status, target_crews=None, target_flights=None, existing_solution=None):
        """
        【微创改造】构建初始解。可选择在现有解基础上，只对特定机组和航班进行操作。
        """
        is_subset_problem = target_crews is not None and target_flights is not None

        if not is_subset_problem:
            target_crews = list(self.crew_df['crewId'])
            uncovered_flights = set(self.flight_df['id'])
            solution = defaultdict(list)
            print("2. [阶段一] 使用'资质优先'的半贪心策略（含激励锁定）构建快速初始解...")
        else:
            uncovered_flights = target_flights.copy()
            solution = existing_solution if existing_solution is not None else defaultdict(list)

        crew_latest_status = copy.deepcopy(crew_initial_status)

        sorted_flights = sorted(list(uncovered_flights), key=lambda fid: self.segment_details[fid]['std'])

        desc_text = "修复子问题" if is_subset_problem else "构建初始解"
        for flight_id in tqdm(sorted_flights, desc=desc_text, unit="个航班"):
            if flight_id not in uncovered_flights: continue

            candidate_assignments = []

            qualified_crews = [c for c in self.leg_to_crews.get(flight_id, []) if c in target_crews]

            for crew_id in qualified_crews:
                status = crew_latest_status[crew_id]
                available_time = status['end'] + datetime.timedelta(hours=MIN_REST_TIME_HOURS)
                flight_segment = self.segment_details[flight_id]
                if flight_segment['std'] < available_time: continue

                positioning_tasks = []
                positioning_end_time = status['end']

                if 'pre_assigned_task' in status:
                    positioning_tasks.append((status['pre_assigned_task'], 1))
                elif flight_segment['depaAirport'] != status['location']:
                    pos_task_id = self._get_best_positioning_task(status['location'], flight_segment['depaAirport'], status['end'], flight_segment['std'])
                    if pos_task_id:
                        positioning_tasks.append((pos_task_id, 1))
                        positioning_end_time = self.segment_details[pos_task_id]['sta']
                    else:
                        continue

                cost = (flight_segment['std'] - positioning_end_time).total_seconds()

                incentive_cost = 0
                recommended_crew = self.proactive_recommendations.get(flight_id)
                if recommended_crew:
                    if crew_id == recommended_crew:
                        incentive_cost = -1000000
                    else:
                        incentive_cost = 1000000
                cost += incentive_cost

                work_fdp = self._build_fdp_from_seed(flight_id, crew_id, uncovered_flights.copy())
                if work_fdp:
                    full_fdp = positioning_tasks + work_fdp
                    if not self._is_schedule_valid(crew_id, solution.get(crew_id, []) + [full_fdp]):

                        end_airport = self.segment_details[full_fdp[-1][0]]['arriAirport']
                        crew_base = self.crew_bases.get(crew_id)
                        return_bonus = 0 if end_airport == crew_base else 1

                        qual_count = self.crew_qualification_counts.get(crew_id, 9999)

                        fdp_end_time = self.segment_details[full_fdp[-1][0]]['sta']
                        opportunity_score = self._calculate_future_opportunities(
                            crew_id, fdp_end_time, end_airport, uncovered_flights
                        )

                        candidate_assignments.append({
                            'crew_id': crew_id, 'cost': cost, 'fdp': full_fdp,
                            'qual_count': qual_count, 'return_bonus': return_bonus,
                            'opportunity_score': opportunity_score
                        })

            if candidate_assignments:
                # 【已修改】明确贪心策略优先级为“基地优先”
                candidate_assignments.sort(key=lambda x: (
                    x['qual_count'],        # 1. 资质 (越少越优先)
                    x['return_bonus'],      # 2. 基地优先 (0 = 是, 1 = 否)
                    -x['opportunity_score'],# 3. 未来机会 (越多越优先)
                    x['cost']               # 4. 成本 (越低越优先)
                ))
                selected_assignment = candidate_assignments[0]

                assigned_crew = selected_assignment['crew_id']
                assigned_fdp = selected_assignment['fdp']

                solution[assigned_crew].append(assigned_fdp)
                crew_latest_status[assigned_crew].update({'end': self.segment_details[assigned_fdp[-1][0]]['sta'], 'location': self.segment_details[assigned_fdp[-1][0]]['arriAirport']})
                for seg_id, is_ddh in assigned_fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        uncovered_flights.discard(seg_id)

        return solution, uncovered_flights

    def _improve_solution_with_crew_centric_heuristic(self, solution, uncovered_flights):
        """中间改进环节: 采用“事务性”修改，确保覆盖率不会下降"""
        print(f"\n[中间改进阶段] 尝试为低任务量(<=4)机长优化排班...")

        crews_to_improve = []
        for crew_id in self.crew_df['crewId']:
            num_fdps = len(solution.get(crew_id, []))
            num_ground_duties = len(self.fixed_ground_duties.get(crew_id, []))
            if (num_fdps + num_ground_duties) <= 4:
                crews_to_improve.append(crew_id)

        if not crews_to_improve or not uncovered_flights:
            print("没有符合条件的机组或未覆盖航班，跳过中间改进。")
            return solution, uncovered_flights

        for crew_id in tqdm(crews_to_improve, desc="中间改进 (低任务量机长)", unit="名机长"):
            original_fdps = solution.get(crew_id, []).copy()

            original_covered_flights = set()
            for fdp in original_fdps:
                for seg_id, is_ddh in fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        original_covered_flights.add(seg_id)

            temp_uncovered = uncovered_flights.union(original_covered_flights)

            initial_end_time = OPTIMIZATION_START_DATE - datetime.timedelta(hours=MIN_REST_TIME_HOURS)
            initial_location = self.crew_stay_stations.get(crew_id)
            crew_ground_duties = self.fixed_ground_duties.get(crew_id, [])
            if crew_ground_duties:
                valid_duties = [d for d in crew_ground_duties if pd.notna(self.segment_details[d]['sta'])]
                if valid_duties:
                    last_duty_end_time = max(self.segment_details[did]['sta'] for did in valid_duties)
                    if pd.notna(last_duty_end_time) and last_duty_end_time > initial_end_time:
                        last_duty_id = max(valid_duties, key=lambda did: self.segment_details[did]['sta'])
                        last_duty_seg = self.segment_details[last_duty_id]
                        initial_end_time = last_duty_seg['sta']
                        initial_location = last_duty_seg['arriAirport']

            crew_status = {'end': initial_end_time, 'location': initial_location}

            new_fdps = []
            rebuild_uncovered = temp_uncovered.copy()

            while True:
                best_next_flight_info = None
                best_cost = float('inf')

                for flight_id in rebuild_uncovered:
                    if flight_id not in self.crew_to_legs.get(crew_id, set()):
                        continue

                    flight_segment = self.segment_details[flight_id]
                    available_time = crew_status['end'] + datetime.timedelta(hours=MIN_REST_TIME_HOURS)
                    if flight_segment['std'] < available_time: continue

                    positioning_tasks = []
                    positioning_end_time = crew_status['end']
                    if flight_segment['depaAirport'] != crew_status['location']:
                        pos_task_id = self._get_best_positioning_task(crew_status['location'], flight_segment['depaAirport'], available_time, flight_segment['std'])
                        if pos_task_id:
                            positioning_tasks.append((pos_task_id, 1))
                            positioning_end_time = self.segment_details[pos_task_id]['sta']
                        else:
                            continue

                    cost = (flight_segment['std'] - positioning_end_time).total_seconds()

                    if cost < best_cost:
                        potential_fdp = self._build_fdp_from_seed(flight_id, crew_id, rebuild_uncovered.copy())
                        if potential_fdp:
                            full_fdp = positioning_tasks + potential_fdp
                            if not self._is_schedule_valid(crew_id, new_fdps + [full_fdp]):
                                best_cost = cost
                                best_next_flight_info = {
                                    'fdp': full_fdp,
                                    'flights_covered': {seg_id for seg_id, is_ddh in potential_fdp if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight'}
                                }

                if best_next_flight_info:
                    fdp_to_add = best_next_flight_info['fdp']
                    new_fdps.append(fdp_to_add)
                    rebuild_uncovered -= best_next_flight_info['flights_covered']
                    crew_status['end'] = self.segment_details[fdp_to_add[-1][0]]['sta']
                    crew_status['location'] = self.segment_details[fdp_to_add[-1][0]]['arriAirport']
                else:
                    break

            newly_covered_flights = set()
            for fdp in new_fdps:
                for seg_id, is_ddh in fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        newly_covered_flights.add(seg_id)

            if len(newly_covered_flights) > len(original_covered_flights):
                solution[crew_id] = new_fdps
                uncovered_flights.difference_update(newly_covered_flights)
                flights_to_add_back = original_covered_flights - newly_covered_flights
                uncovered_flights.update(flights_to_add_back)

        if len(self.flight_df) > 0:
            final_coverage = (len(self.flight_df) - len(uncovered_flights)) / len(self.flight_df) * 100
            print(f"中间改进阶段完成！当前航班覆盖率: {final_coverage:.2f}%")

        return solution, uncovered_flights

    def _select_operator(self):
        """使用轮盘赌选择法根据权重选择一个破坏算子"""
        total_weight = sum(self.operator_weights.values())
        pick = random.uniform(0, total_weight)
        current = 0
        for name, weight in self.operator_weights.items():
            current += weight
            if current > pick:
                return name
        return random.choice(list(self.operator_weights.keys()))

    def _update_weights(self):
        """根据一个迭代段内的算子表现更新权重"""
        for name in self.operator_weights:
            if self.operator_uses[name] > 0:
                self.operator_weights[name] = (1 - self.alns_reaction_factor) * self.operator_weights[name] + \
                                                self.alns_reaction_factor * self.operator_scores[name] / self.operator_uses[name]
            self.operator_scores[name] = 0
            self.operator_uses[name] = 0

    def _random_ruin(self, solution, uncovered_flights):
        """破坏算子1: 随机移除1%的FDP"""
        all_fdps_flat = [(cid, fdp_idx) for cid, fdps in solution.items() for fdp_idx in range(len(fdps))]
        if not all_fdps_flat:
            return solution, uncovered_flights, set()

        num_to_ruin = max(1, int(len(all_fdps_flat) * 0.01))
        fdps_to_ruin_info = random.sample(all_fdps_flat, min(num_to_ruin, len(all_fdps_flat)))

        return self._apply_ruin(solution, uncovered_flights, fdps_to_ruin_info)

    def _worst_fdp_ruin(self, solution, uncovered_flights):
        """破坏算子2: 移除评价最差的FDP"""
        fdp_scores = []
        for crew_id, fdps in solution.items():
            for fdp_idx, fdp in enumerate(fdps):
                idle_time = 0
                for i in range(len(fdp) - 1):
                    idle_time += (self.segment_details[fdp[i+1][0]]['std'] - self.segment_details[fdp[i][0]]['sta']).total_seconds()
                fdp_scores.append({'score': idle_time, 'info': (crew_id, fdp_idx)})

        if not fdp_scores:
            return solution, uncovered_flights, set()

        fdp_scores.sort(key=lambda x: x['score'], reverse=True)
        num_to_ruin = max(1, int(len(fdp_scores) * 0.01))
        fdps_to_ruin_info = [item['info'] for item in fdp_scores[:num_to_ruin]]

        return self._apply_ruin(solution, uncovered_flights, fdps_to_ruin_info)

    def _related_ruin(self, solution, uncovered_flights):
        """破坏算子3: 移除时间或空间上相关的FDP"""
        all_flights_in_solution = [
            (seg_id, crew_id, fdp_idx)
            for crew_id, fdps in solution.items()
            for fdp_idx, fdp in enumerate(fdps)
            for seg_id, is_ddh in fdp if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight'
        ]
        if not all_flights_in_solution:
            return solution, uncovered_flights, set()

        seed_flight_id, _, _ = random.choice(all_flights_in_solution)
        seed_seg = self.segment_details[seed_flight_id]

        fdps_to_ruin_info = set()

        for flight_id, crew_id, fdp_idx in all_flights_in_solution:
            if flight_id == seed_flight_id:
                fdps_to_ruin_info.add((crew_id, fdp_idx))
                continue

            current_seg = self.segment_details[flight_id]
            time_diff = abs((current_seg['std'] - seed_seg['std']).total_seconds())
            if time_diff < 6 * 3600:
                fdps_to_ruin_info.add((crew_id, fdp_idx))

        num_to_ruin = max(1, int(len(all_flights_in_solution) * 0.01))
        fdps_to_ruin_info = list(fdps_to_ruin_info)
        if len(fdps_to_ruin_info) > num_to_ruin:
            fdps_to_ruin_info = random.sample(fdps_to_ruin_info, num_to_ruin)

        return self._apply_ruin(solution, uncovered_flights, fdps_to_ruin_info)

    def _uncovered_flight_ruin(self, solution, uncovered_flights):
        """破坏算子4: "攻坚"算子"""
        if not uncovered_flights:
            return solution, uncovered_flights, set()

        target_flight_id = random.choice(list(uncovered_flights))
        target_seg = self.segment_details[target_flight_id]

        if not pd.notna(target_seg['std']):
            return solution, uncovered_flights, set()

        best_candidate = {'crew_id': None, 'time_diff': datetime.timedelta.max}
        qualified_crews = self.leg_to_crews.get(target_flight_id, [])

        for crew_id in qualified_crews:
            last_end_time = OPTIMIZATION_START_DATE - datetime.timedelta(hours=MIN_REST_TIME_HOURS)
            for fdp in solution.get(crew_id, []):
                fdp_end_time = self.segment_details[fdp[-1][0]]['sta']
                if pd.notna(fdp_end_time) and fdp_end_time < target_seg['std']:
                    last_end_time = max(last_end_time, fdp_end_time)

            for gd_id in self.fixed_ground_duties.get(crew_id, []):
                gd_end_time = self.segment_details[gd_id]['sta']
                if pd.notna(gd_end_time) and gd_end_time < target_seg['std']:
                    last_end_time = max(last_end_time, gd_end_time)

            time_diff = target_seg['std'] - (last_end_time + datetime.timedelta(hours=MIN_REST_TIME_HOURS))
            if time_diff >= datetime.timedelta(0) and time_diff < best_candidate['time_diff']:
                best_candidate['crew_id'] = crew_id
                best_candidate['time_diff'] = time_diff

        if not best_candidate['crew_id']:
            return solution, uncovered_flights, set()

        crew_to_disrupt = best_candidate['crew_id']

        closest_fdp = {'idx': -1, 'time_diff': float('inf')}
        for fdp_idx, fdp in enumerate(solution.get(crew_to_disrupt, [])):
            fdp_start_time = self.segment_details[fdp[0][0]]['std']
            if pd.notna(fdp_start_time):
                time_diff = abs((fdp_start_time - target_seg['std']).total_seconds())
                if time_diff < closest_fdp['time_diff']:
                    closest_fdp['idx'] = fdp_idx
                    closest_fdp['time_diff'] = time_diff

        if closest_fdp['idx'] != -1:
            fdp_to_ruin_info = [(crew_to_disrupt, closest_fdp['idx'])]
            return self._apply_ruin(solution, uncovered_flights, fdp_to_ruin_info)

        return solution, uncovered_flights, set()


    def _apply_ruin(self, solution, uncovered_flights, fdps_to_ruin_info):
        """通用破坏执行逻辑"""
        temp_solution = copy.deepcopy(solution)
        temp_uncovered = uncovered_flights.copy()

        flights_to_reassign = set()
        fdps_to_ruin_info.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for crew_id, fdp_idx in fdps_to_ruin_info:
            if crew_id in temp_solution and fdp_idx < len(temp_solution[crew_id]):
                fdp = temp_solution[crew_id].pop(fdp_idx)
                for seg_id, is_ddh in fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        flights_to_reassign.add(seg_id)

        temp_uncovered.update(flights_to_reassign)
        return temp_solution, temp_uncovered, flights_to_reassign

    def _recreate(self, solution, uncovered_flights, flights_to_reassign):
        """修复算子: 使用贪心策略重建解"""
        temp_solution = solution
        temp_uncovered = uncovered_flights

        if not flights_to_reassign:
            return temp_solution, temp_uncovered

        crew_latest_status = {}
        for cid in self.crew_df['crewId']:
            crew_latest_status[cid] = {
                'end': OPTIMIZATION_START_DATE - datetime.timedelta(hours=MIN_REST_TIME_HOURS),
                'location': self.crew_stay_stations.get(cid)
            }
            all_duties_for_crew = []
            for ground_duty_id in self.fixed_ground_duties.get(cid, []):
                gd_seg = self.segment_details[ground_duty_id]
                if pd.notna(gd_seg['sta']):
                    all_duties_for_crew.append({'end': gd_seg['sta'], 'location': gd_seg['arriAirport']})
            for fdp in temp_solution.get(cid, []):
                if fdp and pd.notna(self.segment_details[fdp[-1][0]]['sta']):
                    all_duties_for_crew.append({'end': self.segment_details[fdp[-1][0]]['sta'], 'location': self.segment_details[fdp[-1][0]]['arriAirport']})

            if all_duties_for_crew:
                last_real_duty = max(all_duties_for_crew, key=lambda x: x['end'])
                if last_real_duty['end'] > crew_latest_status[cid]['end']:
                    crew_latest_status[cid] = last_real_duty


        sorted_reassign_flights = sorted(list(flights_to_reassign), key=lambda fid: self.segment_details[fid]['std'])

        for flight_id in sorted_reassign_flights:
            if flight_id not in temp_uncovered: continue

            best_assignment = {'crew_id': None, 'cost': float('inf'), 'fdp': None}
            qualified_crews = self.leg_to_crews.get(flight_id, [])

            for crew_id in qualified_crews:
                status = crew_latest_status[crew_id]
                available_time = status['end'] + datetime.timedelta(hours=MIN_REST_TIME_HOURS)
                flight_segment = self.segment_details[flight_id]
                if flight_segment['std'] < available_time: continue

                positioning_tasks = []
                positioning_end_time = status['end']

                if flight_segment['depaAirport'] != status['location']:
                    pos_task_id = self._get_best_positioning_task(status['location'], flight_segment['depaAirport'], status['end'], flight_segment['std'])
                    if pos_task_id:
                        positioning_tasks.append((pos_task_id, 1))
                        positioning_end_time = self.segment_details[pos_task_id]['sta']
                    else:
                        continue

                cost = (flight_segment['std'] - positioning_end_time).total_seconds()

                if cost < best_assignment['cost']:
                    work_fdp = self._build_fdp_from_seed(flight_id, crew_id, temp_uncovered.copy())
                    if work_fdp:
                        full_fdp = positioning_tasks + work_fdp
                        if not self._is_schedule_valid(crew_id, temp_solution.get(crew_id, []) + [full_fdp]):
                            best_assignment.update({'crew_id': crew_id, 'cost': cost, 'fdp': full_fdp})

            if best_assignment['crew_id']:
                assigned_crew, assigned_fdp = best_assignment['crew_id'], best_assignment['fdp']
                if assigned_crew not in temp_solution:
                    temp_solution[assigned_crew] = []
                temp_solution[assigned_crew].append(assigned_fdp)
                crew_latest_status[assigned_crew].update({'end': self.segment_details[assigned_fdp[-1][0]]['sta'], 'location': self.segment_details[assigned_fdp[-1][0]]['arriAirport']})
                for seg_id, is_ddh in assigned_fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        uncovered_flights.discard(seg_id)

        return temp_solution, temp_uncovered

    def _get_coverage_rate(self, uncovered_flights):
        """计算当前覆盖率"""
        if len(self.flight_df) == 0:
            return 1.0
        return (len(self.flight_df) - len(uncovered_flights)) / len(self.flight_df)

    def _calculate_total_flight_time(self, solution):
        """计算当前方案的总飞行时间（分钟）"""
        total_flight_time = 0
        for crew_id, fdps in solution.items():
            for fdp in fdps:
                for seg_id, is_ddh in fdp:
                    segment = self.segment_details[seg_id]
                    if segment['type'] == 'flight' and is_ddh == 0:
                        if (pd.notna(segment['std']) and pd.notna(segment['sta']) and
                            segment['std'] < OPTIMIZATION_END_DATE and
                            segment['sta'] > OPTIMIZATION_START_DATE):
                            total_flight_time += segment['flyTime']
        return total_flight_time

    def _redistribute_tasks_between_crews(self, solution, uncovered_flights, crews):
        """在多个机组之间重新分配任务：完全清空→合并任务池→重新分配"""
        selected_crews = crews

        temp_solution = copy.deepcopy(solution)

        all_crew_flights = set()
        for crew_id in selected_crews:
            if crew_id in temp_solution:
                for fdp in temp_solution[crew_id]:
                    for seg_id, is_ddh in fdp:
                        if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                            all_crew_flights.add(seg_id)
                temp_solution[crew_id] = []

        all_flights_to_assign = uncovered_flights.union(all_crew_flights)

        qualified_flights_union = set()
        for crew_id in selected_crews:
            crew_qualified = set(self.crew_to_legs.get(crew_id, set()))
            qualified_flights_union.update(crew_qualified)

        assignable_flights = all_flights_to_assign & qualified_flights_union

        if not assignable_flights:
            return None, None

        temp_uncovered = self._greedy_assign_flights(temp_solution, assignable_flights, selected_crews)

        final_uncovered = temp_uncovered.union(all_flights_to_assign - assignable_flights)

        for crew_id in selected_crews:
            violations = self._is_schedule_valid(crew_id, temp_solution.get(crew_id, []))
            if violations:
                return None, None

        return temp_solution, final_uncovered

    def _greedy_assign_flights(self, solution, flights_to_assign, target_crews):
        """贪心算法为指定机组分配航班"""
        uncovered = set(flights_to_assign)

        crew_status = {}
        for crew_id in target_crews:
            crew_status[crew_id] = {
                'end': OPTIMIZATION_START_DATE - datetime.timedelta(hours=MIN_REST_TIME_HOURS),
                'location': self.crew_stay_stations.get(crew_id)
            }
            duties = self.fixed_ground_duties.get(crew_id, [])
            if duties:
                valid_duties = [d for d in duties if pd.notna(self.segment_details[d]['sta'])]
                if valid_duties:
                    last_duty_id = max(valid_duties, key=lambda did: self.segment_details[did]['sta'])
                    last_duty_seg = self.segment_details[last_duty_id]
                    if last_duty_seg['sta'] > crew_status[crew_id]['end']:
                        crew_status[crew_id] = {
                            'end': last_duty_seg['sta'],
                            'location': last_duty_seg['arriAirport']
                        }

        sorted_flights = sorted(list(uncovered), key=lambda fid: self.segment_details[fid]['std'])

        for flight_id in sorted_flights:
            if flight_id not in uncovered:
                continue

            best_assignment = {'crew_id': None, 'cost': float('inf'), 'fdp': None}

            for crew_id in target_crews:
                if flight_id not in self.crew_to_legs.get(crew_id, set()):
                    continue

                status = crew_status[crew_id]
                available_time = status['end'] + datetime.timedelta(hours=MIN_REST_TIME_HOURS)
                flight_segment = self.segment_details[flight_id]

                if flight_segment['std'] < available_time:
                    continue

                positioning_tasks = []
                positioning_end_time = status['end']

                if flight_segment['depaAirport'] != status['location']:
                    pos_task_id = self._get_best_positioning_task(
                        status['location'], flight_segment['depaAirport'],
                        available_time, flight_segment['std']
                    )
                    if pos_task_id:
                        positioning_tasks.append((pos_task_id, 1))
                        positioning_end_time = self.segment_details[pos_task_id]['sta']
                    else:
                        continue

                cost = safe_time_diff_seconds(positioning_end_time, flight_segment['std'])
                if cost == float('inf'):
                    continue

                if cost < best_assignment['cost']:
                    work_fdp = self._build_fdp_from_seed(flight_id, crew_id, uncovered.copy())
                    if work_fdp:
                        full_fdp = positioning_tasks + work_fdp
                        if self._is_fdp_valid(full_fdp):
                            test_fdps = solution.get(crew_id, []) + [full_fdp]
                            test_violations = self._is_schedule_valid(crew_id, test_fdps)
                            if not test_violations:
                                best_assignment.update({
                                    'crew_id': crew_id,
                                    'cost': cost,
                                    'fdp': full_fdp
                                })

            if best_assignment['crew_id']:
                assigned_crew = best_assignment['crew_id']
                assigned_fdp = best_assignment['fdp']

                if assigned_crew not in solution:
                    solution[assigned_crew] = []
                solution[assigned_crew].append(assigned_fdp)

                crew_status[assigned_crew].update({
                    'end': self.segment_details[assigned_fdp[-1][0]]['sta'],
                    'location': self.segment_details[assigned_fdp[-1][0]]['arriAirport']
                })

                for seg_id, is_ddh in assigned_fdp:
                    if is_ddh == 0 and self.segment_details[seg_id]['type'] == 'flight':
                        uncovered.discard(seg_id)

        return uncovered

    # =================================================================================
    # ===== 代码修改区域 START =====
    # =================================================================================

    def _calculate_official_score_and_metrics(self, solution, uncovered_flights):
        """
        根据官方评分公式计算全局分数，并返回所有核心指标。
        """
        # 1. 计算总飞行时间, 总置位次数, 以及每个机组的飞行执勤日并加总
        total_flight_time_minutes = 0
        total_flight_duty_days = 0 
        total_positioning_count = 0

        for crew_id in self.crew_df['crewId']:
            fdps = solution.get(crew_id, [])
            crew_flight_duty_dates = set()
            for fdp in fdps:
                total_positioning_count += sum(1 for _, is_ddh in fdp if is_ddh == 1)
                for seg_id, is_ddh in fdp:
                    segment = self.segment_details[seg_id]
                    if segment['type'] == 'flight' and is_ddh == 0 and pd.notna(segment['std']):
                        total_flight_time_minutes += segment['flyTime']
                        start_date = segment['std'].date()
                        end_date = segment['sta'].date()
                        current_date = start_date
                        while current_date <= end_date:
                            if OPTIMIZATION_START_DATE.date() <= current_date <= OPTIMIZATION_END_DATE.date():
                                crew_flight_duty_dates.add(current_date)
                            current_date += datetime.timedelta(days=1)
            total_flight_duty_days += len(crew_flight_duty_dates)

        # 2. 计算日均飞时和覆盖率
        avg_daily_fly_hours = (total_flight_time_minutes / 60) / total_flight_duty_days if total_flight_duty_days > 0 else 0
        base_score = avg_daily_fly_hours * FINAL_SCORE_BASE_MULTIPLIER

        num_uncovered = len(uncovered_flights)
        num_total_flights = len(self.flight_df)
        coverage_rate = (num_total_flights - num_uncovered) / num_total_flights * 100 if num_total_flights > 0 else 100.0

        # 3. 计算各项扣分
        penalty_uncovered = num_uncovered * FINAL_SCORE_PENALTY_UNCOVERED
        
        # 4. 计算外站过夜天数
        total_foreign_overnight_days = 0
        for crew_id in self.crew_df['crewId']:
            base = self.crew_bases.get(crew_id)
            all_duties = []
            fdps = solution.get(crew_id, [])
            for fdp in fdps:
                if fdp and not any(pd.isna(self.segment_details[seg_id]['std']) for seg_id, _ in fdp):
                    all_duties.append({
                        'start': self.segment_details[fdp[0][0]]['std'],
                        'end': self.segment_details[fdp[-1][0]]['sta'],
                        'end_loc': self.segment_details[fdp[-1][0]]['arriAirport']
                    })
            for ground_duty_id in self.fixed_ground_duties.get(crew_id, []):
                gd_seg = self.segment_details[ground_duty_id]
                if pd.notna(gd_seg['std']) and pd.notna(gd_seg['sta']):
                    all_duties.append({'start': gd_seg['std'], 'end': gd_seg['sta'], 'end_loc': gd_seg['arriAirport']})
            
            if not all_duties:
                if self.crew_stay_stations.get(crew_id) != base:
                    days_in_window = (OPTIMIZATION_END_DATE.date() - OPTIMIZATION_START_DATE.date()).days + 1
                    total_foreign_overnight_days += days_in_window
                continue
            
            all_duties.sort(key=lambda x: x['start'])
            last_loc = self.crew_stay_stations.get(crew_id)
            last_time = OPTIMIZATION_START_DATE
            if last_loc != base:
                days_diff = (all_duties[0]['start'].date() - last_time.date()).days
                total_foreign_overnight_days += max(0, days_diff)

            for i in range(len(all_duties) - 1):
                prev_duty = all_duties[i]
                next_duty = all_duties[i+1]
                if prev_duty['end_loc'] != base:
                    days_diff = (next_duty['start'].date() - prev_duty['end'].date()).days
                    total_foreign_overnight_days += max(0, days_diff)
            
            last_duty_in_plan = all_duties[-1]
            if last_duty_in_plan['end_loc'] != base:
                days_diff = (OPTIMIZATION_END_DATE.date() - last_duty_in_plan['end'].date()).days
                total_foreign_overnight_days += max(0, days_diff)

        penalty_foreign_overnight = total_foreign_overnight_days * FINAL_SCORE_PENALTY_FOREIGN_OVERNIGHT
        penalty_positioning = total_positioning_count * FINAL_SCORE_PENALTY_POSITIONING

        # 5. 计算最终分数并打包返回所有指标
        final_score = base_score - penalty_uncovered - penalty_foreign_overnight - penalty_positioning
        
        return {
            "score": final_score,
            "avg_daily_fly_hours": avg_daily_fly_hours,
            "uncovered_count": num_uncovered,
            "coverage_rate": coverage_rate,
            "foreign_overnight_days": total_foreign_overnight_days,
            "positioning_count": total_positioning_count
        }

    def _multi_stage_task_stealing(self, solution, uncovered_flights):
        """
        【新增】两阶段抢任务优化总指挥
        """
        print("\n[进入两阶段抢任务优化流程]")
        
        # --- 阶段一：8人小组，大范围探索 ---
        solution_stage1, uncovered_stage1 = self._task_stealing_optimization(
            solution, 
            uncovered_flights, 
            group_size=8, 
            iterations=TASK_STEALING_ITERATIONS_STAGE_1,
            stage_name="阶段一 (8人模式)"
        )
        
        # --- 阶段二：4人小组，精细调整 ---
        solution_stage2, uncovered_stage2 = self._task_stealing_optimization(
            solution_stage1, 
            uncovered_stage1, 
            group_size=4, 
            iterations=TASK_STEALING_ITERATIONS_STAGE_2,
            stage_name="阶段二 (4人模式)"
        )
        
        print("[两阶段抢任务优化流程结束]")
        return solution_stage2, uncovered_stage2

    def _task_stealing_optimization(self, solution, uncovered_flights, group_size, iterations, stage_name):
        """
        【改造】通用化的抢任务优化函数，以最大化全局官方得分为目标，并动态显示分项
        """
        print(f"\n[{stage_name}] 开始优化，目标: 最大化全局官方得分...")

        # 在循环开始前，计算初始状态
        best_metrics = self._calculate_official_score_and_metrics(solution, uncovered_flights)
        best_solution = copy.deepcopy(solution)
        best_uncovered = uncovered_flights.copy()

        # 初始化进度条
        pbar_desc = (
            f"最优分: {best_metrics['score']:.2f} | "
            f"日均飞时: {best_metrics['avg_daily_fly_hours']:.2f}h | "
            f"覆盖率: {best_metrics['coverage_rate']:.2f}% | "
            f"外站/置位: {best_metrics['foreign_overnight_days']}/{best_metrics['positioning_count']}"
        )
        pbar = tqdm(range(iterations), desc=pbar_desc, unit="次")

        for iteration in pbar:
            # --- (智能选择小组的逻辑保持不变) ---
            all_crew_ids = set(self.crew_df['crewId'])
            idle_crews = [cid for cid in all_crew_ids if len(solution.get(cid, [])) <= 2]
            busy_crews = [cid for cid in all_crew_ids if len(solution.get(cid, [])) > 2]

            if len(busy_crews) + len(idle_crews) < group_size:
                if iteration == 0:
                    print(f"总机组数少于{group_size}，无法执行此阶段优化。")
                break

            selected_crews = []
            # 随机选择繁忙和空闲机组的比例
            num_busy_to_select = random.randint(1, group_size - 1)
            num_busy_to_select = min(len(busy_crews), num_busy_to_select)
            if num_busy_to_select > 0:
                selected_busy = random.sample(busy_crews, num_busy_to_select)
                selected_crews.extend(selected_busy)

            remaining_slots = group_size - len(selected_crews)
            if remaining_slots > 0 and idle_crews:
                idle_candidates = [c for c in idle_crews if c not in selected_crews]
                num_idle_to_select = min(len(idle_candidates), remaining_slots)
                if num_idle_to_select > 0:
                    selected_idle = random.sample(idle_candidates, num_idle_to_select)
                    selected_crews.extend(selected_idle)

            # 如果数量不足，从其他机组中补充
            if len(selected_crews) < group_size:
                all_candidates = [c for c in all_crew_ids if c not in selected_crews]
                if all_candidates:
                    additional_needed = group_size - len(selected_crews)
                    additional_sample = random.sample(all_candidates, min(len(all_candidates), additional_needed))
                    selected_crews.extend(additional_sample)

            if len(selected_crews) < group_size:
                continue
            # --- 智能选择逻辑结束 ---

            # 尝试重分配
            temp_solution, temp_uncovered = self._redistribute_tasks_between_crews(
                solution, uncovered_flights, selected_crews
            )

            if temp_solution is None:
                continue

            # 计算新方案的全局指标
            new_metrics = self._calculate_official_score_and_metrics(temp_solution, temp_uncovered)

            # 如果新方案的全局分数更高，则接受它
            if new_metrics['score'] > best_metrics['score']:
                # 更新当前解
                solution = temp_solution
                uncovered_flights = temp_uncovered
                
                # 更新最优解
                best_solution = copy.deepcopy(solution)
                best_uncovered = uncovered_flights.copy()
                best_metrics = new_metrics
                
                # 更新进度条以反映新的最优指标
                pbar_desc = (
                    f"最优分: {best_metrics['score']:.2f} | "
                    f"日均飞时: {best_metrics['avg_daily_fly_hours']:.2f}h | "
                    f"覆盖率: {best_metrics['coverage_rate']:.2f}% | "
                    f"外站/置位: {best_metrics['foreign_overnight_days']}/{best_metrics['positioning_count']}"
                )
                pbar.set_description(pbar_desc)
        
        pbar.close()
        final_coverage = self._get_coverage_rate(best_uncovered)
        print(f"{stage_name} 完成！最终最优分: {best_metrics['score']:.2f}, 最终覆盖率: {final_coverage*100:.2f}%")

        return best_solution, best_uncovered
        
    # =================================================================================
    # ===== 代码修改区域 END =====
    # =================================================================================

    def solve(self):
        """【重构】主求解函数，执行“混合初始化与全局优化”流程"""
        initial_start_time = time.time()

        # 阶段一 & 二：构建混合初始解
        initial_solution, uncovered_flights = self._build_hybrid_initial_solution()

        num_total_flights = len(self.flight_df)
        if num_total_flights > 0:
            initial_coverage = (len(self.flight_df) - len(uncovered_flights)) / len(self.flight_df) * 100
            print(f"混合初始解构建完成，耗时: {time.time() - initial_start_time:.2f} 秒, 航班覆盖率: {initial_coverage:.2f}%")

        # 阶段三：全局优化
        print("\n[全局优化阶段] 开始对高质量混合解进行打磨...")

        # 1. 中间改进
        solution, uncovered = self._improve_solution_with_crew_centric_heuristic(initial_solution, uncovered_flights)

        # 2. 调用两阶段抢任务优化
        solution, uncovered = self._multi_stage_task_stealing(solution, uncovered)

        # 3. ALNS 迭代 (*** 此处为核心修改区域 ***)
        current_solution, best_solution = solution, copy.deepcopy(solution)
        current_uncovered, best_uncovered = uncovered, uncovered.copy()
        
        # --- 修改1: 使用官方评分函数初始化ALNS的分数和指标 ---
        best_metrics = self._calculate_official_score_and_metrics(current_solution, current_uncovered)
        current_score = best_metrics["score"]
        best_score = current_score

        print("\n开始通过ALNS进行迭代优化 (目标: 官方总分)...")
        optimization_start_time = time.time()
        temperature, alpha, MAX_ITERATIONS = 1000.0, 0.999, 5000

        # --- 修改2: 初始化进度条，使其与抢任务阶段的格式完全一致 ---
        pbar_desc = (
            f"最优分: {best_metrics['score']:.2f} | "
            f"日均飞时: {best_metrics['avg_daily_fly_hours']:.2f}h | "
            f"覆盖率: {best_metrics['coverage_rate']:.2f}% | "
            f"外站/置位: {best_metrics['foreign_overnight_days']}/{best_metrics['positioning_count']}"
        )
        pbar = tqdm(range(MAX_ITERATIONS), desc=pbar_desc, unit="次")

        for i in pbar:
            ruin_operator_name = self._select_operator()
            ruin_operator = self.ruin_operators[ruin_operator_name]

            temp_solution, temp_uncovered, flights_to_reassign = ruin_operator(current_solution, current_uncovered)
            temp_solution, temp_uncovered = self._recreate(temp_solution, temp_uncovered, flights_to_reassign)

            # --- 修改3: 在迭代中也使用官方评分函数，替换旧的目标函数 ---
            new_metrics = self._calculate_official_score_and_metrics(temp_solution, temp_uncovered)
            new_score = new_metrics["score"]

            accepted = False
            if new_score > current_score or random.random() < math.exp((new_score - current_score) / temperature):
                current_solution, current_score, current_uncovered = temp_solution, new_score, temp_uncovered
                accepted = True

                if new_score > best_score:
                    best_score = new_score
                    best_solution = copy.deepcopy(current_solution)
                    best_uncovered = current_uncovered.copy()
                    best_metrics = new_metrics # 保存最优解对应的完整指标

                    # --- 修改4: 找到更优解时，立即更新进度条为最新、最全的信息 ---
                    pbar_desc = (
                        f"最优分: {best_metrics['score']:.2f} | "
                        f"日均飞时: {best_metrics['avg_daily_fly_hours']:.2f}h | "
                        f"覆盖率: {best_metrics['coverage_rate']:.2f}% | "
                        f"外站/置位: {best_metrics['foreign_overnight_days']}/{best_metrics['positioning_count']}"
                    )
                    pbar.set_description(pbar_desc)

            temperature *= alpha
            
            # 更新算子权重 (这部分逻辑保持不变)
            if new_score > best_score: # Corresponds to new_best_found
                self.operator_scores[ruin_operator_name] += self.alns_score_global_best
            elif accepted and new_score > current_score:
                self.operator_scores[ruin_operator_name] += self.alns_score_better
            elif accepted:
                self.operator_scores[ruin_operator_name] += self.alns_score_accepted
            self.operator_uses[ruin_operator_name] += 1

            if (i + 1) % self.alns_segment_length == 0:
                self._update_weights()

        pbar.close()
        print(f"\n全局优化完成! 优化过程耗时: {time.time() - optimization_start_time:.2f} 秒")
        print(f"最终算子权重: {self.operator_weights}")
        return best_solution, best_uncovered


    def _patch_connectivity_issues(self, solution):
        """【升级】后优化补丁：修复纯地勤人员的地点衔接问题，增加安全验证"""
        print("\n[后优化补丁] 正在检查并修复纯地勤人员的地点衔接问题...")

        all_crew_ids = set(self.crew_df['crewId'])
        crews_with_flights = set(cid for cid, fdps in solution.items() if fdps)

        crews_to_patch = []
        for crew_id in all_crew_ids:
            if crew_id not in crews_with_flights and crew_id in self.fixed_ground_duties and crew_id not in {m['crew_id'] for m in self.proactive_moves}:
                crews_to_patch.append(crew_id)

        if not crews_to_patch:
            print("没有发现需要修复的纯地勤人员。")
            return

        print(f"发现 {len(crews_to_patch)} 名纯地勤人员可能需要修复地点衔接。")

        for crew_id in tqdm(crews_to_patch, desc="修复纯地勤衔接", unit="人"):
            current_schedule = solution.get(crew_id, [])

            ground_duties = self.fixed_ground_duties[crew_id]
            if not ground_duties: continue

            sorted_gd = sorted(ground_duties, key=lambda gid: self.segment_details[gid]['std'])
            first_duty_id = sorted_gd[0]
            first_duty_details = self.segment_details[first_duty_id]

            start_location = self.crew_stay_stations.get(crew_id)
            target_location = first_duty_details['depaAirport']

            if start_location == target_location: continue

            historical_time_window_start = OPTIMIZATION_START_DATE - datetime.timedelta(hours=72)
            time_window_end = first_duty_details['std']

            pos_task_ac = self._get_best_positioning_task(start_location, target_location, historical_time_window_start, time_window_end)
            if pos_task_ac:
                new_fdp = [(pos_task_ac, 1)]
                # 【安全补丁】构建完整的预补丁计划并进行终审
                proposed_full_schedule = current_schedule + [new_fdp]
                if not self._is_schedule_valid(crew_id, proposed_full_schedule):
                    solution[crew_id].append(new_fdp)
                    self.patched_crew_ids.append(crew_id)
                    continue

        print(f"后优化补丁执行完毕。成功为 {len(self.patched_crew_ids)} 名纯地勤人员添加了置位任务。")

    def _calculate_final_score_and_metrics(self, solution, uncovered_flights):
        """【已修正】根据官方评价指标，计算最终方案的详细分数和各项指标"""
        print("\n4. 正在根据官方规则计算最终得分...")

        self.all_violations_details = []

        # 初始化所有全局累加器
        total_flight_time_minutes = 0
        total_flight_duty_days = 0
        total_foreign_overnight_days = 0
        total_positioning_count = 0
        airports_used_for_layover = set()

        # 遍历每个机组，独立计算其指标，然后累加到全局
        for crew_id in self.crew_df['crewId']:
            fdps = solution.get(crew_id, [])
            base = self.crew_bases.get(crew_id)

            # --- 核心修正逻辑：采用与抢任务阶段相同的计算方式 ---
            crew_flight_duty_dates = set()
            for fdp in fdps:
                # 累加置位次数
                total_positioning_count += sum(1 for _, is_ddh in fdp if is_ddh == 1)

                # 累加飞行时间，并记录该机组的执勤日
                for seg_id, is_ddh in fdp:
                    segment = self.segment_details[seg_id]
                    if segment['type'] == 'flight' and is_ddh == 0 and pd.notna(segment['std']):
                        total_flight_time_minutes += segment['flyTime']

                        start_date = segment['std'].date()
                        end_date = segment['sta'].date()
                        current_date = start_date
                        while current_date <= end_date:
                            if OPTIMIZATION_START_DATE.date() <= current_date <= OPTIMIZATION_END_DATE.date():
                                crew_flight_duty_dates.add(current_date)
                            current_date += datetime.timedelta(days=1)
            
            # 将当前机组的执勤天数累加到总数中
            total_flight_duty_days += len(crew_flight_duty_dates)
            # --- 核心修正逻辑结束 ---


            # --- 保留原有的、用于计算其他罚分项的复杂逻辑 ---
            all_duties = []
            for fdp in fdps:
                if not fdp or any(pd.isna(self.segment_details[seg_id]['std']) for seg_id, _ in fdp): continue
                all_duties.append({
                    'start': self.segment_details[fdp[0][0]]['std'],
                    'end': self.segment_details[fdp[-1][0]]['sta'],
                    'start_loc': self.segment_details[fdp[0][0]]['depaAirport'],
                    'end_loc': self.segment_details[fdp[-1][0]]['arriAirport'],
                    'obj': fdp
                })

            for ground_duty_id in self.fixed_ground_duties.get(crew_id, []):
                gd_seg = self.segment_details[ground_duty_id]
                if pd.notna(gd_seg['std']) and pd.notna(gd_seg['sta']):
                    all_duties.append({
                        'start': gd_seg['std'], 'end': gd_seg['sta'],
                        'start_loc': gd_seg['depaAirport'], 'end_loc': gd_seg['arriAirport'],
                        'obj': ground_duty_id, 'is_flight_period_task': False
                    })

            if all_duties:
                all_duties.sort(key=lambda x: x['start'])
                
                # 计算外站过夜天数
                last_location = self.crew_stay_stations.get(crew_id)
                last_time = OPTIMIZATION_START_DATE
                if last_location != base:
                    days_diff = (all_duties[0]['start'].date() - last_time.date()).days
                    total_foreign_overnight_days += max(0, days_diff)

                for i in range(len(all_duties) - 1):
                    prev_duty = all_duties[i]
                    next_duty = all_duties[i+1]
                    if prev_duty['end_loc'] != base:
                        days_diff = (next_duty['start'].date() - prev_duty['end'].date()).days
                        total_foreign_overnight_days += max(0, days_diff)
                
                last_duty_in_plan = all_duties[-1]
                if last_duty_in_plan['end_loc'] != base:
                    days_diff = (OPTIMIZATION_END_DATE.date() - last_duty_in_plan['end'].date()).days
                    total_foreign_overnight_days += max(0, days_diff)

                # 收集所有被用作过夜的机场
                historical_station = self.crew_stay_stations.get(crew_id)
                if all_duties[0]['start_loc'] == historical_station:
                    airports_used_for_layover.add(historical_station)
                for duty in all_duties:
                    airports_used_for_layover.add(duty['start_loc'])
                    airports_used_for_layover.add(duty['end_loc'])

            # 检查当前机组的违规情况
            crew_violations = self._is_schedule_valid(crew_id, fdps)
            if crew_violations:
                self.all_violations_details.extend(crew_violations)

        # --- 所有机组遍历完毕，开始计算最终分数 ---
        total_violations = len(self.all_violations_details)

        # 使用正确累加的指标计算日均飞时
        avg_daily_fly_hours = (total_flight_time_minutes / 60) / total_flight_duty_days if total_flight_duty_days > 0 else 0
        base_score = avg_daily_fly_hours * FINAL_SCORE_BASE_MULTIPLIER

        # 计算各项罚分
        num_uncovered = len(uncovered_flights)
        penalty_uncovered = num_uncovered * FINAL_SCORE_PENALTY_UNCOVERED

        num_total_flights = len(self.flight_df)
        coverage_rate = (num_total_flights - num_uncovered) / num_total_flights * 100 if num_total_flights > 0 else 100.0

        num_new_layover = len(airports_used_for_layover - self.layover_stations)
        penalty_new_layover = num_new_layover * FINAL_SCORE_PENALTY_NEW_LAYOVER

        penalty_foreign_overnight = total_foreign_overnight_days * FINAL_SCORE_PENALTY_FOREIGN_OVERNIGHT
        penalty_positioning = total_positioning_count * FINAL_SCORE_PENALTY_POSITIONING
        penalty_violations = total_violations * FINAL_SCORE_PENALTY_VIOLATION

        # 计算最终总分
        final_score = base_score - penalty_uncovered - penalty_new_layover - penalty_foreign_overnight - penalty_positioning - penalty_violations

        # 准备返回结果
        metrics = {
            "--- 基础得分 ---": base_score,
            "日均飞时 (小时/飞行执勤日)": avg_daily_fly_hours,
            "总飞行时间 (分钟)": total_flight_time_minutes,
            "总飞行执勤日历日": total_flight_duty_days,
            "航班覆盖率 (%)": coverage_rate,
            "--- 扣分项 ---": "--- 金额 ---",
            f"未覆盖航班 ({num_uncovered}个 * {FINAL_SCORE_PENALTY_UNCOVERED}分)": -penalty_uncovered,
            f"新增过夜站点 ({num_new_layover}个 * {FINAL_SCORE_PENALTY_NEW_LAYOVER}分)": -penalty_new_layover,
            f"外站过夜 ({total_foreign_overnight_days}天 * {FINAL_SCORE_PENALTY_FOREIGN_OVERNIGHT}分)": -penalty_foreign_overnight,
            f"置位次数 ({total_positioning_count}次 * {FINAL_SCORE_PENALTY_POSITIONING}分)": -penalty_positioning,
            f"规则违规 ({total_violations}次 * {FINAL_SCORE_PENALTY_VIOLATION}分)": -penalty_violations,
            "--- 最终总分 ---": final_score
        }
        return metrics

    def save_to_excel(self, solution, filename="crew_schedule_optimized.xlsx"):
        """将最终排班方案保存到Excel文件"""
        print(f"\n5. 正在将详细排班结果保存到 {filename}...")

        output_data = []
        all_assigned_in_fdps = set()
        for fdps in solution.values():
            for fdp in fdps:
                for seg_id, _ in fdp:
                    all_assigned_in_fdps.add(seg_id)

        for crew_id in self.crew_df['crewId']:
            task_blocks = []

            fdps = solution.get(crew_id, [])
            for fdp in fdps:
                if fdp:
                    start_time = self.segment_details[fdp[0][0]]['std']
                    task_blocks.append({'start': start_time, 'block': fdp})

            for ground_duty_id in self.fixed_ground_duties.get(crew_id, []):
                if ground_duty_id not in all_assigned_in_fdps:
                    start_time = self.segment_details[ground_duty_id]['std']
                    task_blocks.append({'start': start_time, 'block': [(ground_duty_id, self.segment_details[ground_duty_id]['isDuty'])]})

            if not task_blocks: continue

            task_blocks.sort(key=lambda x: x['start'])

            for item in task_blocks:
                task_list = item['block']
                for seg_id, is_ddh in task_list:
                    segment = self.segment_details[seg_id]
                    output_data.append({
                        'crewId': crew_id, 'taskID': seg_id, 'isDDH': is_ddh,
                        'depa': segment['depaAirport'], 'arri': segment['arriAirport'],
                        'td': segment['std'], 'ta': segment['sta'], 'aircraftNo': segment.get('aircraftNo')
                    })

        if not output_data:
            print("警告: 没有排班数据可供保存。")
            return None

        output_df = pd.DataFrame(output_data)
        if 'td' in output_df.columns:
            output_df['td'] = pd.to_datetime(output_df['td']).dt.strftime('%Y-%m-%d %H:%M')
        if 'ta' in output_df.columns:
            output_df['ta'] = pd.to_datetime(output_df['ta']).dt.strftime('%Y-%m-%d %H:%M')

        output_df.to_excel(filename, index=False, engine='openpyxl')
        print("详细排班保存成功！")
        return output_df

    def save_roster_result_to_csv(self, detailed_schedule_df, filename="rosterResult.csv"):
        """根据详细排班DataFrame，生成rosterResult.csv"""
        if detailed_schedule_df is None:
            print(f"警告: 由于没有详细排班数据，无法生成 {filename}。")
            return

        print(f"\n正在生成 rosterResult 并保存到 {filename}...")

        # 1. 从完整的排班DataFrame中选择需要的列
        #    注意：原始列名为 'taskID'
        roster_df = detailed_schedule_df[['crewId', 'taskID', 'isDDH']].copy()

        # 2. 按照图片要求，将 'taskID' 列重命名为 'taskId'
        roster_df.rename(columns={'taskID': 'taskId'}, inplace=True)

        # 3. 将结果保存为CSV文件，不包含索引列
        roster_df.to_csv(filename, index=False)
        print(f"{filename} 保存成功！")

    def save_summary_to_excel(self, solution, filename="crew_summary_optimized.xlsx"):
        """为每个机组生成每日摘要并保存到Excel文件"""
        print(f"\n6. 正在生成机组每日摘要并保存到 {filename}...")
        summary_data = []

        all_crew_tasks = defaultdict(list)
        all_assigned_in_fdps = set()
        for crew_id, fdps in solution.items():
            for fdp in fdps:
                all_crew_tasks[crew_id].extend(fdp)
                for seg_id, _ in fdp:
                    all_assigned_in_fdps.add(seg_id)

        for crew_id, ground_duties in self.fixed_ground_duties.items():
            for ground_duty_id in ground_duties:
                if ground_duty_id not in all_assigned_in_fdps:
                    all_crew_tasks[crew_id].append((ground_duty_id, self.segment_details[ground_duty_id]['isDuty']))

        for crew_id in self.crew_df['crewId']:
            crew_base = self.crew_bases.get(crew_id)
            last_location = self.crew_stay_stations.get(crew_id)

            tasks = sorted(all_crew_tasks.get(crew_id, []), key=lambda x: self.segment_details[x[0]]['std'])
            task_idx = 0

            for day_offset in range((OPTIMIZATION_END_DATE.date() - OPTIMIZATION_START_DATE.date()).days + 1):
                current_day = OPTIMIZATION_START_DATE.date() + datetime.timedelta(days=day_offset)

                daily_tasks = []
                while task_idx < len(tasks) and self.segment_details[tasks[task_idx][0]]['std'].date() == current_day:
                    daily_tasks.append(tasks[task_idx])
                    task_idx += 1

                if not daily_tasks:
                    summary_data.append({
                        '机组ID': crew_id,
                        '日期': current_day.strftime('%Y-%m-%d'),
                        '状态': '休息',
                        '执勤日飞行时间(分钟)': 0,
                        '执勤日执勤时间(分钟)': 0,
                        '当日出发地': None,
                        '当日到达地': None,
                        '过夜地点': last_location,
                        '是否为可过夜机场': last_location in self.layover_stations,
                        '最后到达机场是否为基地': last_location == crew_base,
                        '首任务开始时间': None,
                        '末航班结束时间': None,
                    })
                    continue

                first_task_start_time = self.segment_details[daily_tasks[0][0]]['std']

                dep_location = self.segment_details[daily_tasks[0][0]]['depaAirport']
                arr_location = self.segment_details[daily_tasks[-1][0]]['arriAirport']

                daily_flight_time = 0
                last_flight_end_time = None

                flight_tasks_today = []
                for seg_id, is_ddh in daily_tasks:
                    segment = self.segment_details[seg_id]
                    if segment['type'] == 'flight' and is_ddh == 0:
                        daily_flight_time += segment['flyTime']
                        flight_tasks_today.append(segment)

                duty_duration_minutes = 0
                if flight_tasks_today:
                    last_flight_end_time = max(f['sta'] for f in flight_tasks_today)
                    duty_duration = last_flight_end_time - first_task_start_time
                    duty_duration_minutes = duty_duration.total_seconds() / 60

                summary_data.append({
                    '机组ID': crew_id,
                    '日期': current_day.strftime('%Y-%m-%d'),
                    '状态': '执勤',
                    '执勤日飞行时间(分钟)': daily_flight_time,
                    '执勤日执勤时间(分钟)': round(duty_duration_minutes, 2) if duty_duration_minutes > 0 else 0,
                    '当日出发地': dep_location,
                    '当日到达地': arr_location,
                    '过夜地点': arr_location,
                    '是否为可过夜机场': arr_location in self.layover_stations,
                    '最后到达机场是否为基地': arr_location == crew_base,
                    '首任务开始时间': first_task_start_time.strftime('%H:%M'),
                    '末航班结束时间': last_flight_end_time.strftime('%H:%M') if last_flight_end_time else None,
                })
                last_location = arr_location

        if not summary_data:
            print("警告: 没有排班数据可供保存。")
            return

        summary_df = pd.DataFrame(summary_data)
        column_order = [
            '机组ID', '日期', '状态', '执勤日飞行时间(分钟)', '执勤日执勤时间(分钟)',
            '当日出发地', '当日到达地', '过夜地点', '是否为可过夜机场',
            '最后到达机场是否为基地', '首任务开始时间', '末航班结束时间'
        ]
        summary_df = summary_df[column_order]

        summary_df.to_excel(filename, index=False, engine='openpyxl')
        print("机组每日摘要保存成功！")

    def save_violations_to_excel(self, filename="violations_summary.xlsx"):
        """将所有违规详情保存到Excel文件"""
        print(f"\n7. 正在生成违规详情报告并保存到 {filename}...")
        if not self.all_violations_details:
            print("无违规记录，不生成报告文件。")
            return

        violations_df = pd.DataFrame(self.all_violations_details)
        violations_df = violations_df[['crewId', 'type', 'details']]
        violations_df.rename(columns={
            'crewId': '机组ID',
            'type': '违规类型',
            'details': '详细信息'
        }, inplace=True)

        violations_df.to_excel(filename, index=False, engine='openpyxl')
        print("违规详情报告保存成功！")


def main():
    try:
        print("开始加载数据文件...")
        # ----------------------------------------------------------------
        # !!! 注意：请将下面的文件路径修改为您自己的实际路径 !!!
        # ----------------------------------------------------------------
        # 示例: base_path = "C:/Your/Path/To/Data/"
        # 请根据您的文件存放位置修改下面的路径
        # ----------------------------------------------------------------
        scheduler = CrewScheduler(
            pd.read_csv('crewLegMatch.csv'),
            pd.read_csv('flight.csv'),
            pd.read_csv('busInfo.csv'),
            pd.read_csv('crew.csv'),
            pd.read_csv('layoverStation.csv'),
            pd.read_csv('groundDuty.csv')
        )

        final_solution, uncovered = scheduler.solve()

        if final_solution is None:
            print("未能找到有效的解决方案。"); return

        scheduler._patch_connectivity_issues(final_solution)

        if scheduler.patched_crew_ids:
            print("\n" + "-"*20 + " 后优化修复详情 " + "-"*20)
            print(f"成功为以下 {len(scheduler.patched_crew_ids)} 名机组修复了地点不衔接问题:")
            for i in range(0, len(scheduler.patched_crew_ids), 10):
                print(", ".join(map(str, scheduler.patched_crew_ids[i:i+10])))
            print("-" * (44 + len(" 后优化修复详情 ")))

        print("\n" + "="*50 + "\n最终优化结果详细评分\n" + "="*50)
        final_metrics = scheduler._calculate_final_score_and_metrics(final_solution, uncovered)
        for key, value in final_metrics.items():
            if isinstance(value, str):
                print(f"{key}: {value}")
            else:
                print(f"{key:<40}: {value:>15.2f}")

        scheduler.save_violations_to_excel("violations_summary.xlsx")
        
        # 调用save_to_excel并接收返回的DataFrame
        detailed_df = scheduler.save_to_excel(final_solution)
        
        # 使用返回的DataFrame生成rosterResult.csv
        scheduler.save_roster_result_to_csv(detailed_df)
        
        scheduler.save_summary_to_excel(final_solution, "crew_summary_optimized.xlsx")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}。请确保所有CSV文件都在正确的位置，并已修改main函数中的文件路径。")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
