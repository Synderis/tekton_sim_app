from datetime import datetime

selection_time = datetime.now()
import math
import statistics
import random
import tkinter as tk
import inflect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import sys
import os



result_array = ["hit", "miss"]
attack_level = 118
strength_level = 118
piety_attack = 1.2
piety_strength = 1.23
effective_strength_lvl = int(strength_level * piety_strength + 3 + 8)
effective_attack_lvl = int(attack_level * piety_attack + 0 + 8)
effective_spec_attack_lvl = int(attack_level * piety_attack + 3 + 8)
effective_spec_strength_lvl = int(strength_level * piety_strength + 0 + 8)
hammer_count_list = []
anvil_count_list = []
times = []
tick_times_list = []
tick_times_one_anvil = []
hp_check_list = []

crush = 'crush'
slash = 'slash'
stab = 'stab'

param_container = {}


def map_parameters(**parameters):
    if parameters:
        param_container.update(parameters)
        return run_script()


def run_script():
    class NewParams:
        def __init__(self, ring, cm, inq, feros, tort, fang, five_tick_only, preveng, veng_camp, vuln, book_of_water, short_lure):
            self.ring = ring
            self.cm = cm
            self.inq = inq
            self.feros = feros
            self.tort = tort
            self.fang = fang
            self.five_tick_only = five_tick_only
            self.preveng = preveng
            self.veng_camp = veng_camp
            self.vuln = vuln
            self.book_of_water = book_of_water
            self.short_lure = short_lure


    # These are the metrics that we will be measuring EACH trial and they will reset to initial conditions at the start
    class Offensive:
        def __init__(self, four_tick_hit_counter, five_tick_hit_counter, time_parameter, phase, idle_time, fang_spec_status,
                     specced_last_anvil, hammer_missed_count, hammer_hit_count, hp_pool):
            self.four_tick_hit_counter = four_tick_hit_counter
            self.five_tick_hit_counter = five_tick_hit_counter
            self.time_parameter = time_parameter
            self.phase = phase
            self.idle_time = idle_time
            self.fang_spec_status = fang_spec_status
            self.specced_last_anvil = specced_last_anvil
            self.hammer_missed_count = hammer_missed_count
            self.hammer_hit_count = hammer_hit_count
            self.hp_pool = hp_pool

        def reset(self):
            self.four_tick_hit_counter = 0
            self.five_tick_hit_counter = 0
            self.time_parameter = 0.0
            self.phase = 0
            self.idle_time = 0
            self.fang_spec_status = True
            self.specced_last_anvil = False
            self.hammer_missed_count = 0
            self.hammer_hit_count = 0
            self.hp_pool = 121


    # This is the block where gear is stored this will stay static throughout each trial.
    class Gear:
        def __init__(self, dwh_att_bonus, dwh_str_bonus, four_tick_att_bonus, four_tick_str_bonus, fang_att_bonus,
                     fang_str_bonus, scy_att_bonus, scy_str_bonus, gear_multiplier, static_crush_weapon, five_tick_weapon):
            self.dwh_att_bonus = dwh_att_bonus
            self.dwh_str_bonus = dwh_str_bonus
            self.four_tick_att_bonus = four_tick_att_bonus
            self.four_tick_str_bonus = four_tick_str_bonus
            self.fang_att_bonus = fang_att_bonus
            self.fang_str_bonus = fang_str_bonus
            self.scy_att_bonus = scy_att_bonus
            self.scy_str_bonus = scy_str_bonus
            self.gear_multiplier = gear_multiplier
            self.static_crush_weapon = static_crush_weapon
            self.five_tick_weapon = five_tick_weapon

    def create_objects():
        trial_parameters_pre = NewParams(**param_container)
        return trial_parameters_pre

    try:
        trial_parameters = create_objects()
        print(trial_parameters)
    except:
        trial_parameters = None
        print(trial_parameters)

    if param_container:
        if trial_parameters.fang:
            scythe = False
        else:
            scythe = True

        if trial_parameters.five_tick_only:
            four_and_five = False
        else:
            four_and_five = True

    # Function that will modify the gear setup based on selections from the gui prompt
    # Now that im looking at it again gear selection and loadout adjuster are probably redundant functions
    def gear_selection():
        attack_gear = 0
        strength_gear = 0
        ring_stats = {'Select Ring': (0, 0), 'b_ring': (0, 8), 'brim': (4, 4), 'ultor_ring': (0, 12), 'lightbearer': (0, 0)}
        attack_gear += ring_stats[trial_parameters.ring][0]
        strength_gear += ring_stats[trial_parameters.ring][1]
        if trial_parameters.tort:
            attack_gear += 5
            strength_gear += 2
        if trial_parameters.feros:
            attack_gear += 4
            strength_gear += 2
        if trial_parameters.fang:
            five_tick_weapon = stab
        else:
            if trial_parameters.inq:
                five_tick_weapon = crush
            else:
                five_tick_weapon = slash
        loadout_adjuster(attack_gear, strength_gear, five_tick_weapon)
        return loadout

    def loadout_adjuster(att_modifier, str_modifier, five_tick_style):
        loadout_list = [loadout.dwh_str_bonus, loadout.four_tick_str_bonus, loadout.fang_str_bonus, loadout.scy_str_bonus,
                        loadout.dwh_att_bonus, loadout.four_tick_att_bonus, loadout.fang_att_bonus, loadout.scy_att_bonus]
        loadout_list[:3] = [i + str_modifier for i in loadout_list[:3]]
        loadout_list[4:] = [i + att_modifier for i in loadout_list[4:]]
        (loadout.dwh_str_bonus, loadout.four_tick_str_bonus, loadout.fang_str_bonus, loadout.scy_str_bonus,
         loadout.dwh_att_bonus, loadout.four_tick_att_bonus, loadout.fang_att_bonus, loadout.scy_att_bonus) = loadout_list
        loadout.five_tick_weapon = five_tick_style
        return


    # These are the default loadouts that will be selected based on gui selection
    if param_container:
        if trial_parameters.inq:
            loadout = Gear(dwh_att_bonus=183, dwh_str_bonus=136, four_tick_att_bonus=183, four_tick_str_bonus=140,
                           fang_att_bonus=155, fang_str_bonus=154, scy_att_bonus=90, scy_str_bonus=118, gear_multiplier=1.025,
                           static_crush_weapon=crush, five_tick_weapon='')
            gear_selection()
            print('loadout bonuses selected: ', loadout.dwh_att_bonus, loadout.dwh_str_bonus, loadout.four_tick_att_bonus,
                  loadout.four_tick_str_bonus, loadout.fang_att_bonus, loadout.fang_str_bonus, loadout.scy_att_bonus,
                  loadout.scy_str_bonus, loadout.gear_multiplier)
            print('----------')
        else:
            loadout = Gear(dwh_att_bonus=151, dwh_str_bonus=144, four_tick_att_bonus=151, four_tick_str_bonus=148,
                           fang_att_bonus=163, fang_str_bonus=162, scy_att_bonus=138, scy_str_bonus=126, gear_multiplier=1,
                           static_crush_weapon=crush, five_tick_weapon='')
            gear_selection()
            print('loadout bonuses selected: ', loadout.dwh_att_bonus, loadout.dwh_str_bonus, loadout.four_tick_att_bonus,
                  loadout.four_tick_str_bonus, loadout.fang_att_bonus, loadout.fang_str_bonus, loadout.scy_att_bonus,
                  loadout.scy_str_bonus, loadout.gear_multiplier)
            print('----------')


    # These are the stats of the npc itself and its active statuses
    class NPC:
        def __init__(self, hp, defence, stab_def, slash_def, crush_def, veng_count):
            self.hp = hp
            self.defence = defence
            self.stab_def = stab_def
            self.slash_def = slash_def
            self.crush_def = crush_def
            self.veng_count = veng_count

        def lower_hp(self, damage):
            self.hp -= damage
            if self.hp < 0:
                self.hp = 0

        def lower_def(self, amount):
            self.defence -= amount
            if self.defence < 0:
                self.defence = 0

        def reset(self):
            if trial_parameters.cm:
                self.hp = 450
                self.defence = 246
                self.stab_def = 155
                self.slash_def = 165
                self.crush_def = 105
                self.veng_count = 0
            else:
                self.hp = 300
                self.defence = 205
                self.stab_def = 155
                self.slash_def = 165
                self.crush_def = 105
                self.veng_count = 0



    def create_trial_objects():
        hit_metrics_pre = Offensive(0, 0, time_parameter=0.0, phase=0, idle_time=0, fang_spec_status=True,
                                specced_last_anvil=False, hammer_missed_count=0, hammer_hit_count=0, hp_pool=121)
        if trial_parameters.cm:
            tekton_pre = NPC(450, 246, 155, 165, 105, veng_count=0)
            base_hp_pre, base_def_pre = [450, 246]
        else:
            tekton_pre = NPC(300, 205, 155, 165, 105, veng_count=0)
            base_hp_pre, base_def_pre = [300, 205]
        return hit_metrics_pre, tekton_pre, base_hp_pre, base_def_pre

    if param_container:
        hit_metrics, tekton, base_hp, base_def = create_trial_objects()
    else:
        hit_metrics, tekton, base_hp, base_def = [None] * 4


    # This damage value selection based on adjusted gear and whether the hit chance function determines a hit
    def hit_value_roll(spec_bonus, four_tick, five_tick, max_hit_modifier=1.0):
        def strength_selector():
            if spec_bonus:
                return loadout.dwh_str_bonus
            elif four_tick:
                return loadout.four_tick_str_bonus
            elif five_tick:
                if trial_parameters.fang:
                    return loadout.fang_str_bonus
                elif scythe:
                    return loadout.scy_str_bonus
            else:
                raise ValueError("Error in strength selector function")


        damage_selection = 0
        if spec_bonus:
            max_hit = int(
                int(0.5 + effective_spec_strength_lvl * ((strength_selector() + 64) / 640)) * 1.5 * loadout.gear_multiplier)
            return int(random.randint(0, max_hit))
        else:
            max_hit = int(int(0.5 + effective_strength_lvl * ((strength_selector() + 64) / 640)) * loadout.gear_multiplier)
            if four_tick:
                return int(random.randint(0, max_hit))
            elif five_tick:
                if trial_parameters.fang:
                    max_hit = int(((0.5 + effective_strength_lvl * ((strength_selector() + 64) / 640)) * max_hit_modifier))
                    min_hit = int((0.5 + effective_strength_lvl * ((strength_selector() + 64) / 640)) * .15)
                    return int(random.randint(min_hit, max_hit))
                else:
                    return int(random.randint(0, int(max_hit * max_hit_modifier)))
        raise ValueError("Error in hit_value_roll function")


    # I forget exactly why I added these but I believe there was some weird rounding error that math.ceiling wasnt fixing
    def is_whole(whole):
        return whole % 1 == 0


    def adjust_def_integer():
        if not is_whole(tekton.defence):
            tekton.defence = int(tekton.defence) + 1


    # Attack roll function that determines the roll that will be used in hit chance based on gear loadout and NPC stats
    def attack_roll(spec_attack, four_tick, five_tick, multiplier):
        def attack_selector():
            if spec_attack:
                return loadout.dwh_att_bonus
            elif four_tick:
                return loadout.four_tick_att_bonus
            elif five_tick:
                if trial_parameters.fang:
                    return loadout.fang_att_bonus
                elif scythe:
                    return loadout.scy_att_bonus
            else:
                raise ValueError("Error in attack selector function")

        if spec_attack:
            max_attack_roll_basic = int(effective_spec_attack_lvl * (attack_selector() + 64))
        else:
            max_attack_roll_basic = int(effective_attack_lvl * (attack_selector() + 64))
        if trial_parameters.ring == 'lightbearer':
            if trial_parameters.fang:
                max_attack_roll = int(max_attack_roll_basic * multiplier)
            else:
                max_attack_roll = int(max_attack_roll_basic * loadout.gear_multiplier)
        else:
            max_attack_roll = int(max_attack_roll_basic * loadout.gear_multiplier)
        first_roll = random.randint(0, max_attack_roll)
        return first_roll


    # Function that will take the attack roll and defense roll and determine hit or miss of main damage phase
    def hit_chancer(spec, four_tick, five_tick, fang_spec_hit, status):
        attack_roll_check = 0
        def_roll_check = defence_roll(spec, four_tick, five_tick, status)
        if spec:
            attack_roll_check = attack_roll(True, False, False, multiplier=1.0)
        elif four_tick:
            attack_roll_check = attack_roll(False, True, False, multiplier=1.0)
        elif five_tick:
            if trial_parameters.fang:
                if fang_spec_hit:
                    attack_roll_check = attack_roll(False, False, True, multiplier=1.5)
                    attack_roll_check2 = attack_roll(False, False, True, multiplier=1.5)
                else:
                    attack_roll_check = attack_roll(False, False, True, multiplier=1.0)
                    attack_roll_check2 = attack_roll(False, False, True, multiplier=1.0)
                roll_list = [attack_roll_check, attack_roll_check2]
                return True if any(i > def_roll_check for i in roll_list) else False
            else:
                attack_roll_check = attack_roll(False, False, True, multiplier=1.0)
        return True if attack_roll_check > def_roll_check else False


    # Function that determines whether vulnerability hit which lowers initial tekton defense before any other def. reduction
    # I need to add something that lets this be more adjustable based on varying gear
    def vuln_applicator():
        if trial_parameters.vuln:
            vuln = np.random.choice(result_array, 1, replace=True, p=[.62, (1 - .62)])
            if trial_parameters.book_of_water:
                book_of_water = .15
            else:
                book_of_water = .10
            if vuln:
                tekton.lower_def(int(tekton.defence * book_of_water))
                adjust_def_integer()
            else:
                tekton.lower_def(0)
            return
        else:
            return


    def hammer_check():
        if hit_metrics.hammer_hit_count in [0, 1, 2]:
            hammer_count_list.append(hit_metrics.hammer_hit_count)
        return


    def hammer_missed():
        tekton.lower_def(int((tekton.defence * .05)))
        adjust_def_integer()
        hit_metrics.hammer_missed_count += 1
        return


    # Function that will take the attack roll and defense roll and determine hit or miss of specs for initial def reduction
    def spec_hit(status):
        damage_val = hit_value_roll(spec_bonus=True, four_tick=False, five_tick=False)
        tekton.lower_hp(damage_val)
        tekton.lower_def(int((tekton.defence * .3)))
        adjust_def_integer()
        hit_metrics.hammer_hit_count += 1
        defence_roll(True, False, False, status)
        if hit_chancer(True, False, False, False, status):
            damage_val = hit_value_roll(spec_bonus=True, four_tick=False, five_tick=False)
            tekton.lower_hp(damage_val)
            if damage_val > 0:
                tekton.lower_def(int((tekton.defence * .3)))
                adjust_def_integer()
                hit_metrics.hammer_hit_count += 1
            else:
                hammer_missed()
        else:
            hammer_missed()
        return


    # Function that will be called to indicate number of hits in damage phase for mace
    def four_tick_hit(instances, status):
        for _ in range(instances):
            if tekton.hp > 0:
                hit_metrics.four_tick_hit_counter += 1
            else:
                hit_metrics.four_tick_hit_counter += 0
            defence_roll(False, True, False, status)
            if hit_chancer(False, True, False, False, status):
                damage_val = hit_value_roll(spec_bonus=False, four_tick=True, five_tick=False)
                tekton.lower_hp(damage_val)
            else:
                damage_val = 0
                tekton.lower_hp(damage_val)
        return


    # Function that will be called to make each scythe hit roll seperately for each of the 3 instances of dmg
    def scy_dmg(step_down, status):
        if hit_chancer(False, False, True, False, status):
            damage_val = hit_value_roll(False, four_tick=False, five_tick=True, max_hit_modifier=step_down)
            tekton.lower_hp(damage_val)
        else:
            damage_val = 0
            tekton.lower_hp(damage_val)
        return damage_val


    # Function that will be called to indicate number of hits in damage phase for 5 tick weapon
    def five_tick_hit(instances, status, fang_spec_pass_var):
        for _ in range(instances):
            defence_roll(False, False, True, status)
            if tekton.hp > 0:
                hit_metrics.five_tick_hit_counter += 1
            if trial_parameters.fang:
                if fang_spec_pass_var:
                    if hit_chancer(False, False, True, fang_spec_pass_var, status):
                        damage_val = hit_value_roll(spec_bonus=False, four_tick=False, five_tick=True, max_hit_modifier=1)
                        tekton.lower_hp(damage_val)
                    else:
                        damage_val = 0
                        tekton.lower_hp(damage_val)
                elif hit_chancer(False, False, True, fang_spec_pass_var, status):
                    damage_val = hit_value_roll(spec_bonus=False, four_tick=False, five_tick=True, max_hit_modifier=.85)
                    tekton.lower_hp(damage_val)
                else:
                    damage_val = 0
                    tekton.lower_hp(damage_val)
            elif scythe:
                for i in [1, .5, .25]:
                    scy_dmg(i, status)
            else:
                if hit_chancer(False, False, True, False, status):
                    damage_val = hit_value_roll(False, four_tick=False, five_tick=True)
                    tekton.lower_hp(damage_val)
                else:
                    damage_val = 0
                    tekton.lower_hp(damage_val)
        return


    def veng_calc():
        if trial_parameters.cm:
            if trial_parameters.veng_camp:
                if tekton.veng_count < 2:
                    return 58
                else:
                    return 65
            else:
                return 65
        else:
            if trial_parameters.veng_camp:
                if tekton.veng_count < 2:
                    return 39
                else:
                    return 44
            else:
                return 44


    def veng_applicator(pre_veng_check):
        if pre_veng_check:
            tekton.lower_hp(random.randint(1, veng_calc()))
            return tekton.hp
        elif trial_parameters.veng_camp:
            for _ in range(2):
                if tekton.veng_count < 4:
                    if hit_metrics.hp_pool > veng_calc():
                        tekton.lower_hp(random.randint(1, veng_calc()))
                    elif hit_metrics.hp_pool > (veng_calc() * 2):
                        tekton.lower_hp(random.randint(1, math.ceil((veng_calc() * .5))))
                    else:
                        return tekton.hp
                    tekton.veng_count += 1
                else:
                    return tekton.hp
        else:
            return tekton.hp


    def anvil_adjustment():
        if tekton.hp > 0:
            cycle_select = random.randint(3, 6)
            hit_metrics.idle_time += ((cycle_select * 3) + 10)
        else:
            return tekton.hp
        return tekton.hp


    def min_regen():
        if tekton.hp > 0:
            if tekton.hp < base_hp:
                tekton.hp += 1
                return tekton.hp
            if tekton.defence < base_def:
                tekton.defence += 1
                return tekton.hp
            return tekton.hp
        else:
            return tekton.hp


    def time():
        idle_total = hit_metrics.idle_time
        four_total = hit_metrics.four_tick_hit_counter * 4
        five_total = hit_metrics.five_tick_hit_counter * 5
        hit_metrics.time_parameter = (five_total + four_total + idle_total + 12 + 17) * 0.6
        times.append(hit_metrics.time_parameter)
        hit_metrics.time_parameter = (five_total + four_total + idle_total + 12 + 17)
        tick_times_list.append(hit_metrics.time_parameter)
        anvil_count_list.append(hit_metrics.phase)
        return


    def pre_anvil():
        if four_and_five:
            spec_hit(False)
            hammer_check()
            # hammer_count_list.append(hit_metrics.hammer_hit_count)
            for four_num, five_num in [(3, 1), (1, 2)]:
                four_tick_hit(four_num, False)
                five_tick_hit(five_num, False, False)
        else:
            spec_hit(False)
            hammer_check()
            # hammer_count_list.append(hit_metrics.hammer_hit_count)
            five_tick_hit(6, False, False)
        return


    def can_i_spec():
        if trial_parameters.ring == 'lightbearer':
            hit_metrics.fang_spec_status = True
            return hit_metrics.fang_spec_status
        else:
            if hit_metrics.specced_last_anvil:
                hit_metrics.fang_spec_status = False
            else:
                hit_metrics.fang_spec_status = True
            return hit_metrics.fang_spec_status


    def post_anvil(spec_alternation):
        hit_metrics.phase += 1
        if four_and_five:
            four_tick_hit(5, True)
            if trial_parameters.ring == 'lightbearer':
                if spec_alternation:
                    five_tick_hit(1, False, True)
                    hit_metrics.specced_last_anvil = True
                    can_i_spec()
                else:
                    five_tick_hit(1, False, False)
                    hit_metrics.specced_last_anvil = False
                    can_i_spec()
                four_tick_hit(6, False)
            else:
                four_tick_hit(6, False)
                five_tick_hit(1, False, False)
            if trial_parameters.short_lure:
                four_tick_hit(1, False)
            else:
                four_tick_hit(2, False)

            five_tick_hit(1, False, False)
        else:
            five_tick_hit(4, True, False)
            if trial_parameters.ring == 'lightbearer':
                if spec_alternation:
                    five_tick_hit(1, False, True)
                    hit_metrics.specced_last_anvil = True
                    can_i_spec()
                else:
                    five_tick_hit(1, False, False)
                    hit_metrics.specced_last_anvil = False
                    can_i_spec()
                five_tick_hit(7, False, False)
            else:
                five_tick_hit(8, False, False)
        return


    def defence_roll(spec, four_tick, five_tick, enraged):
        test_weapon = ""
        if enraged:
            tekton.stab_def, tekton.slash_def, tekton.crush_def = [280, 280, 180]
        else:
            tekton.stab_def, tekton.slash_def, tekton.crush_def = [155, 165, 105]
        if spec or four_tick:
            test_weapon = loadout.static_crush_weapon
        elif five_tick:
            test_weapon = loadout.five_tick_weapon
        def_roll_dict = {'crush': math.ceil((tekton.defence + 9) * (tekton.crush_def + 64)),
                         'stab': math.ceil((tekton.defence + 9) * (tekton.stab_def + 64)),
                         'slash': math.ceil((tekton.defence + 9) * (tekton.slash_def + 64))}
        return random.randint(0, def_roll_dict[test_weapon])


    def sql_import():
        print(len(anvil_count_list), len(hammer_count_list))

        results_df = pd.DataFrame(list(zip(tick_times_list, anvil_count_list, hammer_count_list, hp_check_list)),
                                  columns=['tick_times', 'anvil_count', 'hammer_count', 'hp_after_pre_anvil'])
        print(param_container)
        for name in param_container:
            if param_container[name]:
                results_df[name] = 1
            else:
                results_df[name] = 0

        results_df['ring'] = trial_parameters.ring

        print(results_df)
        import_df = results_df.copy()
        connection_string = r"Driver={ODBC Driver 17 for SQL Server}; Server=DESKTOP-J8L86O2; Database=tekton_sim_data; Trusted_Connection=yes;"
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        engine = create_engine(connection_url)
        conn_id = engine.connect()
        conn = engine.connect()
        max_id = conn_id.execute(text("""SELECT max(ID)
                                FROM tekton_results""")).first()[0]
        if max_id == None:
            max_id = 0
        conn_id.close()
        import_df.index += max_id + 1
        print(import_df)
        # noinspection PyUnboundLocalVariable
        import_df.to_sql('tekton_results', con=conn, if_exists='append', index_label='ID')
        return import_df



    if param_container:
        for x in range(10000):
            tekton.reset()
            hit_metrics.reset()
            if four_and_five:
                vuln_applicator()
                pre_anvil()
                veng_applicator(trial_parameters.preveng)
                hp_check_list.append(tekton.hp)
                anvil_adjustment()
                hit_metrics.hp_pool += 44
                min_regen()
                while True:
                    if tekton.hp > 0:
                        post_anvil(spec_alternation=hit_metrics.fang_spec_status)
                        veng_applicator(False)
                        anvil_adjustment()
                        min_regen()
                        continue
                    else:
                        time()
                        break
            elif trial_parameters.five_tick_only:
                vuln_applicator()
                pre_anvil()
                veng_applicator(trial_parameters.preveng)
                hp_check_list.append(tekton.hp)
                anvil_adjustment()
                hit_metrics.hp_pool += 44
                min_regen()
                while True:
                    if tekton.hp > 0:
                        post_anvil(spec_alternation=hit_metrics.fang_spec_status)
                        veng_applicator(False)
                        anvil_adjustment()
                        min_regen()
                        continue
                    else:
                        time()
                        break
    sql_import()
    return



# if sql_import.get():
#     temp = input('port to sql?')
#     if temp == 'y':
#         max_id = conn_id.execute(text("""SELECT max(ID)
#                         FROM tekton_results""")).first()[0]
#         if max_id == None:
#             max_id = 0
#         conn_id.close()
#         import_df.index += max_id + 1
#         print(import_df)
#         # noinspection PyUnboundLocalVariable
#         import_df.to_sql('tekton_results', con=conn, if_exists='append', index_label='ID')
# if __name__ == '__main__':
#     sys.argv[] #pass arguments if given and whatnot
#     map_parameters(sys.argv)