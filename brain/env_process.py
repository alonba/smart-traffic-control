import pandas as pd
def get_turns_on_red(env) -> pd.DataFrame:
    """
    Extract all the legal turns on red - the right turns which are always possible - from the environment's non-fluents.
    If turn-on-red is False, the df will return empty.
    """
    
    turns_on_red = []
    for k,v in env.non_fluents.items():
        if ('GREEN' in k) and ('ALL-RED4' in k) and v:
            k = k.replace('__', '-')
            green, l, from_1, to_1, l, from_2, to_2, all, red = k.split('-')
            turns_on_red.append({
                'from': from_1, 
                'pivot': to_1,
                'to': to_2,
                'is_turn_on_red': True
                })
                
    return pd.DataFrame(turns_on_red)

def get_phases(env) -> pd.DataFrame:
    """
    Extract the data about the phases -
    Which phases do we have and what do they mean?
    """
    
    ALL_RED = 'ALL-RED'
    
    phases_info = []
    green_turns_raw = env._visualizer.green_turns_by_intersection_phase
    for agent_name in green_turns_raw:
        turns_on_red = set(green_turns_raw[agent_name][ALL_RED])
        for phase_name, green_turns_during_phase in green_turns_raw[agent_name].items():
            if ALL_RED not in phase_name:
                turns_in_phase_without_turns_on_red = set(green_turns_during_phase) - turns_on_red
                for turn in turns_in_phase_without_turns_on_red:
                    turn_parsed = {
                        'phase': phase_name,
                        'from': turn[0].split('-')[1],
                        'pivot': agent_name,
                        'to': turn[1].split('-')[2]
                    }
                    phases_info.append(turn_parsed)
                    
    return pd.DataFrame(phases_info)