import pandas as pd
def get_turns_on_red(env) -> pd.DataFrame:
    """
    Extract all the legal turns on red - the right turns which are always possible - from the environment's non-fluents.
    If turn-on-red is False, the df will return empty.
    """
    
    ALL_RED4 = env.model._objects['signal-phase'][6]

    turns_on_red = []
    for k,v in env.non_fluents.items():
        if ('GREEN' in k) and (ALL_RED4 in k) and v:
            k = k.replace('__', '-')
            green, l, from_1, to_1, l, from_2, to_2, all, red = k.split('-')
            turns_on_red.append({
                'from': from_1, 
                'pivot': to_1,
                'to': to_2,
                'is_turn_on_red': True
                })
                
    return pd.DataFrame(turns_on_red)

def get_phases(env, turns_on_red: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the data about the phases -
    Which phases do we have and what do they mean?
    """
    
    # greens = []
    # for v in env.model._nonfluents['GREEN']:
    #     if v:
    #         greens.append(v)
    # greens = pd.DataFrame(greens)
    
    phases = []
    for k,v in env.non_fluents.items():
        if ('LEFT' in k) or ('THROUGH' in k) and v:
            k = k.replace('__', '-')
            green, l, from_1, to_1, l, from_2, to_2, nw, se, left_thourgh = k.split('-')
            if(to_1 == from_2 and from_1 != to_2):
                phases.append({
                    'dir': nw + '_' + se,
                    'left_through': left_thourgh,
                    'from': from_1, 
                    'pivot': to_1,
                    'to': to_2
                })
    phases = pd.DataFrame(phases)
            
    new_df = pd.merge(phases, turns_on_red,  how='left', on=['from','pivot','to'])
    new_df = new_df.fillna('False')
    new_df = new_df[new_df['is_turn_on_red'] != True].iloc[:,:-1].reset_index(drop=True)
    
    turns = []
    for k,v in env.non_fluents.items():
        if ('TURN' in k) and v:
            k = k.replace('__', '-')
            turn, l, from_1, to_1, l, from_2, to_2 = k.split('-')
            turns.append({
                'from': from_1, 
                'pivot': to_1,
                'to': to_2,
                'turn': True
            })
    turns = pd.DataFrame(turns)
    