import pandas as pd
def get_turns_on_red(env):
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
                'to': to_2
                })
                
    return pd.DataFrame(turns_on_red)