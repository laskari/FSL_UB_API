from collections import defaultdict


def merge_donut_output(donut_out_old, donut_out_new, keys_from_old):
    try:
        print("In process of merging from OLD keys")
        old_values_for_keys = donut_out_old.set_index("Key").loc[keys_from_old, 'Value'].to_dict()

        donut_out_new['Value'] = donut_out_new.apply(
            lambda row: old_values_for_keys.get(row['Key'], row['Value']),
            axis = 1
        )

        return donut_out_new[['Key', 'Value']]
    
    except Exception as e:
        raise e
    

def merge_key_aggregated_scores(scores_old, scores_new, keys_from_old):
    """
    Merges two key aggregated scores dictionaries, updating values for specified keys from the old scores.

    Parameters:
    scores_old (defaultdict(float)): The old key aggregated scores.
    scores_new (defaultdict(float)): The new key aggregated scores.
    keys_from_old (list): A list of keys to retain values from the old scores.

    Returns:
    defaultdict(float): Merged key aggregated scores.
    """
    # Create a copy of the new scores to avoid modifying the original
    merged_scores = defaultdict(float, scores_new)
    
    # Update values for specified keys from the old scores
    for key in keys_from_old:
        if key in scores_old:
            merged_scores[key] = scores_old[key]

    return merged_scores