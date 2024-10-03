# Stratified sampling procedure
import numpy as np
import pandas as pd


def all_peers_meet_minimum_firm_count(data, level, sector_columns, minimum_firm_count):
    """
    Checks whether all peer groups at a specific sector level have at least the minimum number of firms.

    This function first identifies the parent sector of the current level and retrieves all sibling sectors
    under this parent. It then counts the number of unique instruments in each sibling sector and checks if
    all these sibling sectors meet the minimum firm count requirement.

    Args:
        data (pd.DataFrame): The DataFrame containing the instruments and sector information.
        level (int): The index of the sector level to check, with 0 being the broadest and increasing
                     indices corresponding to more specific levels.
        sector_columns (list of str): A list of sector column names ordered from the broadest to the most specific.
        minimum_firm_count (int): The minimum number of firms required in each peer group.

    Returns:
        tuple: A tuple containing:
            - bool: Whether all peer groups meet the minimum firm count.
            - np.ndarray: An array of sibling sector names at the specified level.
    """
    tests = []
    child = data.iloc[0][sector_columns[level]]
    parent = data.loc[data[sector_columns[level]] == child, sector_columns[level-1]].iloc[0]
    siblings = pd.Series(data.loc[data[sector_columns[level-1]] == parent, sector_columns[level]]).unique()
    
    for sibling in siblings:
        n_firms = data.loc[data[sector_columns[level]] == sibling, 'instrument'].nunique()
        tests.append(n_firms >= minimum_firm_count)

    tests_all_pass = all(tests)
    return tests_all_pass, siblings


def store_peer_group(data, level, sector_columns, siblings, results_list, assigned_instruments):
    """
    Stores instruments into peer groups based on the sector level and removes them from the dataset.

    Args:
        data (pd.DataFrame): The DataFrame containing the instruments and sector information.
        level (int): The index of the sector level being processed.
        sector_columns (list of str): A list of sector column names ordered from the broadest to the most specific.
        siblings (np.ndarray): An array of sibling sector names at the specified level.
        results_list (list): A list to store the peer group results, with each entry being a DataFrame.
        assigned_instruments (set): A set to keep track of instruments that have already been assigned to a peer group.

    Returns:
        pd.DataFrame: The modified DataFrame with the assigned instruments removed.
    """
    instruments_to_drop = []
    
    for ind in siblings:
        # Only consider instruments that haven't been assigned yet
        instruments = data.loc[(data[sector_columns[level]] == ind) & (~data['instrument'].isin(assigned_instruments)), 'instrument'].unique()
        if instruments.size > 0:
            sub = pd.DataFrame({'instrument': instruments, 'peer_group': ind})
            results_list.append(sub)
            assigned_instruments.update(instruments)  # Mark these instruments as assigned

            # Collect instruments to drop
            instruments_to_drop.extend(instruments)

    # Filter out these instruments from the main dataframe
    data = data.loc[~data['instrument'].isin(instruments_to_drop)]
    return data


def build_peer_group(data, sector_columns, minimum_firm_count):
    """
    Constructs peer groups for instruments by iteratively checking sector levels from specific to broad.

    This function starts from the most specific sector level and checks if all peer groups at that level meet
    the minimum firm count. If they do, it stores the peer group and removes those instruments from further
    processing. If not, it moves up to a broader sector level and repeats the process. The function continues
    until all instruments have been assigned to a peer group or all sector levels have been exhausted.

    Args:
        data (pd.DataFrame): The DataFrame containing the instruments and sector information.
        sector_columns (list of str): A list of sector column names ordered from the broadest to the most specific.
        minimum_firm_count (int): The minimum number of firms required in each peer group.

    Returns:
        pd.DataFrame: A DataFrame containing the peer group assignments for each instrument.
    """
    results = []
    assigned_instruments = set()  # Track assigned instruments
    level = len(sector_columns) - 1
    
    while len(data) > 0:
        # Check if all remaining firms belong to the same sector across all levels
        unique_sector_combinations = data[sector_columns].drop_duplicates()
        
        if len(unique_sector_combinations) == 1:
            # Assign all remaining firms to their activity sector
            data['peer_group'] = data['activity_sector']
            results.append(data[['instrument', 'peer_group']])
            break
        
        # Start from the deepest level
        current_level = level
        
        while current_level >= 0:
            tests_all_pass, siblings = all_peers_meet_minimum_firm_count(data, current_level, sector_columns, minimum_firm_count)
            if tests_all_pass:
                data = store_peer_group(data, current_level, sector_columns, siblings, results, assigned_instruments)
                break  # Exit the loop and start again from the deepest level with the reduced data
            else:
                current_level -= 1  # Move to the broader sector level
        
        # After processing, reset level to start from the deepest level again
        level = len(sector_columns) - 1
    
    # Combine results into a single DataFrame
    peer_groups_df = pd.concat(results, ignore_index=True)
    
    # Remove duplicate instruments, keeping the first occurrence
    peer_groups_df = peer_groups_df.drop_duplicates(subset='instrument', keep='first')
    
    return peer_groups_df

def get_peer_groups_and_folds(df, target_variable, sector_features, minimum_firms_per_fold, k_folds, seed_num, verbose=False):
    print(f"Initial number of observations: {len(df)}")

    df = df[df[target_variable].notnull()]
    print(f"After filtering for non-missing target variable: {len(df)}")

    # Build peer groups - this returns a DataFrame with one row per unique instrument
    peer_groups_df = build_peer_group(data=df.drop_duplicates(subset='instrument'), sector_columns=sector_features, minimum_firm_count=minimum_firms_per_fold)
    
    # Assign folds within each peer group
    data_with_folds = assign_folds_within_peer_groups(peer_groups_df, peer_group_col='peer_group', k=k_folds, random_state=seed_num)
    
    # Merge the peer group and fold assignments back into the original DataFrame
    df = df.merge(data_with_folds[['instrument', 'peer_group', 'fold']], on='instrument', how='left')
    print(f"After merging peer groups and folds back into df: {len(df)}")

    if verbose:
        # Print summaries as before
        unique_firms_per_fold = df.drop_duplicates(subset='instrument')['fold'].value_counts().sort_index()
        observations_per_fold = df['fold'].value_counts().sort_index()
        print(f"\nFold Summary for Target Variable '{target_variable}':")
        print(pd.DataFrame({
            'Unique Firms': unique_firms_per_fold,
            'Total Observations': observations_per_fold
        }))
        
        print(f"\nPeer Group Summary Split by Fold:")
        print(df.groupby(['peer_group', 'fold']).agg(
            Unique_Firms=('instrument', 'nunique'),
            Total_Observations=('instrument', 'count')
        ).reset_index().pivot(index='peer_group', columns='fold', values=['Unique_Firms', 'Total_Observations']))
    
    return df


def min_firms_per_peer_group(folds, min_firms_per_fold):
    return folds * min_firms_per_fold


def assign_folds_within_peer_groups(data, peer_group_col='peer_group', k=5, random_state=None):
    """
    Randomly assigns firms within each peer group to one of k folds, ensuring roughly equal distribution per peer group.

    Args:
        data (pd.DataFrame): DataFrame containing the data with assigned peer groups.
        peer_group_col (str): The column name containing peer group information.
        k (int): Number of folds.
        random_state (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'fold' column indicating the assigned fold.
    """
    
    data = data.reset_index(drop=True)
    
    # Initialize a list to store fold assignments
    fold_assignments = np.zeros(len(data), dtype=int)
    
    # Set the random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        
    grouped = data.groupby(peer_group_col)

    # Iterate over each group
    for _, group in grouped:
        group_indices = group.index.to_numpy()

        # Check if any index exceeds the size of fold_assignments
        if group_indices.max() >= len(fold_assignments):
            raise IndexError(f"Index {group_indices.max()} is out of bounds for axis 0 with size {len(fold_assignments)}")

        np.random.shuffle(group_indices)
        
        # Assign folds in a round-robin fashion
        for i, idx in enumerate(group_indices):
            fold_assignments[idx] = (i % k) + 1  # Folds are 1-indexed

    data['fold'] = fold_assignments
    return data


def get_peer_groups_and_folds(df, target_variable, sector_features, minimum_firms_per_fold, k_folds, seed_num, verbose=False):
    """
    Assigns peer groups and folds to each instrument within the dataset based on sector features and specified criteria.

    This function performs the following steps:
    1. Subsets the dataframe to where the target variable is non-missing.
    2. Determines the minimum number of firms required per peer group based on the specified minimum firms per fold and the number of folds.
    3. Builds peer groups for each instrument based on the smallest sector level where the peer group meets the minimum firm requirement.
    4. Randomly assigns instruments within each peer group to one of k folds, ensuring that the distribution is roughly equal across folds.
    5. Merges the assigned peer groups and folds back into the original DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing instrument data and sector features.
        target_variable (str): The target variable we want to model on.
        sector_features (list of str): A list of column names representing the sector hierarchy, ordered from broadest to most specific.
        minimum_firms_per_fold (int): The minimum number of firms required per fold.
        k_folds (int): The number of folds to create.
        seed_num (int): Seed for the random number generator to ensure reproducibility in fold assignment.
        verbose (bool): If True, prints the summary of firms and observations per fold and peer group.

    Returns:
        pd.DataFrame: The original DataFrame with additional columns for 'peer_group' and 'fold', indicating the assigned peer group and fold for each instrument.
    """
    # Filter out rows where the target variable is missing
    df = df[df[target_variable].notnull()]
    
    # Determine the minimum number of firms required per peer group
    minimum_firm_count = min_firms_per_peer_group(folds=k_folds, min_firms_per_fold=minimum_firms_per_fold)
    
    # Check if this number exceeds the number of unique firms in the broadest sector level
    min_firms_in_broadest_sector = df.groupby(sector_features[0])['instrument'].nunique().min()
    if minimum_firm_count > min_firms_in_broadest_sector:
        raise ValueError(f"The required minimum number of firms per peer group ({minimum_firm_count}) exceeds the number of unique firms "
                         f"in the broadest sector level ({min_firms_in_broadest_sector}). Adjust the number of folds or minimum firms per fold.")

    # Build peer groups
    peer_groups_df = build_peer_group(data=df, sector_columns=sector_features, minimum_firm_count=minimum_firm_count)

    # Assign folds within each peer group
    data_with_folds = assign_folds_within_peer_groups(peer_groups_df, peer_group_col='peer_group', k=k_folds, random_state=seed_num)
    
    # Merge the assigned peer groups and folds back into the original DataFrame
    df = df.merge(data_with_folds, on='instrument', how='left')
   
    if verbose:
        # Firms and observations per fold
        unique_firms_per_fold = df.drop_duplicates(subset='instrument')['fold'].value_counts().sort_index()
        observations_per_fold = df['fold'].value_counts().sort_index()
        
        print(f"\nFold Summary for Target Variable '{target_variable}':")
        fold_summary = pd.DataFrame({
            'Unique Firms': unique_firms_per_fold,
            'Total Observations': observations_per_fold
        })
        print(fold_summary)
        
        # Firms and observations per peer group, split by fold
        peer_group_fold_summary = df.groupby(['peer_group', 'fold']).agg(
            Unique_Firms=('instrument', 'nunique'),
            Total_Observations=('instrument', 'count')
        ).reset_index()
        
        print(f"\nPeer Group Summary Split by Fold:")
        print(peer_group_fold_summary.pivot(index='peer_group', columns='fold', values=['Unique_Firms', 'Total_Observations']))
    
    return df