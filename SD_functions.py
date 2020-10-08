def SDlabel_trials(behavioral_df):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()
    behavioral_df["SD_CONDITION"] = ""
    row = 0
    for idx, col in behavioral_df.iterrows():
        if (behavioral_df.loc[row, "Stim_diff"] < 0) & (behavioral_df.loc[row, "responseError"] < 0):
            behavioral_df.set_value(row, "SD_CONDITION", "SD")
            row += 1
        elif (behavioral_df.loc[row, "Stim_diff"] > 0) & (behavioral_df.loc[row, "responseError"] > 0):
            behavioral_df.set_value(row, "SD_CONDITION", "SD")
            row += 1
        elif np.isnan(behavioral_df.loc[row, "Stim_diff"]) or np.isnan(behavioral_df.loc[row, "responseError"]):
            row += 1
        else:
            behavioral_df.set_value(row, "SD_CONDITION", "NO_SD")
            row += 1
    return behavioral_df