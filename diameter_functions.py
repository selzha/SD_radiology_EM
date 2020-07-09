import pandas as pd
from statsFuncs import mean_confidence_interval, numbers
import numpy as np
import matplotlib.pyplot as plt
import pupil_functions
import extract_diameter


def main(behavioral_path, EM_path, subINIT, baseline_correction,  times_selected, title, time_limit=3, ylimit=[-0.2, 0.5], ylabel='Pupil diameter change\n from fixation baseline', xlabel='Time from Image Onset (secs)'):
    """
    :param behavioral_path: str: path to folder where csv file with SD behavioral data is stored
    :param EM_path: str: path to folder with eye-tracking data
    :param subINIT: str: subject initials or subject identification
    :param baseline_correction: str: either 'PERTRIAL' or 'ALLTRIALS'.
    :param times_selected: np.ndarray: timestamps to average pupil size across all the period.
    :param title:
    :param time_limit:
    :param ylimit:
    :param ylabel:
    :param xlabel:
    :return:
    """

    # behavioral_path: str: path to folder where csv file with SD behavioral data is stored
    # EM_path:
    # subINIT:
    # baseline_correction:
    # times_selected: n
            # To calculate this, use linspace inputing time_before_onset, max_time and how many samples you want.


    # load behavioral data:
    behavioral_data = pd.read_csv(behavioral_path + subINIT + '_VM_output_1nBack.csv')

    # cut file just to have the 255 rows of data
    behavioral_data = behavioral_data.iloc[range(255)]

    # load timestamps
    timestamp_path = EM_path + 'world_timestamps.npy'
    annotation_data = pupil_functions.load_annotations(EM_path)

    # extract diameter from eye-tracking data and save as csv
    extract_diameter.main(recordings=[EM_path], csv_out=subINIT + '_df_diameter.csv')

    df_diameter = pd.read_csv(EM_path + subINIT + '_df_diameter.csv')

    df_final, trial_sd = pupil_preprocess(df_diameter, annotation_data, behavioral_data)

    if baseline_correction == 'PERTRIAL':

        print('Calculating baseline correction...')
        df_final, fixation_baseline = pupil_baseline(df_final, behavioral_data)
        print('Aligning timestamps...')
        df_matched = align_timestamps_pupil(df_final, behavioral_data, times_selected, 'PS_CORRECTED')

    elif baseline_correction == 'ALLTRIALS':
        print('Calculating Baseline Correction...')
        df_final = pupil_baseline_alltrials(df_final)
        print('Aligning timestamps...')
        df_matched = align_timestamps_pupil(df_final, behavioral_data, times_selected, 'PS_CORRECTED_ALLTRIALS')

    print('Calculating means per timestamp...')
    df_sd, df_nosd = tsmeans_SD_pupil(df_matched, times_selected, 'PUPIL_SIZE_CORRECTED')

    final_title = title + '_' + subINIT
    print('Outputting plot...')
    plot_sd_category(df_sd, df_nosd, time_limit, ylimit, ylabel, xlabel, final_title, behavioral_path)

    print('Done!')
def tsmeans_SD_pupil(df_matched, times_selected, pupil_column_label):
    conditions = ['SD', 'NO_SD']

    for cond in conditions:
        df_matched_wide = pd.DataFrame()
        df_matched_wide.loc[:, 'TIMESTAMP_IMONSET'] = times_selected

        df = df_matched[df_matched['SD_CONDITION'] == cond]

        # get number of trials in this condition
        numTrials = len(df.BLOCK.unique()) * len(df.TRIAL.unique())

        for block in df.BLOCK.unique():
            for trial in df.TRIAL.unique():
                df_tmp = df[(df['BLOCK'] == block)
                            & (df['TRIAL'] == trial)].reset_index(drop=True)

                #             print(df_tmp['PUPIL_SIZE_CORRECTED'])

                column_name = 'BLOCK' + str(block) + 'TRIAL' + str(trial)
                df_matched_wide[column_name] = df_tmp[pupil_column_label]

        # calculate means across trials
        for row in range(len(df_matched_wide)):
            mean, low, high = mean_confidence_interval.mean_sem(df_matched_wide.iloc[row, range(1, numTrials + 1)])
            df_matched_wide.loc[row, 'MEAN'] = mean
            df_matched_wide.loc[row, 'SEM_LOW'] = low
            df_matched_wide.loc[row, 'SEM_HIGH'] = high

            # get number of NaNs on that timestamp
            df_matched_wide.loc[row, 'COUNT_NAN'] = (np.isnan(
                df_matched_wide.iloc[row, range(1, numTrials + 1)]).sum()) / numTrials

        if cond == 'SD':
            df_matched_wide_sd = df_matched_wide

        elif cond == 'NO_SD':
            df_matched_wide_nosd = df_matched_wide

    return df_matched_wide_sd, df_matched_wide_nosd


def initial_clean(df, annotations_df, min_confidence=0.8, min_diameter=1.5, max_diameter=9):
    df_cleaned = df[(df['confidence'] >= min_confidence) &
                    (df['diameter_3d [mm]'] >= min_diameter) &
                    (df['diameter_3d [mm]'] <= max_diameter)]

    exp_begins_ts = annotations_df[annotations_df['exp_event'] == 'Block 1 Begins']['timestamp'].reset_index(drop=True)[
        0]
    exp_ends_ts = annotations_df[annotations_df['exp_event'] == 'Block 3 Ends']['timestamp'].reset_index(drop=True)[0]

    perc_trials_bad_confidence = (len(df[df['confidence'] < min_confidence]) / len(df)) * 100
    print(str(round(perc_trials_bad_confidence, 2)) + '% of data excluded because < mininum confidence')

    perc_trials_outlier = (len(
        df[(df['diameter_3d [mm]'] < min_diameter) & (df['diameter_3d [mm]'] > max_diameter)]) / len(df)) * 100
    print(str(round(perc_trials_outlier, 2)) + '% of data excluded because of pupil size is out of normal values')

    df_cleaned = df_cleaned[
        (df_cleaned['timestamp'] > exp_begins_ts) & (df_cleaned['timestamp'] < exp_ends_ts)].reset_index(drop=True)

    perc_good_data = (len(df_cleaned) / len(df)) * 100
    print('Total of ' + str(round(perc_good_data)) + '% of data kept')

    return df_cleaned


def correct_timestamps(df, annotations_df, behavioral_df):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()

    for block in block_unique:
        if not np.isnan(block):

            for trial_number in trial_unique:
                if not np.isnan(trial_number):
                    # get timestamps of trial events
                    trial_beg_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                              annotations_df['task'] == 'Trial begins')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_end_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                              annotations_df['task'] == 'End trial')][
                        'timestamp'].reset_index(drop=True)[0]

                    trial_imOnset_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                  annotations_df['task'] == 'Image onset')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_fix_beg_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                  annotations_df['task'] == 'Fixation dot')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_fix_end_ts = trial_imOnset_ts
                    trial_adjustment_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                     annotations_df['task'] == 'Adjust task')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_adjustment_ts_corrected = trial_adjustment_ts - trial_imOnset_ts

                    trial = df[(df['timestamp'] >= trial_beg_ts) & (df['timestamp'] <= trial_end_ts)]
                    trial['TIMESTAMP_CORRECTED'] = trial['timestamp'] - trial_beg_ts
                    # now the timestamp correct is in relation to trial beginning of trial

                    # now lets also have a timestamp in relation to image onset
                    trial['TIMESTAMP_CORRECTED_IMONSET'] = trial['timestamp'] - trial_imOnset_ts

                    df.loc[(df['timestamp'] >= trial_beg_ts) & (df['timestamp'] <= trial_end_ts), 'TIMESTAMP_CORRECT'] = \
                    trial['TIMESTAMP_CORRECTED']
                    df.loc[(df['timestamp'] >= trial_beg_ts) & (
                                df['timestamp'] <= trial_end_ts), 'TIMESTAMP_CORRECT_IMONSET'] = trial[
                        'TIMESTAMP_CORRECTED_IMONSET']

                    # label trials and blocks in eye movement data
                    df.loc[(df['timestamp'] >= trial_beg_ts) & (
                                df['timestamp'] <= trial_end_ts), 'TRIAL'] = trial_number
                    df.loc[(df['timestamp'] >= trial_beg_ts) & (
                                df['timestamp'] <= trial_end_ts), 'BLOCK'] = block

    return df


def label_trials_sd(df, behavioral_df):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()

    trial_sd = pd.DataFrame()
    row = 0
    for block in block_unique:
        if not np.isnan(block):

            # print('Processing block ' + str(block) + ' out of ' + str(len(block_unique)))

            for trial in trial_unique:
                if not np.isnan(trial):

                    trial_data = behavioral_df[
                        (behavioral_df['blockNumber'] == block) & (behavioral_df['trialNumber'] == trial)].reset_index(
                        drop=True)

                    if not trial_data.empty:
                        if (trial_data.loc[0, 'Stim_diff'] < 0) & (trial_data.loc[0, 'responseError'] < 0):

                            trial_sd.loc[row, 'BLOCK'] = block
                            trial_sd.loc[row, 'TRIAL'] = trial
                            trial_sd.loc[row, 'SD_CONDITION'] = 'SD'

                            df.loc[df['TRIAL'] == trial, 'SD_CONDITION'] = 'SD'

                            row += 1

                        elif (trial_data.loc[0, 'Stim_diff'] > 0) & (trial_data.loc[0, 'responseError'] > 0):
                            trial_sd.loc[row, 'BLOCK'] = block
                            trial_sd.loc[row, 'TRIAL'] = trial
                            trial_sd.loc[row, 'SD_CONDITION'] = 'SD'

                            df.loc[df['TRIAL'] == trial, 'SD_CONDITION'] = 'SD'

                            row += 1
                        else:

                            trial_sd.loc[row, 'BLOCK'] = block
                            trial_sd.loc[row, 'TRIAL'] = trial
                            trial_sd.loc[row, 'SD_CONDITION'] = 'NO_SD'

                            df.loc[df['TRIAL'] == trial, 'SD_CONDITION'] = 'NO_SD'

                            row += 1

    return df, trial_sd


def label_events(df, annotations_df, behavioral_df):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()

    for block in block_unique:
        if not np.isnan(block):

            for trial_number in trial_unique:
                if not np.isnan(trial_number):
                    # get timestamps of trial events

                    # trial starts & end
                    trial_beg_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                              annotations_df['task'] == 'Trial begins')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_end_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                              annotations_df['task'] == 'End trial')][
                        'timestamp'].reset_index(drop=True)[0]

                    # fixation starts & end
                    trial_fix_beg_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                  annotations_df['task'] == 'Fixation dot')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_fix_end_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                  annotations_df['task'] == 'Image onset')][
                        'timestamp'].reset_index(drop=True)[0]

                    df.loc[(df['timestamp'] >= trial_fix_beg_ts) & (
                                df['timestamp'] < trial_fix_end_ts), 'EVENT'] = 'fixation'

                    # image starts & end
                    trial_imOnset_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                  annotations_df['task'] == 'Image onset')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_imOffset_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                   annotations_df['task'] == 'Adjust task')][
                        'timestamp'].reset_index(drop=True)[0]

                    df.loc[
                        (df['timestamp'] >= trial_imOnset_ts) & (df['timestamp'] < trial_imOffset_ts), 'EVENT'] = 'blob'

                    # adjust starts & end
                    trial_adjust_beg_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                     annotations_df['task'] == 'Adjust task')][
                        'timestamp'].reset_index(drop=True)[0]
                    trial_adjust_end_ts = annotations_df[(annotations_df['block'] == str(int(block))) & (
                                annotations_df['trial'] == str(int(trial_number))) & (
                                                                     annotations_df['task'] == 'End trial')][
                        'timestamp'].reset_index(drop=True)[0]

                    df.loc[(df['timestamp'] >= trial_adjust_beg_ts) & (
                                df['timestamp'] < trial_adjust_end_ts), 'EVENT'] = 'adjustment_task'

                    # beg of trial
                    df.loc[(df['timestamp'] >= trial_beg_ts) & (
                                df['timestamp'] < trial_fix_beg_ts), 'EVENT'] = 'trial_start'

                    # end of trial
                    df.loc[(df['timestamp'] >= trial_adjust_end_ts) & (
                                df['timestamp'] < trial_end_ts), 'EVENT'] = 'trial_end'

    return df


def pupil_baseline(df, behavioral_df):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()

    fixation_baseline = pd.DataFrame()

    row = 0

    for block in block_unique:
        if not np.isnan(block):

            for trial_number in trial_unique:
                if not np.isnan(trial_number):
                    # get fixation data:
                    trial_data = df[
                        (df['BLOCK'] == block) & (df['TRIAL'] == trial_number) & (df['EVENT'] == 'fixation')]

                    # it seems that for some trial there is no fixation data, then grab it from first blob diameter:

                    mean_fix, ci_low_fix, ci_up_fix = mean_confidence_interval.mean_sem(trial_data['diameter_3d [mm]'])

                    fixation_baseline.loc[row, 'BLOCK'] = block
                    fixation_baseline.loc[row, 'TRIAL'] = trial_number
                    fixation_baseline.loc[row, 'MEAN_PS'] = mean_fix
                    fixation_baseline.loc[row, 'CI_LOW_PS'] = ci_low_fix
                    fixation_baseline.loc[row, 'CI_UP_PS'] = ci_up_fix

                    row += 1

                    df.loc[(df['BLOCK'] == block) & (df['TRIAL'] == trial_number), 'PS_CORRECTED'] = df.loc[(df['BLOCK'] == block) & (df['TRIAL'] == trial_number), 'diameter_3d [mm]'] - mean_fix

    return df, fixation_baseline


def pupil_baseline_alltrials(df):
    # get overall mean fixation:
    fix_data = df[df['EVENT'] == 'fixation']
    mean_fix, ci_low_fix, ci_up_fix = mean_confidence_interval.mean_sem(fix_data['diameter_3d [mm]'])

    df['PS_CORRECTED_ALLTRIALS'] = df['diameter_3d [mm]'] - mean_fix

    return df


def pupil_preprocess(df, annotation_df, behavioral_df):
    df_diameter_cleaned = initial_clean(df, annotation_df)

    df_diameter_cleaned_ts = correct_timestamps(df_diameter_cleaned, annotation_df, behavioral_df)

    df_diameter_cleaned_ts_labeled, trial_sd = label_trials_sd(df_diameter_cleaned_ts, behavioral_df)

    df_final = label_events(df_diameter_cleaned_ts_labeled, annotation_df, behavioral_df)

    return df_final, trial_sd


def align_timestamps_pupil(df, behavioral_df, times_selected, pupil_column_label):
    trial_unique = behavioral_df.trialNumber.unique()
    block_unique = behavioral_df.blockNumber.unique()

    df_matched = pd.DataFrame()

    row = 0

    for block in block_unique:
        if not np.isnan(block):

            print('Block number: ' + str(block))
            for trial_number in trial_unique:

                if not np.isnan(trial_number):
                    # print('Trial number: ' + str(trial_number))

                    # first loop through the times selected:
                    df_trial = df[(df['BLOCK'] == block) & (df['TRIAL'] == trial_number) & (
                                (df['EVENT'] == 'blob') | (df['EVENT'] == 'fixation'))].reset_index(drop=True)
                    if not df_trial.empty:
                        type_sd = df_trial.loc[0, 'SD_CONDITION']
                        for time_sel in times_selected:
                            if time_sel < df_trial.iloc[-1]['TIMESTAMP_CORRECT_IMONSET']:
                                # print('Time selected: ' + str(time_sel))
                                # find closest timestamp in df both + and -
                                closest_number, closest_number_idx = numbers.closest_index(
                                    df_trial['TIMESTAMP_CORRECT_IMONSET'], time_sel)
                                #                             print(closest_number)
                                if (closest_number_idx != 0) & (closest_number_idx != len(df_trial)):
                                    # is it above or below the time selected?
                                    if closest_number < time_sel:  # the timestamp found is smaller than the time selected

                                        closest_number_1 = closest_number
                                        closest_number_1_idx = closest_number_idx
                                        # then the next closest number should be + 1 of the first one
                                        closest_number_2_idx = closest_number_1_idx + 1
                                        closest_number_2 = df_trial.loc[closest_number_2_idx, 'TIMESTAMP_CORRECT_IMONSET']

                                    elif closest_number > time_sel:  # the timestamp found is bigger than the time selected
                                        closest_number_2 = closest_number
                                        closest_number_2_idx = closest_number_idx
                                        # then the next closest number should be - 1 of the first one
                                        closest_number_1_idx = closest_number_2_idx - 1

                                        closest_number_1 = df_trial.loc[closest_number_1_idx, 'TIMESTAMP_CORRECT_IMONSET']

                                    duration = closest_number_2 - closest_number_1

                                    pupil_size_1_corrected = df_trial.loc[closest_number_1_idx, pupil_column_label]
                                    pupil_size_2_corrected = df_trial.loc[closest_number_2_idx, pupil_column_label]
                                    change_pupil_size_corrected = pupil_size_2_corrected - pupil_size_1_corrected

                                    pupil_extrapolated_corrected = pupil_size_1_corrected + (
                                                change_pupil_size_corrected * ((time_sel - closest_number_1) / duration))

                                    pupil_size_1 = df_trial.loc[closest_number_1_idx, 'diameter_3d [mm]']
                                    pupil_size_2 = df_trial.loc[closest_number_2_idx, 'diameter_3d [mm]']
                                    change_pupil_size = pupil_size_2 - pupil_size_1

                                    pupil_extrapolated = pupil_size_1 + (
                                                change_pupil_size * ((time_sel - closest_number_1) / duration))

                                    # include in dataframe
                                    df_matched.loc[row, 'BLOCK'] = block
                                    df_matched.loc[row, 'TRIAL'] = trial_number
                                    df_matched.loc[row, 'SD_CONDITION'] = type_sd
                                    df_matched.loc[row, 'TIMESTAMP_IMONSET'] = time_sel
                                    df_matched.loc[row, 'PUPIL_SIZE_CORRECTED'] = pupil_extrapolated_corrected
                                    df_matched.loc[row, 'PUPIL_SIZE'] = pupil_extrapolated

                                elif closest_number_idx == 0:
                                    pupil_extrapolated_corrected = df_trial.loc[0, pupil_column_label]
                                    pupil_extrapolated = df_trial.loc[0, 'diameter_3d [mm]']
                                    df_matched.loc[row, 'BLOCK'] = block
                                    df_matched.loc[row, 'TRIAL'] = trial_number
                                    df_matched.loc[row, 'SD_CONDITION'] = type_sd
                                    df_matched.loc[row, 'TIMESTAMP_IMONSET'] = time_sel
                                    df_matched.loc[row, 'PUPIL_SIZE_CORRECTED'] = pupil_extrapolated_corrected
                                    df_matched.loc[row, 'PUPIL_SIZE'] = pupil_extrapolated

                                elif closest_number_idx == len(df_trial):
                                    pupil_extrapolated_corrected = df_trial.iloc[-1][pupil_column_label]
                                    pupil_extrapolated = df_trial.iloc[-1]['diameter_3d [mm]']
                                    df_matched.loc[row, 'BLOCK'] = block
                                    df_matched.loc[row, 'TRIAL'] = trial_number
                                    df_matched.loc[row, 'SD_CONDITION'] = type_sd
                                    df_matched.loc[row, 'TIMESTAMP_IMONSET'] = time_sel
                                    df_matched.loc[row, 'PUPIL_SIZE_CORRECTED'] = pupil_extrapolated_corrected
                                    df_matched.loc[row, 'PUPIL_SIZE'] = pupil_extrapolated

                                row += 1

    return df_matched


def plot_sd_category(df_sd, df_nosd, time_limit, ylimit, ylabel, xlabel, title, behavioral_path):
    plot_sd = df_sd[df_sd['TIMESTAMP_IMONSET'] <= time_limit]
    plot_nosd = df_nosd[df_nosd['TIMESTAMP_IMONSET'] <= time_limit]

    plt.plot(plot_sd['TIMESTAMP_IMONSET'], plot_sd['MEAN'], color='red', label='SD')
    plt.fill_between(plot_sd['TIMESTAMP_IMONSET'], plot_sd['SEM_LOW'], plot_sd['SEM_HIGH'], color='red', alpha=0.5)
    plt.plot(plot_nosd['TIMESTAMP_IMONSET'], plot_nosd['MEAN'], color='blue', label='NO_SD')
    plt.fill_between(plot_nosd['TIMESTAMP_IMONSET'], plot_nosd['SEM_LOW'], plot_nosd['SEM_HIGH'], color='blue',
                     alpha=0.5)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.axvline(x=0, color='k', linestyle='--', label='Image Onset')
    plt.ylim(ylimit)
    plt.legend()
    plt.title(title)
    plt.savefig(behavioral_path + title + '.png')
    plt.show()
