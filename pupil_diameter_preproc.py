import numpy as np
from scipy import signal

def rawDataFilter(t_ms, diaSamples, rawFiltSettings=None):
    # rawDataFilter Filter the raw data by identifying a subset as 'valid'.
    #
    #   [valOut,speedFiltData,devFiltData] = rawDataFilter(t_ms,diaSamples...
    #                                        ,rawFiltSettings);
    #
    #    valOut is a logical vector indicating which raw samples are 'valid'.
    #
    #    speedFiltData and devFiltData contain data about the intermediate
    #    speed and deviation filter steps.
    #
    #    t_ms is the time vector corresponding to the entries in diaSamples, in
    #    ms.
    #
    #    diaSamples is a vector containing pupil size data.
    #
    #    rawFiltSettings is a struct containing the settings to be used by the
    #    filter. Call this function without arguments to get the standard
    #    settings:
    #
    #     standardRawSettings = rawDataFilter();
    #
    #    See the 'Standard Settings' section below for details about the
    #    settings.
    #
    # --------------------------------------------------------------------------
    #
    #   This code is part of the supplement material to the article:
    #
    #    Preprocessing Pupil Size Data. Guideline and Code.
    #     Mariska Kret & Elio Sjak-Shie. 2018.
    #
    # --------------------------------------------------------------------------
    #
    #     Pupil Size Preprocessing Code (v1.1)
    #      Copyright (C) 2018  Elio Sjak-Shie
    #       E.E.Sjak-Shie@fsw.leidenuniv.nl.
    #
    #     This program is free software: you can redistribute it and/or modify
    #     it under the terms of the GNU General Public License as published by
    #     the Free Software Foundation, either version 3 of the License, or (at
    #     your option) any later version.
    #
    #     This program is distributed in the hope that it will be useful, but
    #     WITHOUT ANY WARRANTY; without even the implied warranty of
    #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    #     General Public License for more details.
    #
    #     You should have received a copy of the GNU General Public License
    #     along with this program.  If not, see <http://www.gnu.org/licenses/>.
    #
    # --------------------------------------------------------------------------


    ## Standard Settings:
    # These are the default settings for this filter, calling this funtion w/o
    # arguments returns these settings in a struct.


    # Generate default settings:
    if rawFiltSettings is None:

        # ----------------------------------------------------------------------
        # Pupil range filter criteria:

        # The minimum and maximum allowable pupil size:
        PupilDiameter_Min = 1.5
        PupilDiameter_Max = 9

        # ----------------------------------------------------------------------
        # Isolated sample filter criteria:

        # 'Sample-islands' are clusters of samples that are temporally seperated
        # from other samples. The minimum distance used to consider
        # samples 'separated' is specified below:
        islandFilter_islandSeperation_ms = 40  # [ms]

        # When clusters are seperated, they need to have the minimum size
        # specified below. Sample islands smaller than this temporal width and
        # separated from other samples by the distance mentioned above are
        # marked as invalid.
        islandFilter_minIslandWidth_ms = 50  # [ms]

        # ----------------------------------------------------------------------
        # Dilation speed filter criteria:

        # The number of medians to use as the cutoff threshold when applying
        # the speed filter:
        dilationSpeedFilter_MadMultiplier = 16  # [# of MADs]

        # Only calculate the speed when the gap between samples is smaller than
        # the distance specified below:
        dilationSpeedFilter_maxGap_ms = 200  # [ms]

        # ----------------------------------------------------------------------
        # Edge removal filter criteria:

        # Sometimes gaps in the data after the dilation speed filter feature
        # artifacts at their edges. These artifacts are not always removed by
        # the dilation speed filter; as such, samples arounds these gaps may
        # also need to be marked as invalid. The settings below indicate when a
        # section of missing data is classified as a gap (samples around these
        # gaps are in turn rejected).
        gapDetect_minWidth  = 75  # [ms]
        gapDetect_maxWidth   = 2000  # [ms]

        # The settings below indicate the section right before the start of a
        # gap (the backwards padding distance) and the section right after a
        # gap (the forward padding distance), in ms, within which samples are
        # to be rejected:
        gapPadding_backward = 50  # [ms]
        gapPadding_forward = 50  # [ms]

        # ----------------------------------------------------------------------
        # Deviation filter criteria:

        # At this point a subset of the original samples are marked as valid,
        # these samples are the input for this filter.

        # The dilation speed filter will not reject samples that do not feature
        # outlying speeds, such as is the case when these samples are clustered
        # together. As such, a deviation from a smooth trendline filter is
        # warranted. The setting below stipulates how many passes the deviation
        # filter will make:
        residualsFilter_passes = 4  # [# of passes]

        # After each pass, all input samples that are outide of the threshold,
        # which is the multiple stated below of the median, are removed. Note
        # that all input samples are considered (including samples which may
        # have been rejected by the previous deviation filter passes).
        residualsFilter_MadMultiplier = 16  # [# of MADs]

        # At each pass, a smooth continuous trendline is generated using the
        # data below, from which the deviation is than calculated and used as
        # the filter criteria:
        residualsFilter_interpFs = 100  # [Hz]
        residualsFilter_lowpassCF = 16  # [Hz]
        [residualsFilter_lowpassB, residualsFilter_lowpassA] = signal.butter(1, residualsFilter_lowpassCF / (residualsFilter_interpFs / 2));

        # ----------------------------------------------------------------------
        # Keep filter data:

        # The following flag enables/disables the storage of the intermediate
        # filter data, set to false to save memory and improve plotting
        # performance:
        keepFilterData = True

        # ----------------------------------------------------------------------

    
    # if nargin == 0; valOut = rawFiltSettings; return; end




    ## Precomputations:


    # Assume all non-NaN values are initially valid:
    isValid = ~np.isnan(diaSamples)

    ## Step 1: Remove Out-of-Bounds Samples:


    # Remove samples that are larger or smaller than the criteria:
    isValid = removeOutOfBounds(t_ms, diaSamples, isValid, rawFiltSettings)

    ## Step 2: Blink Detection via Speed Filter:


    # Remove blinks and other artifacts that exhibit large differences between
    # samples:
    [isValid, speedFiltData] = madSpeedFilter(t_ms, diaSamples, isValid, rawFiltSettings)

    ## Step 3: Outlier Rejection via Residuals Analysis:


    # Remove outliers that exhibit a large deviation from a smooth trendline:
    [isValid, devFiltData] = madDeviationFilter(t_ms, diaSamples, isValid, rawFiltSettings)




    ## Assign output:


    valOut = isValid


    ## Filter Sub-Functions:


# ==========================================================================
def removeOutOfBounds(t_ms, dia, isValid_In, filtSettings):
    # Removes samples that are not withing the acceptable range.
    #
    # --------------------------------------------------------------------------


    # Get and check the settings:
    minVal = PupilDiameter_Min
    maxVal = PupilDiameter_Max

    assert (maxVal > minVal, 'The maximum must be larger than the minumum.')

    # Reject out of range samples:
    tooLarge = dia > maxVal
    tooSmall = dia < minVal
    isValid = ~tooLarge & ~tooSmall & ~np.isnan(dia) & isValid_In

    # Check that the resulting time vector is strictly increasing (if this
    # assertion fails, additional code is necessary to remove time duplicates):
    assert (~any(np.diff(t_ms(isValid)) == 0), 'Time vector not strictly increasing.')

    # Remove isolated samples:
    isValid = removeLoners(t_ms, isValid, filtSettings)

    # Feedback:
    print('Range Filter: #i samples removed.\n' + str(np.sum(~isValid & isValid_In)))

    return isValid

# ==========================================================================
def madSpeedFilter(t_ms, dia, isValid_In, filtSettings):
    # madSpeedFilter filters a diameter timeseries based on the dilation
    # speeds.
    #
    # --------------------------------------------------------------------------


    # Get parameters and current data:
    maxCalcDist = dilationSpeedFilter_maxGap_ms
    madMultiplier = dilationSpeedFilter_MadMultiplier
    curDiameters = dia(isValid_In)
    cur_t_ms = t_ms(isValid_In)
    maxDilationSpeeds = np.empty(len(dia))

    maxDilationSpeeds[:] = np.NaN

    # Calculate the dilation speeds:
    curDilationSpeeds = np.diff(curDiameters)./ np.diff(cur_t_ms)

    # The maximum gap over which a change is considered:
    curDilationSpeeds(np.diff(cur_t_ms) > maxCalcDist) = np.NaN;

    # Generate a two column array with the back and forward dilation speeds:
    backFwdDilations = [[np.NaN, curDilationSpeeds], [curDilationSpeeds, np.NaN]]


    # Calculate the deviation per sample:
    maxDilationSpeeds(isValid_In) = max(abs(backFwdDilations), [], 2)

    # Calculate the MAD stats:
    [med_d, mad, thresh] = madCalc(maxDilationSpeeds, madMultiplier)

    # Determine the outliers:
    isValid_Out = isValid_In & (maxDilationSpeeds <= thresh)

    # Remove remaining islands:
    isValid_Out = removeLoners(t_ms, isValid_Out, filtSettings)

    # Blinks and other artifact with large inter-sample differences may exhibit
    # distort the signal surrounding samples that do not exceed the filter
    # criteria. As such, remove samples surrounding gaps of a certain size:
    isValid_Out = expandGaps(t_ms, isValid_Out, filtSettings)

    # Set output:
    maxDilationSpeeds = maxDilationSpeeds
    isValid = isValid_Out
    mad = mad
    thresh = thresh
    med_d = med_d

    print('Dilation Speed Filter: #i samples removed.\n' + sum(~isValid_Out & isValid_In));

    return [isValid_Out, filtData]

# ==========================================================================
def madDeviationFilter(t_ms, dia, isValid_In, filtSettings):
    # madDeviationFilter filters a diameter timeseries based on the deviation
    # from a smooth trendline.
    #
    # --------------------------------------------------------------------------
    
    
    # Get relevant settings:
    Npasses = residualsFilter_passes
    smoothFiltA = residualsFilter_lowpassA
    smoothFiltB = residualsFilter_lowpassB
    madMultiplier = residualsFilter_MadMultiplier
    tInterp = (t_ms[0]:(1000 / residualsFilter_interpFs):t_ms[-1])'
    # TODO: check this function above.

    # Create a local copy of the valid samples, return it if there is not
    # enough data for the filter calculations:
    isValid_Running = isValid_In
    if sum(isValid_In) < 3:
        isValidPerPass = []
        residualsPerPass = []
        return

    
    # Remove the previously rejected samples, these are no longer to be
    # considered:
    dia(~isValid_In) = np.NaN
    
    # Preallocate:
    isValid_Running = isValid_In
    isValidPerPass = False(size(isValid_In, 1), Npasses)
    residualsPerPass = NaN(size(isValid_In, 1), Npasses)
    threshPerPass = NaN(1, Npasses);
    smoothBaselinePerPass = NaN(size(isValid_In, 1), Npasses)
    
    # Break if the filter is not doing anything:
    isDone = False
    
    # Allow for multiple passes:
    for passIndx in range(Npasses):
    
        # If the last filter step did not have any effect, neither will this
        # one:
        if isDone:
            continue
    
    
        # Track the validity:
        isValid_Start = isValid_Running
    
        # Calculate the smooth baseline and deviations therefrom:
        [residualsPerPass(:, passIndx), smoothBaselinePerPass(:, passIndx)] = deviationCalculator(t_ms, dia, isValid_Running & isValid_In, tInterp, smoothFiltA, smoothFiltB)
    
        # Calculate the MAD stats:
        [~, ~, threshPerPass(passIndx)] = madCalc(residualsPerPass(:, passIndx), madMultiplier)
    
        # Identify the outliers, and run the isolated sample rejection filter:
        isValid_Running = (residualsPerPass(:, passIndx) <= threshPerPass(passIndx))& isValid_In
        isValid_Running = removeLoners(t_ms, isValid_Running, filtSettings)
    
        # Log loop vars:
        isValidPerPass(:, passIndx)    = isValid_Running
        residualsPerPass(:, passIndx)  = residualsPerPass(:, passIndx)
    
        # Determine is this filter step had an effect:
        if (passIndx > 1) & (all(isValid_Start == isValid_Running)):
    
                # Copy the current results to the other columns:
                isValidPerPass(:, passIndx + 1: end) = repmat(isValid_Running, 1, Npasses - passIndx)
            residualsPerPass(:, passIndx + 1: end) = repmat(residualsPerPass(:, passIndx), 1, Npasses - passIndx)
            smoothBaselinePerPass(:, passIndx + 1: end) = repmat(smoothBaselinePerPass(:, passIndx), 1, Npasses - passIndx)
            threshPerPass(:, passIndx + 1: end) = repmat(threshPerPass(:, passIndx), 1, Npasses - passIndx)
            isDone = True
    
    
    
    
    # Set output:
    isValidPerPass = isValidPerPass
    residualsPerPass = residualsPerPass
    threshPerPass = threshPerPass
    smoothBaselinePerPass = smoothBaselinePerPass
    print('Deviation Filter: #i samples removed.\n' + str(sum(~isValid_Running & isValid_In)))
    
    
    return [isValid_Running, filtData]

## Helper Functions:


# --------------------------------------------------------------------------
function[med_d, mad, thresh] = madCalc(d, n)
# madCalc calculates the rejection threshold using the mad method.
#
# --------------------------------------------------------------------------


# Calc the median:
med_d = nanmedian(d);

# Calc the mad:
mad = nanmedian(abs(d - med_d));

# Calc the threshold:
thresh = med_d + (n * mad);

end

# --------------------------------------------------------------------------
function[dev, smoothBaseline] = deviationCalculator(t_ms...
                                                    , dia, isValid_In...
                                                    , tInterp, smoothFiltA, smoothFiltB)
# deviationCalculator Function for calculating the deviation metrics.
#
# --------------------------------------------------------------------------


# Extract currently valid data:
assert (length(t_ms) == length(isValid_In)...
        & & length(dia) == length(isValid_In), 'Vectors do not agree.')
diaValid = dia(isValid_In & ~isnan(dia));
tValid = t_ms(isValid_In & ~isnan(dia));

# Generate smooth signal using linear interpolation, and nearest neighbour
# extrapolation. Use only the currently valid samples:
uniformBaseline = interp1(tValid, diaValid...
                          , tInterp, 'linear');
uniformBaselineExtra = interp1(tValid, diaValid...
                               , tInterp, 'nearest', 'extrap');
uniformBaseline(isnan(uniformBaseline))...
= uniformBaselineExtra(isnan(uniformBaseline));

# Low pass filter the uniform signal and map it back to the original
# timevector:
smoothUniformBaseline = filtfilt(smoothFiltB, smoothFiltA...
                                 , uniformBaseline);
smoothBaseline = interp1(tInterp, smoothUniformBaseline...
                         , t_ms, 'linear');

# Calculate the deviation:
dev = abs(dia - smoothBaseline);

end

# ==========================================================================
function
validzOut = removeLoners(t_ms, validzIn, filtSettings, doPlot)
# Function for removing isolated sections of data, see 'Standard Settings'
# section of this m file.
#
# --------------------------------------------------------------------------


if nargin < 4
    doPlot = false;
end

# Get settings:
maxSep = filtSettings.islandFilter_islandSeperation_ms;
minIslandWidth = filtSettings.islandFilter_minIslandWidth_ms;

# Isolate the usable samples:
validIndx = find(validzIn);
tValidz = t_ms(validIndx);

# Return if there are not enough samples:
if sum(validzIn) < 3
    validzOut = validzIn;
    return
end

# Detect island borders:
theSea = diff(tValidz) > maxSep;
theSeaShoreLeft = [true;
theSea];
theSeaShoreRight = [theSea;
true];


# Place samples into bins (islands), correct for edge exclusion:
islandBinz = [tValidz(theSeaShoreLeft) - 0.001...
    tValidz(theSeaShoreRight) + 0.001]
';
islandBinz = islandBinz(:);

# NOTE: since MATLAB versions older than 2015a do not have the discretize
# function, the histc function is used here instead, otherwise, in newer
# versions: islandNum = discretize(tValidz,islandBinz...
# ,'IncludedEdge','left');
[~, islandNum] = histc(tValidz, islandBinz);

# Detect the small islands, an their samples:
tinyIslands = find((diff(islandBinz) - 0.002) < minIslandWidth);
tinyIslanders = ismember(islandNum, tinyIslands);

# Assign output (map the valid samples back to the original series):
validzOut = validzIn;
validzOut(validIndx(tinyIslanders)) = false;

# Visualize islands:
if doPlot
    hFig = figure();
    t = t_ms / 1000;
    plot(t(validzIn), zeros(length(t(validzIn)), 1), '.');
    hold
    on;
    plot(tValidz(theSeaShoreLeft) / 1000, 0, 'ro')
    plot(tValidz(theSeaShoreRight) / 1000, 0, 'gd')
    plot(kron(islandBinz, [1;
    1;
    NaN]) / 1000...
    , repmat([-1;
    1;
    NaN], length(islandBinz), 1))
    plot(t((validzIn & ~validzOut))...
         , zeros(sum((validzIn & ~validzOut)), 1), 'kx'...
         , 'MarkerSize', 12, 'LineWidth', 2);
    ylim([-4 4])
    printToConsole(4, ['Isolated Samples Filter:'...
                       ' #i samples (#i clusters) removed.\n']...
                   , sum(tinyIslanders), length(tinyIslands));
    uiwait(hFig);
end

end

# ==========================================================================
function
isValid_In = expandGaps(t_ms, isValid_In, filtSettings)
# Function for removing samples around gaps, see 'Standard Settings'
# section of this m file.
#
# --------------------------------------------------------------------------


# Get settings:
minGap = filtSettings.gapDetect_minWidth;
maxGap = filtSettings.gapDetect_maxWidth;
backPadding = filtSettings.gapPadding_backward;
fwdPadding = filtSettings.gapPadding_forward;

# Blinks produce gaps in the data, the edges of these gaps may feature
# artifacts, as such, dilate gaps:
valid_t = t_ms(isValid_In);
validIndx = find(isValid_In);

# Calculate the duration of each gap, and test whether it exceeds the
# thresholds:
if ~isnan(minGap) | | ~isnan(maxGap)

    gaps = diff(valid_t);
    isGapThatNeedsPadding = gaps > minGap & gaps < maxGap;

    # Get the start and end times of each gap:
    gapStartTimes = valid_t(...
                            [isGapThatNeedsPadding;
    false]);
    gapEndTimes = valid_t(...
                          [false;
    isGapThatNeedsPadding]);

    # Padd gaps that need padding:
    if backPadding > 0 | | fwdPadding > 0

        # Detect samples around the gaps:
        isNearGap = any(bsxfun( @ gt, valid_t...
                        , gapStartTimes
        '-backPadding)...
        & bsxfun( @ lt, valid_t...
        , gapEndTimes
        '+fwdPadding),2);

        #         printToConsole(4,'Edge Removal Filter: #i samples
        #         removed.\n'...
        #             ,sum(isNearGap&isValid_In(validIndx)));

        # Reject samples too near a gap:
        isValid_In(validIndx(isNearGap)) = false;



return [valOut, speedFiltData, devFiltData]






