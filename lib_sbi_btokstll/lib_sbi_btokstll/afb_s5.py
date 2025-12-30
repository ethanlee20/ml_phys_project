
import numpy
import pandas


def bin_in_var(
    dataframe,
    name_of_binning_variable, 
    start, 
    stop, 
    num_bins,
    return_bin_edges=False,
):
    
    bin_edges = numpy.linspace(start=start, stop=stop, num=num_bins+1)
    bins = pandas.cut(dataframe[name_of_binning_variable], bin_edges, include_lowest=True) # the interval each event falls into
    groupby_binned = dataframe.groupby(bins, observed=False)

    if return_bin_edges:
        return groupby_binned, bin_edges
    return groupby_binned


def calc_bin_middles(start, stop, num_bins):

    bin_edges, step = numpy.linspace(
        start=start,
        stop=stop,
        num=num_bins+1,
        retstep=True,
    )
    bin_middles = bin_edges[:-1] + step/2 
    return bin_middles


def calc_afb_of_q_sq(dataframe, num_bins, start, stop):

    """
    Calcuate Afb as a function of q squared.
    Afb is the forward-backward asymmetry.
    """

    def calc_num_forward(df):

        return df["costheta_mu"][(df["costheta_mu"] > 0) & (df["costheta_mu"] < 1)].count()
    
    def calc_num_backward(df):

        return df["costheta_mu"][(df["costheta_mu"] > -1) & (df["costheta_mu"] < 0)].count()

    def calc_afb(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        return (f - b) / (f + b)

    def calc_afb_err(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)
        return 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins
    )
    
    afbs = groupby_binned.apply(calc_afb)
    errs = groupby_binned.apply(calc_afb_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, afbs, errs


def calc_afb_of_q_sq_over_dc9(dataframe, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    afbs_over_dc9 = []
    afb_errs_over_dc9 = []

    for dc9, df in (dataframe.groupby("dc9")):

        dc9_values.append(dc9)
        bin_mids, afbs, afb_errs = calc_afb_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_mids)
        afbs_over_dc9.append(afbs)
        afb_errs_over_dc9.append(afb_errs)

    return dc9_values, bin_mids_over_dc9, afbs_over_dc9, afb_errs_over_dc9


def calc_s5_of_q_sq(dataframe, num_bins, start, stop):
    
    def calc_num_forward(df):

        cos_theta_k = df["costheta_K"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 0) & (chi < numpy.pi/2)))
            | (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 3*numpy.pi/2) & (chi < 2*numpy.pi)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > numpy.pi/2) & (chi < 3*numpy.pi/2)))
        ].count().min()

    def calc_num_backward(df):

        cos_theta_k = df["costheta_K"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > numpy.pi/2) & (chi < 3*numpy.pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 0) & (chi < numpy.pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 3*numpy.pi/2) & (chi < 2*numpy.pi)))
        ].count().min()

    def calc_s5(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        try: 

            s5 = 4/3 * (f - b) / (f + b)

        except ZeroDivisionError:

            print("S5 calculation: division by 0, returning nan")
            s5 = numpy.nan

        return s5

    def calc_s5_err(df):

        """
        Calculate the error of S_5.

        The error is calculated by assuming the forward and backward
        regions have uncorrelated Poisson errors and propagating
        the errors.
        """

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)

        try: 

            err =  4/3 * 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

        except ZeroDivisionError:

            print("S5 error calculation: division by 0, returning nan")
            err = numpy.nan
        
        return err

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins,     
    )
    
    s5s = groupby_binned.apply(calc_s5)
    errs = groupby_binned.apply(calc_s5_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, s5s, errs


def calc_s5_of_q_sq_over_dc9(dataframe, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    s5s_over_dc9 = []
    s5_errs_over_dc9 = []

    for dc9, df in (dataframe.groupby("dc9")):

        dc9_values.append(dc9)
        bin_middles, s5s, s5_errs = calc_s5_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_middles)
        s5s_over_dc9.append(s5s)
        s5_errs_over_dc9.append(s5_errs)

    return dc9_values, bin_mids_over_dc9, s5s_over_dc9, s5_errs_over_dc9