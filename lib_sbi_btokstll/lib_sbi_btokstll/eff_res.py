
import numpy


def calculate_resolution(detector_level_dataframe, name_of_variable):

    """
    Calculate the resolution.

    The resolution of a variable is defined as the
    reconstructed value minus the MC truth value.

    If the variable is chi, periodicity is accounted for.
    """

    def apply_periodicity(resolution):

        resolution = resolution.where(resolution < numpy.pi, resolution - 2 * numpy.pi)
        resolution = resolution.where(resolution > -numpy.pi, resolution + 2 * numpy.pi)
        return resolution
    
    measured = detector_level_dataframe[name_of_variable]
    generated = detector_level_dataframe[name_of_variable+'_mc']
    resolution = measured - generated

    if name_of_variable == "chi":

        resolution = apply_periodicity(resolution)

    return resolution


def calculate_efficiency(generator_level_series, detector_level_series, num_bins, bounds):
    
    """
    Calculate the efficiency per bin.

    The efficiency of bin i is defined as the number of
    detector entries in i divided by the number of generator
    entries in i.

    The error for bin i is calculated as the squareroot of the
    number of detector entries in i divided by the number of
    generator entries in i.
    """

    def make_bins(bounds, num_bins):

        assert len(bounds) == 2
        bin_edges, bin_width = numpy.linspace(start=bounds[0], stop=bounds[1], num=num_bins+1, retstep=True)
        bin_middles = numpy.linspace(start=bounds[0]+bin_width/2, stop=bounds[1]-bin_width/2, num=num_bins)
        return bin_edges, bin_middles

    bin_edges, bin_middles = make_bins(bounds, num_bins)

    generator_level_histogram, _ = numpy.histogram(generator_level_series, bins=bin_edges)
    detector_level_histogram, _ = numpy.histogram(detector_level_series, bins=bin_edges)

    print("num events generator: ", generator_level_histogram.sum())
    print("num events detector: ", detector_level_histogram.sum())

    efficiencies = detector_level_histogram / generator_level_histogram
    errors = numpy.sqrt(detector_level_histogram) / generator_level_histogram

    return bin_middles, efficiencies, errors
