class PiecewiseObjective(object):

    def __init__(self, intervals_values, name="Piecewise"):
        self.intervals_values = intervals_values
        self.name = name
        super().__init__()

    def __str__(self):
        interval_list = []
        for interval, params in self.intervals_values.items():
            value = ""
            for k, v in params.items():
                value += "{}={}|".format(k, v)
            value = value[:-1]
            interval_list.append([interval[0], interval[1], value])

        weight_list = sorted(interval_list, key=lambda s: s[0])
        weight_list_str = "\n".join(["({}, {}) -> {}".format(*w) for w in weight_list])
        return self.name + ":\n" + weight_list_str

    def __repr__(self):
        return self.__str__()

    def _find_params(self, diff, params=["coef"]):
        for lower_bound, upper_bound in self.intervals_values.keys():
            if (diff > lower_bound) and (diff <= upper_bound):
                return [self.intervals_values[(lower_bound, upper_bound)][p] for p in params]
        return [1]
