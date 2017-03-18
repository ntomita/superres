import matplotlib.pyplot as plt
import numpy


class Plots:
    def __init__(self, nrows=1, ncols=1):
        plt.close('all')
        self.f, self.axs = plt.subplots(nrows, ncols)
        self.nrows = nrows
        self.ncols = ncols
        self._flatten()
        self._set_lines()
        plt.show(False)

    def _flatten(self):
        if self.nrows > 1 and self.ncols > 1:
            self.axs = [ax for raxs in self.axs for caxs in raxs]
        elif self.nrows == 1 and self.ncols == 1:
            self.axs = [self.axs]
        else:
            self.axs = [ax for ax in self.axs]

    def _set_lines(self):
        self.lines = []
        for i in range(self.nrows*self.ncols):
            l, = self.axs[i].plot([], [], 'o')
            self.lines.append([l])

    def _calc_index(self, rind, cind):
        """rind and cind are 1-indexed
        """
        if self.nrows > 1 and self.ncols > 1:
            return self.nrows*(rind-1) + (cind-1)
        else:
            return rind+cind-2

    def set_xlabel(self, xlab="", rind=1, cind=1):
        i = self._calc_index(rind, cind)
        self.axs[i].set_xlabel(xlab)
        plt.draw()

    def set_ylabel(self, ylab="", rind=1, cind=1):
        i = self._calc_index(rind, cind)
        self.axs[i].set_ylabel(ylab)
        plt.draw()

    def set_labels(self, xlab="", ylab="", rind=1, cind=1):
        self.set_xlabel(xlab, rind, cind)
        self.set_ylabel(ylab, rind, cind)

    def set_title(self, title="", rind=1, cind=1):
        i = self._calc_index(rind, cind)
        self.axs[i].set_title(title)
        plt.draw()

    def set_legend(self, labels=[], rind=1, cind=1):
        i = self._calc_index(rind, cind)
        self.axs[i].legend(labels)
        plt.draw()

    def add_line(self, rind=1, cind=1):
        i = self._calc_index(rind, cind)
        l, = self.axs[i].plot([], [], 'o')
        self.lines[i].append(l)

    def update(self, xy_pair, rind=1, cind=1, lind=1):
        """
            Add data points on a plot
            Args:
                xy_pari (iterable cointaining two values, x and y)
                rind (int): row index of subplot (1-indexed)
                    use this option if you set rows on subplot
                cind (int): column index of subplot (1-indexed)
                    use this option if you set columns on subplot
                lind (int): line index of a subplot (1-indexed)
                    use this option if you added extra lines on a plot
        """
        i = self._calc_index(rind, cind)
        line = self.lines[i][lind-1]
        line.set_xdata(numpy.append(line.get_xdata(), xy_pair[0]))
        line.set_ydata(numpy.append(line.get_ydata(), xy_pair[1]))
        self.axs[i].relim()
        self.axs[i].autoscale_view()
        plt.draw()
        plt.pause(0.01)

    def save(self, filename="out.png"):
        plt.savefig(filename)

