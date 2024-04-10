import os
import sys
import subprocess
import json
import codecs
import math
deps_installed = True
try:
    import numpy as np
except ModuleNotFoundError:
    deps_installed = False
    do_it = input('\n\nNumPy not found, do you want to install? [y/n] ')
    while not do_it.lower() in ['y', 'n', 'yes', 'no']:
        print("Wrong input, try again")
        do_it = input('\n\nMatPlotLib not found, do you want to install? [y/n] ')
    if do_it in ['y', 'yes']:
        deps_installed = True
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
        import numpy as np
    else:
        input("\nCannot go further")
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    deps_installed = False
    do_it = input('\n\nMatPlotLib not found, do you want to install? [y/n] ')
    while not do_it.lower() in ['y', 'n', 'yes', 'no']:
        print("Wrong input, try again")
        do_it = input('\n\nMatPlotLib not found, do you want to install? [y/n] ')
    if do_it in ['y', 'yes']:
        deps_installed = True
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
        import matplotlib
        import matplotlib.pyplot as plt
    else:
        input("\nCannot go further")
# from PIL import Image
# matplotlib.use('Qt5Agg')


fig_size_h = 0
fig_size_w = 0
fig_dpi = 0
fig_format = 'png'

class RawSubplotData():
    def __init__(self, pltype, plzero, x, y, xerr_mode, xerr, yerr_mode, yerr,
                axes_labels, axes_pupils, color, fmt, description):
        self.type = pltype
        self.zero = plzero
        self.color = color
        self.x = x
        self.y = y
        self.xerr_mode = xerr_mode
        self.xerr = xerr
        self.yerr_mode = yerr_mode
        self.yerr = yerr
        self.axes_labels = axes_labels
        self.axes_pupils = axes_pupils
        self.color = color
        self.fmt = fmt
        self.description = description

    def print(self):
        print("type: ",        self.type,         '\n',
            "zero: ",        self.zero,          '\n',
            "x: ",           self.x,            '\n',
            "y: ",           self.y,            '\n',
            "xerr_mode: ",   self.xerr_mode,    '\n',
            "xerr: ",        self.xerr,         '\n',
            "yerr_mode: ",   self.yerr_mode,    '\n',
            "yerr: ",        self.yerr,         '\n',
            "axes_labels: ", self.axes_labels,  '\n',
            "axes_pupils: ", self.axes_pupils,  '\n',
            "color: ",       self.color,        '\n',
            "shape: ",       self.fmt,          '\n',
            "description: ", self.description,  '\n')

class JsonParser:
    @classmethod
    def read(self, filename):
        with codecs.open(filename, "r", "utf-8") as read_file:
            data = json.load(read_file)
        return data

    @classmethod
    def parse_object(self, data):
        plots = []
        for i, plot in enumerate(data["data"], start=0):
            array = []
            subplots = plot["subplots"]
            for subplot in subplots:
                array.append(JsonParser.parse_subplot(subplot))
            plots.append((array, plot["title"]))
        return plots

    @classmethod
    def parse_subplot(self, subplot):
        pltype = subplot["type"]
        plzero = False
        if pltype.split('_')[-1] == 'zero':
            plzero = True
            pltype = '_'.join(pltype.split('_')[:-1])
        x = np.array(subplot["x"])
        y = np.array(subplot["y"])
        if pltype == "plot":
            xerr_mode = "absolute"
            yerr_mode = "absolute"
            xerr = np.array([0.0 for i in x])
            yerr = np.array([0.0 for i in y])
            fmt = "o"
        else:
            xerr_mode = subplot["xerr_mode"]
            yerr_mode = subplot["yerr_mode"]
            if xerr_mode == "absolute":
                xerr = np.array(subplot["xerr"])
            elif xerr_mode == "constant":
                xerr = np.array([float(subplot["xerr"]) for i in x])
            elif xerr_mode == "relative":
                xerr = np.array([float(subplot["xerr"]) * i for i in x])
            else:
                print("\nWARNING: considering xerr as \"absolute\"")
                xerr = np.array(subplot["xerr"])
            if yerr_mode == "absolute":
                yerr = np.array(subplot["yerr"])
            elif yerr_mode == "constant":
                yerr = np.array([float(subplot["yerr"]) for i in y])
            elif yerr_mode == "relative":
                yerr = np.array([float(subplot["yerr"]) * i for i in y])
            else:
                print("\nWARNING: considering yerr as \"absolute\"")
                yerr = np.array(subplot["yerr"])
            fmt = subplot["shape"]
        color = subplot["color"]
        axes_labels = subplot["axes_labels"]
        axes_pupils = subplot["axes_pupils"]
        description = subplot["description"]
        return RawSubplotData(pltype, plzero, x, y, xerr_mode, xerr, yerr_mode, yerr,
                            axes_labels, axes_pupils, color, fmt, description)

class Plotter:
    @classmethod
    def plot(self, plots):
        self.makedirs()
        if fig_size_w and fig_size_h:
            fig = plt.figure(figsize=(fig_size_w, fig_size_h))
        else:
            fig = plt.figure()
        axes = [] 
        for i, plot in enumerate(plots):
            f = codecs.open("generated_files/coefs.txt", 'a', "utf-8")
            f.write(plot[1] + '\n\n')
            f.close()
            axes.append(fig.add_subplot(1, len(plots), i+1))
            for subplot in plot[0]:
                Plotter.plot_subplot(axes[i], subplot)
            axes[i].set_title(plot[1])
        with codecs.open("generated_files/coefs.txt", 'a', "utf-8") as f:
            f.write('-----------------------------------------------------\n\n')
        img = open("images/fig."+fig_format, 'w')
        plt.show()
        if fig_dpi:
            fig.savefig("images/fig."+fig_format, fig_dpi=fig_dpi, format=fig_format)
        else:
            fig.savefig("images/fig."+fig_format, format=fig_format)

    @classmethod
    def plot_subplot(self, ax, s):
        x = np.log(s.x) if 'log' in s.type.split('_') and 'x' in s.type.split('_') else s.x
        y = np.log(s.y) if 'log' in s.type.split('_') and 'y' in s.type.split('_') else s.y
        if s.zero:
            _x = np.append(x, x * -1)
            _y = np.append(y, y * -1)
        else:
            _x = x
            _y = y
        # ax.scatter(0, 0, color='white')
        ax.minorticks_on()
        ax.grid(True, which='major', linewidth=1)
        ax.grid(True, which='minor', linewidth=0.5)
        ax.set_xlabel(s.axes_labels[0] + ', ' + s.axes_pupils[0], fontsize=15)
        ax.set_ylabel(s.axes_labels[1] + ', ' + s.axes_pupils[1], fontsize=15)
        border_left = min(x) - 0.2*(max(x)-min(x))
        border_left = 0 if border_left > 0 and s.zero else border_left
        border_right = max(x) + 0.2*(max(x)-min(x))
        border_right = 0 if border_right < 0 and s.zero else border_right
        r = np.linspace(border_left, border_right)

        f = codecs.open("generated_files/coefs.txt", 'a', "utf-8")
        if s.type == 'lsq':
            A = np.vstack([_x, np.ones(len(_y))]).T
            k, b = np.linalg.lstsq(A, _y, rcond=None)[0]
            sigma_k, sigma_b = Plotter.sigma_eval(_x, _y, k, b)
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' +\
                    ': k=' + str(k) + ' b='+str(b) + ' sigma_k='+\
                        str(sigma_k)+' sigma_b='+str(sigma_b)+'\n\n')
            ax.plot(r, k*r+b, color=s.color, label=s.description, linewidth=1)
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3,
                        linewidth=1, color=s.color, ecolor=s.color, capsize=0)

        elif s.type == 'dots':
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3, linewidth=1,
                        color=s.color, label=s.description, ecolor=s.color, capsize=0)

        elif 'log' in s.type.split('_'):
            A = np.vstack([_x, np.ones(len(_y))]).T
            k, b = np.linalg.lstsq(A, _y, rcond=None)[0] 
            sigma_k, sigma_b = Plotter.sigma_eval(_x, _y, k, b)
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' +\
                    ': k=' + str(k) + ' b='+str(b) + ' sigma_k='+\
                    str(sigma_k)+' sigma_b='+str(sigma_b)+'\n\n')
            ax.plot(r, k*r+b, color=s.color, label=s.description, linewidth=1)
            ax.errorbar(x, y, fmt=s.fmt, markersize=3, linewidth=1,
                        color=s.color, ecolor=s.color, capsize=0)

        elif s.type.rstrip('_0123456789') == 'poly':
            coefs = np.polyfit(_x, _y, int(s.type.split('_')[1]))
            ys = np.zeros(len(r))
            for i, c in enumerate(coefs):
                ys += c * r ** (len(coefs)-i-1)
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' + ":\n")
            for i, c in enumerate(coefs):
                f.write("a_"+str(len(coefs)-i-1) + "=" + str(c) + '\n')
            f.write('\n')
            ax.plot(r, ys, color=s.color, label=s.description, linewidth=1)
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3,
                        linewidth=1, color=s.color, ecolor=s.color, capsize=0)

        elif s.type == "plot":
            ax.plot(s.x, s.y, linewidth=1, label=s.description, color=s.color)

        ax.legend()
        f.close()

    @staticmethod 
    def makedirs():
        if not os.path.exists('generated_files'):
            os.mkdir('generated_files')
        fopen = codecs.open("generated_files/coefs.txt", 'a', "utf-8")
        fopen.write('-----------------------------------------------------\n\n')
        fopen.close()
        if not os.path.isdir('images'):
            os.mkdir('images')

    @classmethod
    def sigma_eval(self, x, y, k, b):
        xdisp = np.var(x)
        ydisp = np.var(y)
        sigma_k = np.sqrt((ydisp/xdisp - k ** 2) / (len(x)-2))
        sigma_b = sigma_k * np.sqrt(np.average(x * x))
        return (sigma_k, sigma_b)


if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        args = arg.split('=')
        if len(args) > 1:
            if args[0] == '--size' and len(args[1].split("x")) > 1:
                fig_size_w = float(args[1].split("x")[0])
                fig_size_h = float(args[1].split("x")[1])
            if args[0] == '--dpi':
                fig_dpi = float(args[1])
            if args[0] == '--format':
                fig_format = args[1]

if __name__ == "__main__":
    try:
        if deps_installed:
            data = JsonParser.read("conf.json")
            plots = JsonParser.parse_object(data)
            Plotter.plot(plots)
    except TypeError:
        print("\nData error: check that number of points in x, y, xerr and yerr matches")
        input()
    except json.decoder.JSONDecodeError:
        print("\nData error: check that all points in x, y, xerr \
              and yerr are floating-point numbers")
        input()
    except KeyError:
        print("\nData error: something necessary is missing in conf.json")
        input()
    except FileNotFoundError:
        print("\nData error: conf.json file not found near to main.py")
        input()
    #except ValueError:
    #    print("\nData error: necessary subplot data is missing")
    #    input()
