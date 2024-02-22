import os
import json
import codecs
try:
    import numpy as np
except ModuleNotFoundError:
    print('\n\nNumPy not found, installing...\n\n')
    os.system("python -m pip install numpy")
    import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    print('\n\nMatPlotLib not found, installing...\n\n')
    os.system("python -m pip install matplotlib")
    import matplotlib
import matplotlib.pyplot as plt
# from PIL import Image
# matplotlib.use('Qt5Agg')

class RawSubplotData():
    def __init__(self, pltype, x, y, xerr_mode, xerr, yerr_mode, yerr, axes_labels, axes_pupils, color, fmt, description):
        self.type = pltype
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
        x = np.array(subplot["x"])
        y = np.array(subplot["y"])
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
        axes_labels = subplot["axes_labels"]
        axes_pupils = subplot["axes_pupils"]
        color = subplot["color"]
        fmt = subplot["shape"]
        description = subplot["description"]
        return RawSubplotData(pltype, x, y, xerr_mode, xerr, yerr_mode, yerr, axes_labels, axes_pupils, color, fmt, description)

class Plotter:
    @classmethod
    def plot(self, plots):
        self.makedirs()
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
        img = open("images/fig.png", 'w')
        plt.show()
        fig.savefig("images/fig.png")

    @classmethod
    def plot_subplot(self, ax, s):
        # ax.scatter(0, 0, color='white') 
        ax.minorticks_on()
        ax.grid(True, which='major', linewidth=1)
        ax.grid(True, which='minor', linewidth=0.5)
        ax.set_xlabel(s.axes_labels[0] + ', ' + s.axes_pupils[0], fontsize=15)
        ax.set_ylabel(s.axes_labels[1] + ', ' + s.axes_pupils[1], fontsize=15) 
        r = np.linspace(0, 1.2*s.x[len(s.x)-1]) 

        f = codecs.open("generated_files/coefs.txt", 'a', "utf-8")
        if (s.type == 'lsq'):
            A = np.vstack([s.x, np.ones(len(s.y))]).T
            k, b = np.linalg.lstsq(A, s.y, rcond=None)[0]
            sigma_k, sigma_b = Plotter.sigma_eval(s.x, s.y, k, b)
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' +\
                    ': k=' + str(k) + ' b='+str(b) + ' sigma_k='+str(sigma_k)+' sigma_b='+str(sigma_b)+'\n\n')
            ax.plot(r, k*r+b, color=s.color, label=s.description, linewidth=1)
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3, linewidth=1, color=s.color, ecolor=s.color, capsize=0)
            
        elif (s.type == 'dots'):
            A = np.vstack([s.x, np.ones(len(s.y))]).T
            k, b = np.linalg.lstsq(A, s.y, rcond=None)[0]
            sigma_k, sigma_b = Plotter.sigma_eval(s.x, s.y, k, b)
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3, linewidth=1, color=s.color, label=s.description, ecolor=s.color, capsize=0)
        
        elif (s.type == 'log'):
            x = np.log(s.x)
            y = np.log(s.y)
            v = np.linspace(0, 1.2*x[len(x)-1])
            A = np.vstack([x, np.ones(len(y))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0] 
            sigma_k, sigma_b = Plotter.sigma_eval(x, y, k, b) 
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' +\
                    ': k=' + str(k) + ' b='+str(b) + ' sigma_k='+str(sigma_k)+' sigma_b='+str(sigma_b)+'\n\n')
            ax.plot(v, k*v+b, color=s.color, label=s.description, linewidth=1) 
            ax.errorbar(x, y, fmt=s.fmt, markersize=3, linewidth=1, color=s.color, ecolor=s.color, capsize=0)

        elif (s.type.rstrip('_0123456789') == 'poly'):
            coefs = np.polyfit(s.x, s.y, int(s.type.split('_')[1]))
            ys = np.zeros(len(r))
            for i, c in enumerate(coefs):
                ys += c * r ** (len(coefs)-i-1)
            f.write(s.type + ' ' + s.color + ' "' + s.fmt + '"' + ":\n")
            for i, c in enumerate(coefs):
                f.write("a_"+str(len(coefs)-i-1) + "=" + str(c) + '\n')
            f.write('\n')
            ax.plot(r, ys, color=s.color, label=s.description, linewidth=1)
            ax.errorbar(s.x, s.y, s.yerr, s.xerr, fmt=s.fmt, markersize=3, linewidth=1, color=s.color, ecolor=s.color, capsize=0)
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

try:
    data = JsonParser.read("conf.json")
    plots = JsonParser.parse_object(data)
    Plotter.plot(plots)
except TypeError:
    print("\nData error: check that number of points in x, y, xerr and yerr matches")
    input()
except json.decoder.JSONDecodeError:
    print("\nData error: check that all points in x, y, xerr and yerr are floating-point numbers")
    input()
except KeyError:
    print("\nData error: something necessary is missing in conf.json")
    input()
except FileNotFoundError:
    print("\nData error: conf.json file not found near to main.py")
    input()
except ValueError:
    print("\nData error: necessary subplot data is missing")
    input()
