import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

data = pd.read_csv("./elipse_abr.csv", header=None)
data2 = pd.read_csv("./elipse_05.csv", header=None)
data3 = pd.read_csv("./elipse_025.csv", header=None)
data4 = pd.read_csv("./elipse_01.csv", header=None)

data.columns = ["time", "x_com", "y_com", "pos", "E"]
data2.columns = ["time", "x_com", "y_com", "pos", "E"]
data3.columns = ["time", "x_com", "y_com", "pos", "E"]
data4.columns = ["time", "x_com", "y_com", "pos", "E"]

plt.figure()
plt.plot(data["time"], data["y_com"], label=r"$B_\mathrm{g}(t)\propto\Theta(t)$")
plt.plot(data2["time"], data2["y_com"], label=r"$B_\mathrm{g}(t)\propto\frac{2}{\pi}\arctan(\sinh(0.05t))$")
plt.plot(data3["time"], data3["y_com"], label=r"$B_\mathrm{g}(t)\propto\frac{2}{\pi}\arctan(\sinh(0.025t))$")
plt.plot(data4["time"], data4["y_com"], label=r"$B_\mathrm{g}(t)\propto\frac{2}{\pi}\arctan(\sinh(0.01t))$")
# plt.plot(data3["time"], data3["pos"]-1.5, linestyle="dashed", color = "black", linewidth = 1)
# plt.text(x=625,y=3,s=r"$x(t) = av_x (t - t_0)$", fontsize = 12)
# plt.axhline(y=0, color="black", linestyle="dashed")
plt.xlabel(r"$t\vert J \vert$")
plt.ylabel(r"$x(t)/a$")
# plt.xlim([0,1000])
# plt.ylim([-2,9])
plt.ylim([-.3,.2])
plt.legend()
# plt.grid()
plt.savefig("./x_com_t.png", dpi=600)
# plt.show()

# plt.figure()
# plt.plot(data["time"],data["y_com"], label="41x41")
# plt.plot(data2["x_com"],data2["y_com"], label="51x41")
# plt.plot(data3["x_com"],data3["y_com"], label="51x51")
# plt.plot(data4["x_com"],data4["y_com"], label="51x71")
# plt.xlabel("time")
# plt.ylabel("y/a")
# plt.xlim([-.5,11])
# plt.legend()
# plt.show()
