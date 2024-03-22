from main import *


# ---------------------------------- \/ YOUR DEFINITION HERE \/ ------------------------------------
START = 0.0
END = 0.5
N_STEPS = 100

# This function should take single X coordinate and return single Y coordinate:
def equation(x: float) -> float:
    y = (math.sin(x * math.pi * 2) + math.sin(x * math.pi * 6) * 0.5) * 0.5 + 0.25
    return y
# ---------------------------------- /\ YOUR DEFINITION HERE /\ ------------------------------------


step = (END - START) / N_STEPS
sx = [START]
while sx[-1] <= END:
    sx.append(sx[-1] + step)
sy = [equation(x) for x in sx]
fig = plt.figure()
output = '"x" : ['
for i in sx:
    output += str(i) + ', '
output = output[:-2] + '],\n'
output += '"y" : ['
for i in sy:
    output += str(i) + ', '
output = output[:-2] + '],\n'
Plotter.makedirs()
with codecs.open("generated_files/plot.txt", "w", "utf-8") as f:
    f.write(output)
plt.plot(sx, sy, color="blue", linewidth=1)
plt.minorticks_on()
plt.grid(True, which='major', linewidth=1)
plt.grid(True, which='minor', linewidth=0.5)
plt.show()
