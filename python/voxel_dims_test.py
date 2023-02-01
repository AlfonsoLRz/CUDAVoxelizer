from colour import Color
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

performance_folder = "performance"

# Read every file within performance folder
response_time = []
bar_title = []

for file in os.listdir(performance_folder):
    bar_title.append(file.split(".")[0])
    with open(os.path.join(performance_folder, file), "r") as f:
        response_time_list = f.readlines()
        response_time_list = np.asarray([float(x.strip()) for x in response_time_list])

        response_time.append(np.mean(response_time_list))

# Plot the average response time
bar_title_numbering = np.asarray([float(x.strip()) for x in bar_title])
sort_indices = np.argsort(bar_title_numbering)
bar_title = np.asarray(bar_title)[sort_indices]
response_time = np.asarray(response_time)[sort_indices]

# Bar chart
import matplotlib.font_manager as font_manager

font = 'Adobe Devanagari'
regular_font = {'fontname': font, 'size': 23, 'color': 'black', 'weight': 'bold'}
font = font_manager.FontProperties(family=font, size=21)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

red = Color("red")
colors = list(red.range_to(Color("green"), response_time.shape[0]))
colors = [color.rgb for color in colors]
plt.bar(bar_title, response_time, color=colors)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)

for label in ax.get_yticklabels():
    label.set_fontproperties(font)

# Print images over bars
bars = ax.patches
ax.set_ylim([0, np.max(response_time) + 500])

for i, (label, value) in enumerate(zip(bar_title, response_time)):
    # load the image corresponding to label into img
    # with cbook.get_sample_data('ada.png') as image_file:
    #    img = plt.imread(image_file)
    # Read image from system file
    img = plt.imread("images/" + str(8) + ".png")
    imagebox = OffsetImage(img, zoom=0.02)
    imagebox.image.axes = ax

    print(value / 5.0)
    ab = AnnotationBbox(imagebox, (i, 0),  xybox=(0., bars[i].get_height() / 8 + 25 - 25 * bars[i].get_height() * .0003), frameon=False,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0)
    ax.add_artist(ab)
    # Resize img
    #plt.imshow(img, extent=[i - 0.1, i + 0.1, value - 8, value - 7], aspect='auto', zorder=2)


plt.xlabel("Number of voxels", **regular_font, labelpad=10)
plt.ylabel("Response time (ms)", **regular_font, labelpad=10)
plt.title("Model 1", **regular_font, pad=20)
plt.tight_layout()
plt.show()

