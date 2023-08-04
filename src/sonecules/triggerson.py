# might not be a wanted dep
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyamapping import db_to_amp, linlin

# for distance calculation in the data Sonogram
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from sonecules.base import Sonecule


class DataSonogramMBS(Sonecule):
    def __init__(
        self,
        df,
        x,
        y,
        label,
        max_duration=1.5,
        spring_synth="springmass",
        trigger_synth="noise",
        rtime=0.5,
        level=-6,
        play_trigger_sound=True,
        context=None,
    ):
        super().__init__(context=context)

        # prepare synths
        self.trigger_synth = self.context.synths.create(
            trigger_synth, mutable=False
        )  # TODO make sure the synths have metadata
        self.spring_synth = self.context.synths.create(spring_synth, mutable=False)

        # save dataframe
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])

        # check if x and y are valid
        allowed_columns = self.numeric_df.columns
        assert x in allowed_columns, f"x must be in {allowed_columns}"
        assert y in allowed_columns, f"y must be in {allowed_columns}"

        # prepare data for model
        self.labels = self.df[label]
        self.unique_labels = self.labels.unique()
        label2id = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.numeric_labels = [label2id[label] for label in self.labels]
        self.xy_data = self.numeric_df[[x, y]].values
        self.data = self.numeric_df.values

        # get the convex hull of the data
        hull = ConvexHull(self.data)
        hull_data = self.data[hull.vertices, :]
        # get distances of the data points in the hull
        hull_distances = cdist(hull_data, hull_data, metric="euclidean")
        self.max_distance = hull_distances.max()

        # set model parameter
        self.play_trigger_sound = play_trigger_sound
        self.max_duration = max_duration
        self.rtime = rtime
        self.level = level

        self._latency = 0.2

        # prepare plot
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.subplot(111)

        # plot data
        sns.scatterplot(x=x, y=y, hue=label, data=df, ax=self.ax)

        # set callback
        def onclick(event):
            if event.inaxes is None:  # outside plot area
                return
            if event.button != 1:  # ignore other than left click
                return
            click_xy = np.array([event.xdata, event.ydata])
            self.create_shockwave(click_xy)

        self.fig.canvas.mpl_connect("button_press_event", onclick)

    def _prepare_synth_defs(self):
        super()._prepare_synth_defs()

        self.context.synths.add_synth_def(
            "noise",
            r"""
            { |out=0, freq=2000, rq=0.02, amp=0.3, dur=1, pan=0 |
                var noise = WhiteNoise.ar(10);
                var filtsig = BPF.ar(noise, freq, rq);
                var env = Line.kr(1, 0, dur, doneAction: 2).pow(4);
                Out.ar(out, Pan2.ar(filtsig, pan, env*amp));
            }""",
        )
        self.context.synths.add_synth_def(
            "springmass",
            r"""
            { |out=0, freq=2000, amp=0.3, rtime=0.5, pan=0 |
                var exc = Impulse.ar(0);
                var sig = Klank.ar(`[[freq], [0.2], [rtime]], exc);
                DetectSilence.ar(exc+sig, doneAction: Done.freeSelf);
                Out.ar(out, Pan2.ar(sig, pan, amp));
            }""",
        )

    def create_shockwave(self, click_xy):
        self.context.reset()

        # play trigger "shockwave" sound sample
        if self.play_trigger_sound:
            with self.context.now(self._latency) as start_time:
                self.trigger_synth.start()
        else:
            start_time = self.context.playback.time

        # find the point that is the nearest to the click location
        center_idx = np.argmin(np.linalg.norm(self.xy_data - click_xy, axis=1))
        center = self.data[center_idx]

        # get the distances from the other points to this point
        distances_to_center = np.linalg.norm(self.data - center, axis=1)

        # get idx sorted by distances
        order_of_points = np.argsort(distances_to_center)

        # for each point create a sound using the spring synth
        for idx in order_of_points:
            distance = distances_to_center[idx]
            nlabel = self.numeric_labels[idx]
            onset = (distance / self.max_distance) * self.max_duration
            with self.context.at(start_time + onset):
                self.spring_synth.start(
                    freq=2 * (400 + 100 * nlabel),
                    amp=db_to_amp(
                        self.level + linlin(distance, 0, self.max_distance, 0, -30)
                    ),
                    pan=[-1, 1][int(self.xy_data[idx, 0] - click_xy[0] > 0)],
                    rtime=self.rtime,
                    info={"label": self.labels[idx]},
                )
