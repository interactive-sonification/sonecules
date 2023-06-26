from abc import abstractmethod
from typing import Any, Dict, Optional

from mesonic.synth import Synth
from pandas import DataFrame

from sonecules.base import Sonecule
import numbers
import numpy

import sc3nb as scn

# TODO rename modules to scorebased son, ... depending on their usage


class StandardPMSon(Sonecule):
    def __init__(
        self,
        synth: str = "s2",
        parameter_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        context=None,
    ):
        super().__init__(context)
        self._synth = self._pepare_synth(synth)

        if parameter_specs is None:
            parameter_specs = {}

        # treat the additional arguments (e.g., bounds) as synth attributes
        for param, param_spec in parameter_specs.items():
            if param not in self._synth.params:
                raise ValueError(f"{param} is not a parameter of {self._synth}")
            if "bounds" in param_spec:
                getattr(self._synth, param).bounds = param_spec["bounds"]
            if "default" in param_spec:
                getattr(self._synth, param)._default = param_spec["default"]

        # TODO Synth creation vs passing
        # creation here should add metadata which then is added to each produced event
        self._synth.metadata["sonecule_id"] = self.sonecule_id

    @abstractmethod
    def _pepare_synth(self, synth):
        ...

    def synth(self):
        self._synth

    @abstractmethod
    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):  # odfkwargs is a bad name
        # stop after =~~~ release time
        # clear the current events from the timeline
        ...


class StandardContinuousPMSon(StandardPMSon):
    def _pepare_synth(self, synth):
        if isinstance(synth, Synth):
            assert (
                synth.mutable
            ), "Synth needs to be mutable for continuous Parameter Mapping Sonification"
            return synth
        else:
            return self.context.synths.create(synth, track=1)

    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):
        self.reset()

        if sort_by_onset:
            col, _, _ = mapping["onset"]
            df.sort_values(by=col, ascending=True, inplace=True)

        dfkwargs = {"dmin": df.min(), "dmax": df.max()}  # TODO names for dmin, dmax?
        dfkwargs.update(odfkwargs)  # allow overwriting of df

        with self._context.at(at, info={"sonecule_id": self.sonecule_id}):
            self._synth.start()

        max_onset = 0
        for idx in df.index:
            col, fun, mkwargs = mapping["onset"]
            value = getattr(df, col)[idx]

            # get column wise df kwargs
            data_min, data_max = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

            onset = fun(value, **mkwargs, dmin=data_min, dmax=data_max)
            max_onset = max(onset, max_onset)

            params = {}
            for param in [param for param in mapping.keys() if param != "onset"]:
                col, fun, mkwargs = mapping[param]
                value = getattr(df, col)[idx]

                dmin, dmax = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

                params[param] = fun(value, **mkwargs, dmin=dmin, dmax=dmax)

            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.set(params=params)

        with self._context.at(
            at + max_onset + stop_after, info={"sonecule_id": self.sonecule_id}
        ):
            self._synth.stop()


class StandardDiscretePMSon(StandardPMSon):
    def _pepare_synth(self, synth):
        if isinstance(synth, Synth):
            assert (
                not synth.mutable
            ), "Synth needs to be mutable for continuous Parameter Mapping Sonification"
            # self._synth = synth
            return synth
        else:
            #self._synth = self.context.synths.create(synth, track=1, mutable=False)
            return self.context.synths.create(synth, track=1, mutable=False)

    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        stop_after=0.1,
        sort_by_onset=True,
        **odfkwargs,
    ):
        self.reset()

        if sort_by_onset:
            col, _, _ = mapping["onset"]
            df.sort_values(by=col, ascending=True, inplace=True)

        dfkwargs = {"dmin": df.min(), "dmax": df.max()}  # TODO names for dmin, dmax?
        dfkwargs.update(odfkwargs)  # allow overwriting of df

        for idx in df.index:
            col, fun, mkwargs = mapping["onset"]
            value = getattr(df, col)[idx]

            # get column wise df kwargs
            data_min, data_max = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

            onset = fun(value, **mkwargs, dmin=data_min, dmax=data_max)

            params = {}
            for param in [param for param in mapping.keys() if param != "onset"]:
                col, fun, mkwargs = mapping[param]
                value = getattr(df, col)[idx]

                dmin, dmax = dfkwargs["dmin"][col], dfkwargs["dmax"][col]

                params[param] = fun(value, **mkwargs, dmin=dmin, dmax=dmax)

            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.start(params=params)


class TVOSon(Sonecule):
    def __init__(self, data, context=None):
        super().__init__(context=context)

        self.data = data
        ctx = self.context

        # create SynthDef for tvosc
        scn.SynthDef(
            "tvosc-sine-1ch",
            """{ | out=0, freq=400, amp=0.1, pan=0, lg=0.1 | 
            var sig = SinOsc.ar(freq.lag(lg), mul: amp.lag(lg));
            Out.ar(out, Pan2.ar(sig, pan));
        }""",
        ).add()

        ctx._backend.sc.server.sync()

        # create synths
        self.syns = []
        for i in range(data.channels):
            syn = ctx.synths.create(name="tvosc-sine-1ch", track=i, mutable=True)
            self.syns.append(syn)

    def schedule(
        self,
        at=0,
        rate=0.5,
        base_pitch=50,
        pitch_step=7,
        pitch_relwid=0.5,
        amp_mode="absval",
        map_mode="channelwise",
        level=-6,
        reset_flag=True,
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context
        if reset_flag:
            self.reset()   # self.reset

        # start syns (oscillators)
        with ctx.at(time=at):
            for i, syn in enumerate(self.syns):
                syn.start(freq=440, amp=0.1, pan=0, lg=0.1)

        # compute parameters for mapping
        # 1. src ranges for pitch mapping
        if map_mode == "channelwise":
            channel_mins = self.data.sig.min(axis=0)
            channel_maxs = self.data.sig.max(axis=0)
        else:
            channel_mins = numpy.ones(self.data.channels) * self.data.sig.min(axis=0)
            channel_maxs = numpy.ones(self.data.channels) * self.data.sig.max(axis=0)

        if isinstance(pitch_step, numbers.Number):
            pch_centers = [0] * self.data.channels
            pch_wids = [0] * self.data.channels
            for i in range(self.data.channels):
                pch_centers[i] = base_pitch + i * pitch_step 
                pch_wids[i] = pitch_step * pitch_relwid / 2
        elif isinstance(pitch_step, list):
            print(len(pitch_step), self.data.channels)
            assert len(pitch_step) == self.data.channels
            pch_centers = numpy.array(pitch_step) + base_pitch
            pch_wids = numpy.diff([0] + pitch_step) * pitch_relwid 

        global_amp = scn.dbamp(level)
        maxonset = -1
        # modulate oscillators
        for j, r in enumerate(self.data.sig):
            onset = j / self.data.sr / rate
            change = r - self.data.sig[max(0, j - 1)]
            with ctx.at(time=at + onset):
                for i, el in enumerate(r):
                    cp = pch_centers[i]
                    dp = pch_wids[i]
                    pitch = scn.linlin(
                        el, channel_mins[i], channel_maxs[i], cp - dp, cp + dp
                    )
                    self.syns[i].freq = scn.midicps(pitch)
                    if amp_mode == "change":
                        self.syns[i].amp = scn.linlin(
                            abs(change[i]), 0, 0.8, 0, global_amp
                        )
                    elif amp_mode == "absval":
                        srcmax = max(abs(channel_mins[i]), abs(channel_maxs[i]))
                        self.syns[i].amp = scn.linlin(abs(el), 0, srcmax, 0, global_amp)
                    elif amp_mode == "value":
                        self.syns[i].amp = scn.linlin(
                            abs(el), channel_mins[i], channel_maxs[i], 0, global_amp
                        )
            if onset > maxonset:
                maxonset = onset

        # stop oscillators
        with ctx.at(time=at + maxonset):
            for syn in self.syns:
                syn.stop()
        return self
    
    def start(self, **kwargs):
        """start sonification rendering by starting the playback
        kwargs are passed on to start(), so use rate to control speedup, etc.
        """
        print(kwargs)
        # sn.playback().start(**kwargs)



class CPMSonCB(Sonecule):
    def __init__(self, data, synthdef=None, context=None):
        super().__init__(context=context)

        self.data = data
        self.synthdef = synthdef
        if self.synthdef:
            scn.SynthDef("contsyn", synthdef).add()
        else:
            print("no synth definition: use default contsyn")
            scn.SynthDef("contsyn", 
            """{ | out=0, freq=400, amp=0.1, vibfreq=0, vibintrel=0, numharm=0, pulserate=0, pint=0, pwid=1, pan=0 | 
                var vib = SinOsc.ar(vibfreq, mul: vibintrel*freq, add: freq);
                var sig = Blip.ar(vib, mul: amp, numharm: numharm);
                var pulse = LFPulse.ar(freq: pulserate, iphase: 0.0, width: pwid, mul: pint, add: 1-pint);
                Out.ar(out, Pan2.ar(sig * pulse, pan));
            }""").add()

        ctx = self.context

        ctx._backend.sc.server.sync()

        self.syn = ctx.synths.create(name="contsyn", track=1,  mutable=True)

        self.pdict = {}
        for k, v in self.syn.params.items():
            self.pdict[k] = v.default


    @staticmethod
    def mapcol(r, name, cmins, cmaxs, dmi, dma):
        """service mapcol function"""
        return scn.linlin(r[name], cmins[name], cmaxs[name], dmi, dma)

    def schedule(
        self,
        at=0,
        duration=4,
        callback_fn=None,
        reset_flag=True,
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context
        if reset_flag:
            self.reset() 

        # create synths
        with ctx.at(time=at):
            self.syn.start(**self.pdict)

        # compute parameters for mapping
        # 1. src ranges for pitch mapping

        df = self.data
        maxonset = -1
        nrows = df.shape[0]
        cmi = df.min()
        cma = df.max()
        # modulate parameters by data 
        ct = 0 
        for idx, r in df.iterrows():
            onset = scn.linlin(ct, 0, nrows, 0, duration)
            with ctx.at(time=at+onset):
                pdict = callback_fn(r, cmi, cma, self.pdict)
                self.syn.set(pdict)
            if onset > maxonset:
                maxonset = onset
            ct += 1

        # stop oscillators

        # stop oscillator at end
        with ctx.at(time=at+maxonset):
            self.syn.stop()

        return self
    
    def create_callback_template(self, auto_assign=False):
        df = self.data
        tabstr = "    "
        str = "def cbfn(r, cmi, cma, pp):\n"
        str += tabstr + f"# columns are:" 
        feature_list = []
        for i, col in enumerate(df.columns):
            feature = col
            feature_list.append(feature)
            str += f"'{col}' "
            if (i+1) % 4 == 0: 
                str += "\n" + tabstr + '# '
        str += '\n'
        
        fct = 0
        for p in self.pdict:
            if p == 'out': 
                continue
            if auto_assign:
                # assign features automatically
                feature = feature_list[fct]
                fct += 1
                if fct == len(feature_list)-1:
                    fct = 0
                leftstr = f"pp['{p}']"
                bound_left = self.pdict[p]*0.75
                bound_right = self.pdict[p]*1.5
                str += tabstr + f"{leftstr:15s}\t = mapcol(r, '{feature}', cmi, cma, {bound_left}, {bound_right})\n"
                pass
            else:
                str += tabstr + f"pp['{p}']\t = mapcol(r, 'colname', cmi, cma, 1, 2)\n"
            ""
        print(str)
        print("# create sonification e.g. by using\n" +
              "scb.schedule(at=0, duration=2, callback_fn=callback_fn).start(rate=1)\n")
        return str

    def start(self, **kwargs):
        """start sonification rendering by starting the playback
        kwargs are passed on to start(), so use rate to control speedup, etc.
        """
        print(kwargs)
        sn.playback().start(**kwargs)
    