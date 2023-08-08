import numbers
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy
import pyamapping as pam
from mesonic.synth import Synth
from pandas import DataFrame
from pya import Asig

from sonecules.base import SchedulableSonecule

# TODO rename modules to scorebased son, ... depending on their usage

mapping_spec_synonyms = [
    ["xr", "within", "xrange"],
    ["yr", "to", "yrange"],
    ["fn", "via"],
    ["col", "column", "n", "name", "f", "feat", "feature"],
    ["xrq", "within_q", "xqrange"],
]


def pms(*args, **kwargs):
    """Create parameter mapping specification from provided arguments"""
    dd = {}
    if len(args) > 0:
        # assign positional arguments
        arg_keys = ["col", "fn", "yr"]
        for i, arg in enumerate(args):
            kwargs[arg_keys[i]] = arg
    for k, v in reversed(kwargs.items()):
        for syl in mapping_spec_synonyms:
            if k in syl:
                k = syl[0]
                break
        dd[k] = v
    # set default function if not given
    if "fn" not in dd:
        dd["fn"] = "lin"
    return dd


class BasicPMS(SchedulableSonecule):
    def __init__(
        self,
        synth: str = "s2",
        parameter_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        context=None,
    ):
        super().__init__(context)
        self._synth = self._prepare_synth(synth)

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

    def _prepare_synth_defs(self):
        # In Parameter Mapping Sonification we currently offer
        # no default Synth Definitions
        ...

    @abstractmethod
    def _prepare_synth(self, synth):
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
        # reset the current events from the timeline
        ...


def funarg_to_fun(fun):
    """turn function argument into function if needed

    fun can be either a string ("lin", "log", "exp") for corresponding mapping
    or a function such as pam.linlin, or a custom function which
    offers dmin dmax args for the data min/max and y1, y2 args for the resulting
    parameter args

    Args:
        fun (str or callable): argument passed in mapping tuple

    Returns:
        callable: the function that can be used inside the PMS loops
    """

    def linlinr(value, xr, yr, **kwargs):
        return pam.linlin(value, x1=xr[0], x2=xr[1], y1=yr[0], y2=yr[1])

    if fun == "lin":
        return linlinr
    if fun == "exp":
        return lambda value, xr, yr, **kwargs: numpy.exp(
            linlinr(value, xr, [numpy.log(y) for y in yr], **kwargs)
        )
    if fun == "log":
        return lambda value, xr, yr, **kwargs: numpy.log(
            linlinr(value, xr, [numpy.exp(y) for y in yr], **kwargs)
        )
    if callable(fun):
        # TODO Check here for optional parameters and handle them accordingly
        # varnames = fun.__code__.co_varnames[: fun.__code__.co_argcount]
        return fun


class ContinuousPMS(BasicPMS):
    def _prepare_synth(self, synth):
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
        dfp = _mapping_to_score_dataframe(df, mapping, sort_by_onset, **odfkwargs)
        self.mapping_df = dfp

        # start synths at the beginning
        with self._context.at(at, info={"sonecule_id": self.sonecule_id}):
            self._synth.start()

        # process all rows
        for idx in dfp.index:
            parameters = dict(dfp.loc[idx])
            onset = parameters["onset"]
            del parameters["onset"]
            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.set(params=parameters)

        # stop synth at the end
        max_onset = dfp["onset"].max()
        stop_time = at + max_onset + stop_after
        with self._context.at(stop_time, info={"sonecule_id": self.sonecule_id}):
            self._synth.stop()

        return self


def identity(x):
    return x


def _parg_to_fn(pa):
    post_fn = None
    # TODO pam.by_name(name: str) -> Callable
    if isinstance(pa, str):
        if pa in ["db_to_amp", "dbamp"]:
            post_fn = pam.db_to_amp
        elif pa in ["midi_to_cps", "midicps"]:
            post_fn = pam.midi_to_cps
        elif pa in ["cps_to_midi", "cpsmidi"]:
            post_fn = pam.cps_to_midi
        elif pa in ["amp_to_db", "ampdb"]:
            post_fn = pam.amp_to_db
        elif pa in "floor":
            post_fn = numpy.floor
    return post_fn


def _pre_arg_to_fn(pa):
    pre_fn = identity
    if isinstance(pa, str):
        fun = _parg_to_fn(pa)
        if fun is not None:
            pre_fn = fun
        elif pa == "abs":
            pre_fn = numpy.abs
        elif pa == "sign":
            pre_fn = numpy.sign
        elif pa == "diff":

            def series_diff_fn(x):
                return x.diff()

            pre_fn = series_diff_fn  # lambda x: x.diff()
    elif callable(pa):
        pre_fn = pa
    return pre_fn


def _post_arg_to_fn(pa):
    post_fn = identity
    fun = _parg_to_fn(pa)
    if fun is not None:
        post_fn = fun
    elif callable(pa):
        post_fn = pa
    return post_fn


def _mapping_to_score_dataframe(
    df: DataFrame,
    mapping,
    sort_by_onset=True,
    **odfkwargs,
):
    df = df.copy()
    if sort_by_onset:
        assert "onset" in mapping.keys()
        pspec = mapping["onset"]
        if isinstance(pspec, numbers.Number):
            raise ValueError("Wrong specification for onset")
        elif isinstance(pspec, str):
            pspec = {"col": pspec}
        elif isinstance(pspec, (list, tuple)):
            pspec = dict(zip(("col", "fn", "yr"), pspec))
        col = pspec["col"]
        if col != "INDEX":
            df = df.sort_values(by=col, ascending=True).copy()

    # needed to avoid problems with df.iloc[k:] arguments
    df = df.reset_index()  # index should become column named 'index'
    # ToDo: enable working with df.index being a the onset column...

    # get min and max for all features
    dfkwargs = {"min": df.min(), "max": df.max()}

    # create data frame for mapping results
    num_rows = df.shape[0]
    dfp = DataFrame(index=range(num_rows))

    # create output values for all mapped parameters
    for param, pspec in mapping.items():
        # turn str or tuple pspec into proper dictionaries
        if isinstance(pspec, numbers.Number):
            dfp[param] = pspec
            continue
        if isinstance(pspec, str):
            pspec = {"col": pspec}
        elif isinstance(pspec, (list, tuple)):
            pspec = dict(zip(("col", "fn", "yr"), pspec))
        col = pspec["col"]
        fun = funarg_to_fun(pspec["fn"])
        try:
            invals = df[col]
            # print(param, df.shape, df.index)
            xr = dfkwargs["min"][col], dfkwargs["max"][col]
        except KeyError as error:
            if col == "INDEX":
                invals = numpy.arange(num_rows)
                xr = [0, num_rows]
            else:
                raise error

        # process optional pre mapping argument
        pre_fn = None
        if "pre" in pspec:
            pa = pspec["pre"]
            if isinstance(pa, list):
                pre_fn = [_pre_arg_to_fn(el) for el in pa]
            else:
                pre_fn = _pre_arg_to_fn(pa)

        # process optional post mapping argument
        post_fn = None
        if "post" in pspec:
            pa = pspec["post"]
            if isinstance(pa, list):
                post_fn = [_post_arg_to_fn(el) for el in pa]
            else:
                post_fn = _post_arg_to_fn(pa)

        # process optional xr argument
        if "xr" in pspec:  # if given, it should overwrite xrange from data
            xr = pspec["xr"]

        if "xqr" in pspec:  # it should modify xr using histogram...
            print("xqr not yet implemented")

        # process optional yr argument
        if "yr" in pspec:
            yr = pspec["yr"]
        else:
            print(f"error: param {param} lacks default bounds: set to [0, 1]")
            yr = [0, 1]

        # remove extra args for args passed on to mapping (if given as callable)
        tt = pspec.copy()
        entries_to_remove = ("xr", "yr", "col", "fn", "pre", "post")
        for k in entries_to_remove:
            tt.pop(k, None)

        # apply pre mapping function
        if callable(pre_fn):
            invals = pre_fn(invals)
        elif isinstance(pre_fn, list):
            for elfn in pre_fn:
                invals = elfn(invals)

        # apply mapping
        outvals = fun(invals, xr=xr, yr=yr, **tt)

        # apply clipping
        if "clip" in pspec:
            limits = numpy.sort(yr)
            if pspec["clip"] == "minmax":
                outvals.clip(*limits, inplace=True)

        # apply post mapping function
        if callable(post_fn):
            outvals = post_fn(outvals)
        elif isinstance(post_fn, list):
            for elfn in post_fn:
                outvals = elfn(outvals)

        dfp[param] = outvals
    return dfp


class DiscretePMS(BasicPMS):
    def _prepare_synth(self, synth):
        if isinstance(synth, Synth):
            assert not synth.mutable, "Synth must be immutable for DiscretePMS"
            return synth
        else:
            # self._synth = self.context.synths.create(synth, track=1, mutable=False)
            return self.context.synths.create(synth, track=1, mutable=False)

    def schedule(
        self,
        df: DataFrame,
        mapping,
        at=0,
        sort_by_onset=True,
        **odfkwargs,
    ):
        dfp = _mapping_to_score_dataframe(df, mapping, sort_by_onset, **odfkwargs)
        self.mapping_df = dfp

        # generate sonification events
        for idx in dfp.index:
            pvec = dict(dfp.loc[idx])
            onset = pvec["onset"]
            del pvec["onset"]
            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.start(params=pvec)
        self.mapping_df = dfp
        return self


class TVOscBankPMS(SchedulableSonecule):
    def __init__(self, data, context=None):
        super().__init__(context=context)

        self.data = data  # TODO data needs to be a Asig here .channels, .sr, .sig
        assert isinstance(data, Asig)
        ctx = self.context
        # create synths
        self.syns = []
        for i in range(data.channels):
            syn = ctx.synths.create(name="tvosc-sine-1ch", track=i, mutable=True)
            self.syns.append(syn)

    def _prepare_synth_defs(self):
        super()._prepare_synth_defs()
        # create SynthDef for tvosc
        self.context.synths.add_synth_def(
            "tvosc-sine-1ch",
            """{ | out=0, freq=400, amp=0.1, pan=0, lg=0.1 |
            var sig = SinOsc.ar(freq.lag(lg), mul: amp.lag(lg));
            Out.ar(out, Pan2.ar(sig, pan));
        }""",
        )

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
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context

        # start syns (oscillators)
        with ctx.at(time=at, info={"sonecule_id": self.sonecule_id}):
            for i, syn in enumerate(self.syns):
                syn.start(freq=440, amp=0, pan=0, lg=0.1)

        dsig = self.data.sig
        if len(dsig.shape) == 1:
            dsig = dsig[:, numpy.newaxis]
        # compute parameters for mapping
        # 1. src ranges for pitch mapping
        if map_mode == "channelwise":
            channel_mins = dsig.min(axis=0)
            channel_maxs = dsig.max(axis=0)
        else:
            channel_mins = numpy.ones(self.data.channels) * dsig.min(axis=0)
            channel_maxs = numpy.ones(self.data.channels) * dsig.max(axis=0)

        if isinstance(pitch_step, numbers.Number):
            pch_centers = [0] * self.data.channels
            pch_wids = [0] * self.data.channels
            for i in range(self.data.channels):
                pch_centers[i] = base_pitch + i * pitch_step
                pch_wids[i] = pitch_step * pitch_relwid / 2
        elif isinstance(pitch_step, list):
            assert len(pitch_step) == self.data.channels
            pch_centers = numpy.array(pitch_step) + base_pitch
            pch_wids = numpy.diff([0] + pitch_step) * pitch_relwid

        global_amp = pam.db_to_amp(level)
        maxonset = -1
        # modulate oscillators
        for j, r in enumerate(dsig):
            onset = j / self.data.sr / rate
            change = r - dsig[max(0, j - 1)]
            with ctx.at(time=at + onset, info={"sonecule_id": self.sonecule_id}):
                for i, el in enumerate(r):
                    cp = pch_centers[i]
                    dp = pch_wids[i]
                    pitch = pam.linlin(
                        el, channel_mins[i], channel_maxs[i], cp - dp, cp + dp
                    )
                    self.syns[i].freq = pam.midi_to_cps(pitch)
                    if amp_mode == "change":
                        self.syns[i].amp = pam.linlin(
                            abs(change[i]), 0, 0.8, 0, global_amp
                        )
                    elif amp_mode == "absval":
                        srcmax = max(abs(channel_mins[i]), abs(channel_maxs[i]))
                        self.syns[i].amp = pam.linlin(abs(el), 0, srcmax, 0, global_amp)
                    elif amp_mode == "value":
                        self.syns[i].amp = pam.linlin(
                            abs(el), channel_mins[i], channel_maxs[i], 0, global_amp
                        )
            if onset > maxonset:
                maxonset = onset

        # stop oscillators
        with ctx.at(time=at + maxonset, info={"sonecule_id": self.sonecule_id}):
            for syn in self.syns:
                syn.stop()
        return self


def mapcol(r, name, cmins, cmaxs, dmi, dma):
    """service mapcol function"""
    return pam.linlin(r[name], cmins[name], cmaxs[name], dmi, dma)


class ContinuousCallbackPMS(SchedulableSonecule):
    def _prepare_synth_defs(self):
        self.synth_name = "contsyn"
        self.context.synths.add_synth_def(
            "contsyn",
            """{ | out=0, freq=400, amp=0.1, vibfreq=0, vibintrel=0,
                        numharm=0, pulserate=0, pint=0, pwid=1, pan=0 |
                var vib = SinOsc.ar(vibfreq, mul: vibintrel*freq, add: freq);
                var sig = Blip.ar(vib, mul: amp, numharm: numharm);
                var pulse = LFPulse.ar(freq: pulserate, iphase: 0.0,
                                width: pwid, mul: pint, add: 1-pint);
                Out.ar(out, Pan2.ar(sig * pulse, pan));
            }""",
        )

    def __init__(self, data, synth_name: Optional[str] = None, context=None):
        super().__init__(context=context)

        self.data = data
        self.synth_name = synth_name or self.synth_name
        self.syn = self.context.synths.create(
            name=self.synth_name, track=1, mutable=True
        )
        self.parameter_defaults = {
            name: parameter.default for name, parameter in self.syn.params.items()
        }

    def schedule(
        self,
        at=0,
        duration=4,
        callback_fn=None,
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context

        # start synth
        with ctx.at(time=at):
            self.syn.start(**self.parameter_defaults)

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
            onset = pam.linlin(ct, 0, nrows, 0, duration)
            with ctx.at(time=at + onset):
                pdict = callback_fn(r, cmi, cma, self.parameter_defaults)
                self.syn.set(pdict)
            if onset > maxonset:
                maxonset = onset
            ct += 1

        # stop oscillator at end
        with ctx.at(time=at + maxonset):
            self.syn.stop()

        return self

    def create_callback_template(self, auto_assign=False):
        df = self.data
        tabstr = "    "
        callback_code = "def cbfn(r, cmi, cma, pp):\n"
        callback_code += tabstr + "# columns are: "
        feature_list = []
        for i, col in enumerate(df.columns):
            feature = col
            feature_list.append(feature)
            callback_code += f"'{col}' "
            if (i + 1) % 4 == 0:
                callback_code += "\n" + tabstr + "# "
        callback_code += "\n"

        fct = 0
        for parameter_name, default in self.parameter_defaults.items():
            if parameter_name == "out":
                continue
            if auto_assign:
                # assign features automatically
                feature = feature_list[fct]
                fct += 1
                if fct == len(feature_list) - 1:
                    fct = 0
                leftstr = f"pp['{parameter_name}']"
                bound_left = default * 0.75
                bound_right = default * 1.5
                callback_code += (
                    tabstr
                    + f"{leftstr:15s}\t = mapcol(r, '{feature}'"
                    + f", cmi, cma, {bound_left:.2f}, {bound_right:.2f})\n"
                )
            else:
                callback_code += (
                    tabstr
                    + f"pp['{parameter_name}']\t"
                    + " = mapcol(r, 'colname', cmi, cma, 1, 2)\n"
                )
        callback_code += tabstr + "return pp"
        print(callback_code)
        print(
            "# create sonification e.g. by using\n"
            + "sn.gcc().timeline.reset()\n"
            + "# scb.schedule(at=0, duration=5, callback_fn=cbfn).start(rate=1)\n"
        )
        return callback_code


class DiscreteCallbackPMS(SchedulableSonecule):
    def _prepare_synth_defs(self):
        self.synth_name = "dcbpms"
        self.context.synths.add_synth_def(
            self.synth_name,
            r"""{ | out=0, freq=400, dur=0.4, att=0.001, rel=0.5, amp=0,
        vibfreq=0, vibir=0, sharp=0, pan=0 |
var vib = SinOsc.ar(vibfreq, mul: vibir*freq, add: freq);
var sig = HPF.ar(Formant.ar(vib, vib, bwfreq: vib*(sharp+1), mul: amp), 40);
var env = EnvGen.kr(Env.new([0,1,1,0], [att, dur-att-rel, rel]), doneAction:2);
Out.ar(out, Pan2.ar(sig, pan, env));
}""",
        )

    def __init__(self, data, synth_name: Optional[str] = None, context=None):
        super().__init__(context=context)
        self.data = data
        self.synth_name = synth_name or self.synth_name
        self.syn = self.context.synths.create(
            name=self.synth_name, track=1, mutable=False
        )
        self.parameter_defaults = {
            name: parameter.default for name, parameter in self.syn.params.items()
        }

    def schedule(
        self,
        at=0,
        callback_fn=None,
        remove=True,
    ):
        df = self.data
        cmi = df.min()
        cma = df.max()

        # spawn synths for each row
        for idx, r in df.iterrows():
            pdict = callback_fn(r, cmi, cma, self.parameter_defaults)
            onset = pdict["onset"]
            del pdict["onset"]
            with self.context.at(
                time=at + onset, info={"sonecule_id": self.sonecule_id}
            ):
                self.syn.start(params=pdict)
        return self

    def create_callback_template(self, auto_assign=False, duration=5):
        df = self.data
        tabstr = "    "
        str = "def cbfn(r, cmi, cma, pp):\n"
        str += tabstr + "# columns are:"
        feature_list = []
        for i, col in enumerate(df.columns):
            feature = col
            feature_list.append(feature)
            str += f"'{col}' "
            if (i + 1) % 4 == 0:
                str += "\n" + tabstr + "# "
        str += "\n"

        fct = 0
        # go through parameters, with onset being the first
        parameter_names = ["onset"] + list(self.parameter_defaults.keys())
        if "out" in list(self.parameter_defaults.keys()):
            parameter_names.remove("out")
        self.parameter_defaults["onset"] = 0  # set a default value for onset

        for p in parameter_names:
            if auto_assign:
                # assign features automatically
                feature = feature_list[fct]
                fct += 1
                if fct == len(feature_list) - 1:
                    fct = 0
                leftstr = f"pp['{p}']"
                bound_left = self.parameter_defaults[p] * 0.75
                bound_right = self.parameter_defaults[p] * 1.5
                if p == "onset":
                    bound_right = duration
                str += (
                    tabstr
                    + f"{leftstr:15s}\t = mapcol(r, '{feature}'"
                    + f", cmi, cma, {bound_left:.2f}, {bound_right:.2f})\n"
                )
                pass
            else:
                str += tabstr + f"pp['{p}']\t = mapcol(r, 'colname', cmi, cma, 1, 2)\n"
            ""
        str += tabstr + "return pp"
        print(str)
        print(
            "# create sonification e.g. by using\n"
            + "sn.gcc().timeline.reset()\n"
            + "# scb.schedule(at=0, callback_fn=cbfn).start(rate=1)\n"
        )
        return str
