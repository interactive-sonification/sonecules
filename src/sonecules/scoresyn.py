import numbers
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy
import pyamapping as pam
import sc3nb as scn
from mesonic.synth import Synth
from pandas import DataFrame

from sonecules.base import Sonecule

# TODO rename modules to scorebased son, ... depending on their usage

synonym_list = [
    ["xr", "within", "xrange"],
    ["yr", "to", "yrange"],
    ["fn", "via"],
    ["col", "column", "n", "name", "f", "feat", "feature"],
    ["xrq", "within_q", "xqrange"],
]


def pms(*args, **kwargs):
    """parse mapping specification into mapping directory"""
    dd = {}
    if len(args) > 0:
        # assign positional arguments
        arg_keys = ["col", "fn", "yr"]
        for i, arg in enumerate(args):
            kwargs[arg_keys[i]] = arg
    for k, v in reversed(kwargs.items()):
        for syl in synonym_list:
            if k in syl:
                k = syl[0]
                break
        dd[k] = v
    # set default function if not given
    if "fn" not in dd:
        dd["fn"] = "lin"
    return dd


class BasicPMS(Sonecule):
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
        # clear the current events from the timeline
        ...

    def start(self, **kwargs):
        """start sonification rendering by starting the playback
        kwargs are passed on to start(), so use rate to control speedup, etc.
        """
        # print(kwargs)
        self.context.realtime_playback.start(**kwargs)


def fnarg_to_fun_old(fun):
    """turn fnargument into function if needed
    fun can be either a string ("lin", "log", "exp") for corresponding mapping
    or a function such as pam.linlin, or a custom function which
    offers dmin dmax args for the data min/max and y1, y2 args for the resulting
    parameter args

    Args:
        fun (str or callable): argument passed in mapping tuple

    Returns:
        callable: the function that can be used inside the PMS loops
    """
    if fun == "lin":
        return lambda value, dmin, dmax, y1, y2: pam.linlin(
            value, x1=dmin, x2=dmax, y1=y1, y2=y2
        )
    if fun == "exp":
        return lambda value, dmin, dmax, y1, y2: numpy.exp(
            pam.linlin(value, x1=dmin, x2=dmax, y1=numpy.log(y1), y2=numpy.log(y2))
        )
    if fun == "log":
        return lambda value, dmin, dmax, y1, y2: numpy.log(
            pam.linlin(value, x1=dmin, x2=dmax, y1=numpy.exp(y1), y2=numpy.exp(y2))
        )
    if callable(fun):
        varnames = fun.__code__.co_varnames[: fun.__code__.co_argcount]
        if ("x1" in varnames) and ("x2" in varnames):
            return lambda value, dmin, dmax, y1, y2: fun(
                value, x1=dmin, x2=dmax, y1=y1, y2=y2
            )
        print("ERROR:", varnames)
        return lambda value, dmin, dmax, y1, y2: pam.linlin(
            value, x1=dmin, x2=dmax, y1=y1, y2=y2
        )


def fnarg_to_fun(fun):
    """turn fnargument into function if needed
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
            linlinr(value, xr, [numpy.log(y) for y in yr])
        )

    if fun == "log":
        return lambda value, xr, yr, **kwargs: numpy.log(
            linlinr(value, xr, [numpy.exp(y) for y in yr], **kwargs)
        )
    if callable(fun):
        # varnames = fun.__code__.co_varnames[: fun.__code__.co_argcount]
        return fun


def parse_mapping(mapping):
    """parse a mapping tuple using specification candy

    Args:
        mapping (tuple): the tuple contains
        - feature name
        - mapping function
        - argument dictionary

    the function is needed as
    - instead of a mapping function, shortcut strings "lin", "exp", "log"
    should be accepted and converted into functions as needed
    - instead of an argument dictionary a tuple or list of two numbers
    should be accepted, (for specifying the target range) which should be
    converted into a proper dictionary
    the corrected mapping specification is returned

    Returns:
        tuple: column, mapping function, mapping function argument
        tuple (apart from "dmin" and "dmax")
    """
    col, fun, marg = mapping
    fn = fnarg_to_fun_old(fun)
    if (type(marg) is tuple) or (type(marg) is list):
        mkwargs = {"y1": marg[0], "y2": marg[1]}
    if type(marg) is dict:
        mkwargs = marg
    return col, fn, mkwargs


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
        self.remove()

        df = df.copy()
        df.reset_index()
        dfp = _mapping_to_score_dataframe(df, mapping, sort_by_onset, **odfkwargs)
        self.mapping_df = dfp

        # start synths at the beginning
        with self._context.at(at, info={"sonecule_id": self.sonecule_id}):
            self._synth.start()

        # process all rows
        for idx in dfp.index:
            pvec = dict(dfp.loc[idx])
            onset = pvec["onset"]
            del pvec["onset"]
            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.set(params=pvec)

        # stop synth at the end
        max_onset = dfp["onset"].max()
        stop_time = at + max_onset + stop_after
        with self._context.at(stop_time, info={"sonecule_id": self.sonecule_id}):
            self._synth.stop()

        return self


def identity(x):
    return x


def _pre_arg_to_fn(pa):
    pre_fn = identity
    if isinstance(pa, str):
        if pa == "abs":

            def abs_fn(x):
                return numpy.abs(x)

            pre_fn = abs_fn  # numpy.abs  # lambda x: numpy.abs(x)
        elif pa == "sign":
            pre_fn = numpy.sign  # lambda x: numpy.sign(x)
        elif pa == "diff":

            def series_diff_fn(x):
                return x.diff()

            pre_fn = series_diff_fn  # lambda x: x.diff()
        if pa in ["db_to_amp", "dbamp"]:
            pre_fn = pam.db_to_amp  # lambda x: pam.db_to_amp(x)
        elif pa in ["midicps", "midi_to_cps"]:
            pre_fn = pam.midi_to_cps  # lambda x: pam.midi_to_cps(x)
        elif pa in ["cpsmidi", "cps_to_midi"]:
            pre_fn = pam.cps_to_midi  # lambda x: pam.cps_to_midi(x)
        elif pa in ["amp_to_db", "ampdb"]:
            pre_fn = pam.amp_to_db  # lambda x: pam.amp_to_db(x)
        elif pa in "floor":
            pre_fn = numpy.floor  # lambda x: numpy.floor(x)
    elif callable(pa):
        pre_fn = pa
    return pre_fn


def _post_arg_to_fn(pa):
    post_fn = identity
    if isinstance(pa, str):
        if pa in ["db_to_amp", "dbamp"]:
            post_fn = pam.db_to_amp  # lambda x: pam.db_to_amp(x)
        elif pa in ["midicps", "midi_to_cps"]:
            post_fn = pam.midi_to_cps  # lambda x: pam.midi_to_cps(x)
        elif pa in ["cpsmidi", "cps_to_midi"]:
            post_fn = pam.cps_to_midi  # lambda x: pam.cps_to_midi(x)
        elif pa in ["amp_to_db", "ampdb"]:
            post_fn = pam.amp_to_db  # lambda x: pam.amp_to_db(x)
        elif pa in "floor":
            post_fn = numpy.floor  # lambda x: numpy.floor(x)
    elif callable(pa):
        post_fn = pa
    return post_fn


def _mapping_to_score_dataframe(
    df: DataFrame,
    mapping,
    sort_by_onset=True,
    **odfkwargs,
):
    if sort_by_onset:
        assert "onset" in mapping.keys()
        pspec = mapping["onset"]
        if isinstance(pspec, numbers.Number):
            print("error sort_by onset: no column specified")
            return -1
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
        fun = fnarg_to_fun(pspec["fn"])
        try:
            invals = df[col]
            # print(param, df.shape, df.index)
            xr = dfkwargs["min"][col], dfkwargs["max"][col]
        except KeyError:
            if col == "INDEX":
                invals = numpy.arange(num_rows)
                xr = [0, num_rows]
            else:
                print(f"Exception: KeyError accessing column {col}")
                break

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
            assert (
                synth.mutable
            ), "Synth can be mutable, but should stop and free itself for DiscretePMS"
            self._synth = synth
            return synth
        else:
            # self._synth = self.context.synths.create(synth, track=1, mutable=False)
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
        self.remove()

        dfp = _mapping_to_score_dataframe(df, mapping, sort_by_onset, **odfkwargs)

        # generate sonification events
        for idx in dfp.index:
            pvec = dict(dfp.loc[idx])
            onset = pvec["onset"]
            del pvec["onset"]
            with self._context.at(at + onset, info={"sonecule_id": self.sonecule_id}):
                self._synth.start(params=pvec)
        self.mapping_df = dfp
        return self


class TVOscBankPMS(Sonecule):
    def __init__(self, data, context=None):
        super().__init__(context=context)

        self.data = data  # TODO data needs to be a Asig here .channels, .sr, .sig
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
        remove=True,
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context
        if remove:
            self.remove()

        # start syns (oscillators)
        with ctx.at(time=at):
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

        global_amp = scn.dbamp(level)
        maxonset = -1
        # modulate oscillators
        for j, r in enumerate(dsig):
            onset = j / self.data.sr / rate
            change = r - dsig[max(0, j - 1)]
            with ctx.at(time=at + onset):
                for i, el in enumerate(r):
                    cp = pch_centers[i]
                    dp = pch_wids[i]
                    pitch = pam.linlin(
                        el, channel_mins[i], channel_maxs[i], cp - dp, cp + dp
                    )
                    self.syns[i].freq = scn.midicps(pitch)
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
        with ctx.at(time=at + maxonset):
            for syn in self.syns:
                syn.stop()
        return self


class ContinuousCallbackPMS(Sonecule):
    def __init__(self, data, synthdef=None, context=None):
        super().__init__(context=context)

        self.data = data
        self.synthdef = synthdef
        if self.synthdef:
            scn.SynthDef("contsyn", synthdef).add()
        else:
            # print("no synth definition: use default contsyn")
            scn.SynthDef(
                "contsyn",
                """{ | out=0, freq=400, amp=0.1, vibfreq=0, vibintrel=0,
                        numharm=0, pulserate=0, pint=0, pwid=1, pan=0 |
                var vib = SinOsc.ar(vibfreq, mul: vibintrel*freq, add: freq);
                var sig = Blip.ar(vib, mul: amp, numharm: numharm);
                var pulse = LFPulse.ar(freq: pulserate, iphase: 0.0,
                                width: pwid, mul: pint, add: 1-pint);
                Out.ar(out, Pan2.ar(sig * pulse, pan));
            }""",
            ).add()

        ctx = self.context

        ctx._backend.sc.server.sync()

        self.syn = ctx.synths.create(name="contsyn", track=1, mutable=True)

        self.pdict = {}
        for k, v in self.syn.params.items():
            self.pdict[k] = v.default

    @staticmethod
    def mapcol(r, name, cmins, cmaxs, dmi, dma):
        """service mapcol function"""
        return pam.linlin(r[name], cmins[name], cmaxs[name], dmi, dma)

    def schedule(
        self,
        at=0,
        duration=4,
        callback_fn=None,
        remove=True,
    ):
        # here schedule function, with argument for replace default to true
        # "change"
        ctx = self.context
        if remove:
            self.remove()

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
            onset = pam.linlin(ct, 0, nrows, 0, duration)
            with ctx.at(time=at + onset):
                pdict = callback_fn(r, cmi, cma, self.pdict)
                self.syn.set(pdict)
            if onset > maxonset:
                maxonset = onset
            ct += 1

        # stop oscillators

        # stop oscillator at end
        with ctx.at(time=at + maxonset):
            self.syn.stop()

        return self

    def create_callback_template(self, auto_assign=False):
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
        for p in self.pdict:
            if p == "out":
                continue
            if auto_assign:
                # assign features automatically
                feature = feature_list[fct]
                fct += 1
                if fct == len(feature_list) - 1:
                    fct = 0
                leftstr = f"pp['{p}']"
                bound_left = self.pdict[p] * 0.75
                bound_right = self.pdict[p] * 1.5
                str += (
                    tabstr
                    + f"{leftstr:15s}\t = mapcol(r, '{feature}'"
                    + f", cmi, cma, {bound_left:.2f}, {bound_right:.2f})\n"
                )
                pass
            else:
                str += tabstr + f"pp['{p}']\t = mapcol(r, 'colname', cmi, cma, 1, 2)\n"
            ""
        print(str)
        print(
            "# create sonification e.g. by using\n"
            + "scb.schedule(at=0, duration=2, callback_fn=callback_fn).start(rate=1)\n"
        )
        return str
