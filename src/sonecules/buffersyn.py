import numbers
from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd
from mesonic.context import Context
from numpy import linspace
from pya import Asig

from sonecules.base import Sonecule


class BufferSynth(Sonecule, ABC):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        sr: float = None,
        time_column: Union[str, int] = None,
        column: Union[str, int] = 0,
        context: Optional[Context] = None,
    ):
        """
        Construct BufferSynth from Dataframe.

        Creates BufferSynth object from Dataframe using a by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        df : Dataframe
        sr : Number
            sampling rate in Hz
        time_column: string or integer
            name of the column to be used as time index
            if none is given, equidistant data at sampling rate sr is assumed
        column : string or Integer
            Column label to use for data column.

        Returns
        -------
        BufferSynth

        See Also
        --------

        Examples
        --------
        """
        if time_column:
            print("Warning: time column is noy yet implemented")
        if type(df) == pd.core.series.Series:
            series = df
        elif type(df) == pd.DataFrame:
            if column in df.columns:
                series = df.loc[:, column]
            else:
                print("Error: no column {column} in dataframe")
                return
        asig = Asig(series, label="df-{column}")
        if sr:
            asig.sr = sr  # overwrite sampling rate if wished
        return cls(asig, context=context)

    @abstractmethod
    def _prepare_synth_defs(self):
        self.synth_name = None

    def __init__(self, asig, sr=None, channel=None, context=None):
        super().__init__(context=context)
        self._prepare_synth_defs()
        if self.synth_name not in self.context.synths.buffer_synthdefs:
            raise NotImplementedError(
                "the selected Context does not offer an {self.synth_name} Synth"
            )
        if channel:
            self.dasig = asig[:, [channel]]
        else:
            self.dasig = asig
        if sr:
            self.dasig.sr = sr
        self.buf = self.context.buffers.from_data(self.dasig.sig, self.dasig.sr)
        self.synth = self.context.synths.from_buffer(
            self.buf, synth_name=self.synth_name
        )
        self.synth.metadata["sonecule_id"] = self.sonecule_id

    def resample(self, **kwargs):
        """resample to given sampling rate (sr) applying specific resampling rate (rate)
        self.data is assumed to be synchronized with buffer self.buf
        """
        # create Asig
        asig = Asig(self.dasig, sr=...)
        # process using Asig instance...
        self.buf = self.context.buffer.from_asig(asig)
        ...
        return self

    def schedule(self, at=0, params=None, remove=True):
        if remove:
            self.remove()
        with self.context.at(time=at):
            self.synth.start(params=params)
        return self

        # could also offer schedule_from_to(at, until, params) or extend this
        # that changes the rate to match the buffer duration accordingly
        # this would then exclude rate from the viable params

    def stop(self, at=None):
        if at:
            with self.context.at(time=at):
                self.synth.stop()
        else:
            self.synth.stop()


class BasicAUD(BufferSynth):
    def _prepare_synth_defs(self):
        self.synth_name = "playbuf_aud"
        self.context.synths.add_synth_def(
            self.synth_name,
            r"""
            { | bufnum={{BUFNUM}}, rate=1, amp=0.1, pan=0,
                    startpos=0, trfreq=0, loop=0 |
                var trig = Impulse.ar(trfreq, add: -0.5);
                var sig = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum)*rate,
                    trig, startpos, loop: loop, doneAction: 2);
                Out.ar(0, Pan2.ar(sig, pan, amp))
            }""",
        )

    def schedule(
        self, at=0, rate=1, amp=0.1, pan=0, startpos=0, trfreq=0, loop=0, remove=True
    ):
        if remove:
            self.remove()
        with self.context.at(time=at):
            self.synth.start(
                rate=rate, amp=amp, pan=pan, startpos=startpos, trfreq=trfreq, loop=loop
            )
        return self

    def set(self, **kwargs):
        self.synth.set(kwargs)


def _expand_multivariate_channel_kwargs(n, kwargs):
    """for multivariate set() we which to provide a list argument
    to be mapped to each channel. This function returns the
    list of kwargs for the individual items
    n is the number of dimensions
    kwargs the dictionary of kwargs that should be individualized"""
    list_of_kwarg_dicts = []
    for i in range(n):
        kwa = {}
        for k, v in kwargs.items():
            val = None
            if isinstance(v, list):
                if len(v) > i:
                    val = v[i]
            else:
                val = v
            if val is not None:
                kwa[k] = val
        list_of_kwarg_dicts.append(kwa)
    return list_of_kwarg_dicts


class MultivariateBasicAUD(Sonecule):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        sr: float = None,
        time_column: Union[str, int] = None,
        columns: Union[str, int, list] = None,
        context: Optional[Context] = None,
    ):
        """
        Construct MultivariateBasicAUD from Dataframe.

        Creates MultivariateBasicAud object from Dataframe using a
        by columns or by index allowing dtype specification.

        Parameters
        ----------
        df : Dataframe
        sr : Number
            sampling rate in Hz
        time_column: string or integer
            name of the column to be used as time index
            if none is given, equidistant data at sampling rate sr is assumed
        column : string or Integer
            Column label to use for data column.
        context : Optional, passed on to BasicAUD constructor

        Returns
        -------
        MultivariateBasicAUD

        See Also
        --------

        Examples
        --------
        """
        if time_column:
            print("Warning: time column is noy yet implemented")
        if type(df) == pd.core.series.Series:
            df = df.to_frame()
        elif type(df) == pd.DataFrame:
            pass
        else:
            print("error: expecting dataframe or pandas series")

        # ToDo: check whether pya should accept column names to be integers?
        asig = Asig(df.values, cn=[str(n) for n in df.columns], label="from_df")
        if sr:
            asig.sr = sr  # overwrite sampling rate if wished
        return cls(asig, channels=columns, context=context)

    def __init__(self, asig, sr=None, channels=None, context=None):
        if isinstance(channels, str) or isinstance(channels, numbers.Number):
            channels = [channels]
        elif channels is None:
            channels = asig.cn
        self.auds = []
        for i, ch in enumerate(channels):
            aud = BasicAUD(asig, channel=str(ch), context=context)
            self.auds.append(aud)

    def schedule(
        self, at=0, rate=1, amp=0.1, pan=0, startpos=0, trfreq=0, loop=0, remove=True
    ):
        if isinstance(pan, str):
            if pan == "spread":
                pan = list(linspace(-1, 1, len(self.auds), endpoint=True))
        kwargs_list = _expand_multivariate_channel_kwargs(
            len(self.auds), dict(amp=amp, pan=pan)
        )
        for i, aud in enumerate(self.auds):
            aud.schedule(
                at=at,
                rate=rate,
                trfreq=trfreq,
                loop=loop,
                startpos=startpos,
                amp=kwargs_list[i]["amp"],
                pan=kwargs_list[i]["pan"],
            )
        return self

    def set(self, **kwargs):
        # ToDo: check if this causes timing issues
        # (all channels should be kept strictly)
        # synchronous, so bundles should always be processed
        # with reference to the same time!
        kwargs_list = _expand_multivariate_channel_kwargs(len(self.auds), kwargs)
        for i, aud in enumerate(self.auds):
            aud.set(**kwargs_list[i])

    def start(self, **kwargs):
        for aud in self.auds:
            aud.start()

    def stop(self, **kwargs):
        for aud in self.auds:
            aud.stop()


class PhasorAUD(BufferSynth):
    def _prepare_synth_defs(self):
        self.synth_name = "phasor_aud"
        self.context.synths.add_synth_def(
            self.synth_name,
            r"""{
    | out=0, bufnum={{BUFNUM}}, rate=1, amp=0.1, pan=0,
      trig=1, relstart=0, relend=1, respos=0 |
    var nn = BufFrames.kr(bufnum);
    var x = Phasor.ar(trig, rate * BufRateScale.kr(bufnum),
        start: relstart * nn,
        end: relend * nn,
        resetPos: respos * nn);
    Out.ar(out, Pan2.ar(BufRd.ar(1, bufnum, x), pan, amp));
    }""",
        )

    def schedule(
        self,
        at=0,
        rate=1,
        amp=0.1,
        pan=0,
        trig=1,
        relstart=0,
        relend=1,
        respos=0,
        remove=True,
    ):
        if remove:
            self.remove()
        with self.context.at(time=at):
            self.synth.start(
                rate=rate,
                amp=amp,
                pan=pan,
                trig=trig,
                relstart=relstart,
                relend=relend,
                respos=respos,
            )
        return self

    def set(self, **kwargs):
        self.synth.set(kwargs)


class TimbralSon(BufferSynth):
    def _prepare_synth_defs(self):
        self.synth_name = "timbralson"
