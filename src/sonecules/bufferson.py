import numbers
from abc import abstractmethod
from typing import List, Optional, Union

import pandas as pd
from mesonic.context import Context
from numpy import linspace, ndarray
from pya import Asig

from sonecules.base import SchedulableSonecule


class BaseAUD(SchedulableSonecule):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        sr: int = 44100,
        time_column: Optional[Union[str, int]] = None,
        columns: Optional[Union[str, int, List]] = None,
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
        columns : string or Integer
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
            print("Warning: time column is not yet implemented")
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise ValueError("Unsupported Type")

        if columns is None:
            columns = slice(None)

        df = df[columns]
        if isinstance(df, pd.Series):
            df = df.to_frame()

        channel_names = [str(col) for col in df.columns]
        asig = Asig(df.values, sr=sr, cn=channel_names, label=f"df-{channel_names}")
        return cls(asig, sr=None, channels=None, context=context)

    @classmethod
    def from_np(
        cls,
        data: ndarray,
        sr: int = 44100,
        time_column: Optional[Union[str, int]] = None,
        columns: Optional[Union[str, int, List]] = None,
        context: Optional[Context] = None,
    ):
        """
        Construct BufferSynth from numpy ndarray.

        Creates BufferSynth object from numpy array using a by-column or by index
        allowing dtype specification.

        Parameters
        ----------
        data : numpy array
        sr : Number
            sampling rate in Hz
        time_column: string or integer
            name of the column to be used as time index
            if none is given, equidistant data at sampling rate sr is assumed
        columns : string or Integer
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
            print("Warning: time column is not yet implemented")
        if not isinstance(data, ndarray):
            raise ValueError("Unsupported Type")

        if columns is None:
            columns = slice(None)

        data = data[:, columns]

        if isinstance(columns, List):
            channel_names = [str(col) for col in columns]
        elif isinstance(columns, int):
            channel_names = str(columns)
        asig = Asig(data, sr=sr, cn=channel_names, label=f"np-{channel_names}")
        return cls(asig, sr=None, channels=None, context=context)

    def __init__(self, asig, sr=None, channels=None, context=None, sonecule_id=None):
        super().__init__(context=context, sonecule_id=sonecule_id)
        if channels:
            if isinstance(channels, (numbers.Number, str)):
                self.dasig = asig[:, [channels]]
            else:  # assume list or slice
                if channels is None:
                    channels = slice(None)
                print(channels, type(channels))
                self.dasig = asig[:, channels]
        else:
            self.dasig = asig
        if sr:
            self.dasig.sr = sr
        self._create_buffers(self.dasig)

    @abstractmethod
    def _create_buffers(self, asig):
        pass

    def resample(self, target_sr=44100, rate=1, kind="linear"):
        """Resample signal based on interpolation, can process multichannel signals.

        Parameters
        ----------
        target_sr : int
            Target sampling rate (Default value = 44100)
        rate : float
            Rate to speed up or slow down the audio (Default value = 1)
        kind : str
            Type of interpolation (Default value = 'linear')
        """
        self.dasig = self.dasig.resample(target_sr=target_sr, rate=rate, kind=kind)
        self._create_buffers(self.dasig)
        return self

    @abstractmethod
    def set(self, *args, **kwargs):
        ...

    @abstractmethod
    def stop(self, remove=False):
        if remove:
            self.remove()


class BufferSynth(BaseAUD):
    @abstractmethod
    def _prepare_synth_defs(self):
        self.synth_name = None
        ...

    def _create_buffers(self, asig):
        super()._create_buffers(asig)
        self.buf = self.context.buffers.from_data(self.dasig.sig, self.dasig.sr)
        self.synth = self.context.synths.from_buffer(
            self.buf, synth_name=self.synth_name
        )
        self.synth.metadata["sonecule_id"] = self.sonecule_id

    def schedule(self, at=0, **kwargs):
        super().schedule(at, **kwargs)
        with self.context.at(time=at):
            self.synth.start(**kwargs)
        return self

    def set(self, **kwargs):
        super().set()
        self.synth.set(**kwargs)

    def stop(self, remove=False):
        super().stop(remove=remove)
        self.synth.stop()


class BufferSynth1D(BufferSynth):
    def __init__(self, asig, sr=None, channels=None, context=None, sonecule_id=None):
        super().__init__(asig, sr, channels, context, sonecule_id)
        if self.dasig.channels != 1:
            raise ValueError(
                f"{type(self).__name__} does only support 1 dimensional asigs,"
                + " please specify the channel using channels"
            )


class BasicAUD(BufferSynth1D):
    def _prepare_synth_defs(self):
        self.synth_name = "playbuf_aud"
        self.context.synths.add_buffer_synth_def(
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

    def schedule(self, at=0, rate=1, amp=0.1, pan=0, startpos=0, trfreq=0, loop=0):
        # TODO get the defaults/docs for the parameters from the Synth Definition class
        return super().schedule(
            at=at,
            rate=rate,
            amp=amp,
            pan=pan,
            startpos=startpos,
            trfreq=trfreq,
            loop=loop,
        )


class PhasorAUD(BufferSynth1D):
    def _prepare_synth_defs(self):
        self.synth_name = "phasor_aud"
        self.context.synths.add_buffer_synth_def(
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
    ):
        return super().schedule(
            at=at,
            rate=rate,
            amp=amp,
            pan=pan,
            trig=trig,
            relstart=relstart,
            relend=relend,
            respos=respos,
        )


class TimbralPMS(BufferSynth):
    def _prepare_synth_defs(self):
        self.synth_name = "timbralPMS"
        self.context.synths.add_buffer_synth_def(
            self.synth_name,
            r"""{
| bufnum={{BUFNUM}}, freq=90, rate=1, amp=0.1, pan=0, startpos=0, trfreq=0, loop=0 |
    var nch = {{NUM_CHANNELS}};
    var sines = SinOsc.ar(nch.collect{|i| freq*(i+1)});
    var playbufs = PlayBuf.ar(nch, bufnum, BufRateScale.kr(bufnum)*rate,
        Impulse.ar(trfreq)-0.1, startPos: startpos, loop: loop, doneAction: 2);
    var sig = (sines * playbufs).sum;
    Out.ar(0, Pan2.ar(sig, pan, amp))
}""",
        )

    def schedule(
        self,
        at=0,
        rate=1,
        freq=50,
        amp=0.1,
        pan=0,
        startpos=0,
        trfreq=0,
        loop=0,
    ):
        return super().schedule(
            rate=rate,
            freq=freq,
            amp=amp,
            pan=pan,
            startpos=startpos,
            trfreq=trfreq,
            loop=loop,
        )


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


class MultivariateBasicAUD(BaseAUD):
    def _prepare_synth_defs(self):
        return super()._prepare_synth_defs()

    def _create_buffers(self, asig):
        super()._create_buffers(asig)
        auds_kwargs = dict(context=self.context, sonecule_id=self.sonecule_id)
        self.auds = [BasicAUD(asig, channels=ch, **auds_kwargs) for ch in asig.cn]

    def schedule(self, at=0, rate=1, amp=0.1, pan=0, startpos=0, trfreq=0, loop=0):
        if isinstance(pan, str) and pan == "spread":
            pan = list(linspace(-1, 1, len(self.auds), endpoint=True))
        kwargs_list = _expand_multivariate_channel_kwargs(
            len(self.auds), dict(amp=amp, pan=pan)
        )
        with self.context.at(time=at):
            for i, aud in enumerate(self.auds):
                aud.synth.start(
                    rate=rate,
                    trfreq=trfreq,
                    loop=loop,
                    startpos=startpos,
                    amp=kwargs_list[i]["amp"],
                    pan=kwargs_list[i]["pan"],
                )
        return self

    def set(self, **kwargs):
        kwargs_list = _expand_multivariate_channel_kwargs(len(self.auds), kwargs)
        for i, aud in enumerate(self.auds):
            aud.set(**kwargs_list[i])

    def stop(self, remove=False):
        super().stop(remove=remove)
        for aud in self.auds:
            aud.stop()
