import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from typing import Optional

import mesonic
from mesonic.backend.backend_sc3nb import BackendSC3NB
from mesonic.context import Context


def startup(context=None, **kwargs):
    Sonecules.startup(context=context, **kwargs)


def gcc():
    context = Sonecules.default_context
    if context is None:
        raise RuntimeError("No current context. Use startup before.")
    return context


def stop():
    gcc().stop()


def reset(at=0, rate=1):
    gcc().reset(at=at, rate=rate)


def playback():
    pb = Sonecules.default_playback
    if pb is None:
        raise RuntimeError("No current playback")
    return pb


class SoneculeFilter:
    def __init__(self):
        self.deactivated_sonecules = set()

    def __call__(self, event):
        sonecule_id = event.info.get("sonecule_id", None)
        if sonecule_id is not None and sonecule_id in self.deactivated_sonecules:
            return None
        else:
            return event


class Sonecules:
    default_context: Context

    @staticmethod
    def startup(context: Optional[Context] = None):
        if context is None:
            context = mesonic.create_context()

        # what if called twice
        # maybe store dict - context: playback, ...

        Sonecules.default_context = context
        Sonecules.default_context.processor.event_filter = SoneculeFilter()
        # perhaps each context should have a default playback - this would
        # allow using gcc().default_pb instead of the current playback implementation
        Sonecules.default_playback = context.create_playback()

        if isinstance(Sonecules.default_context.backend, BackendSC3NB):
            Sonecules.init_sc3nb_context(context)

    @staticmethod
    def init_sc3nb_context(context: Context):
        import sc3nb as scn

        assert isinstance(context.backend, BackendSC3NB)
        backend: BackendSC3NB = context.backend

        # How could this preperation be placed by the relevant Sonecule instead of here?
        # perhaps using a static methods for this at the classes named f.e. prepare
        # but when should/can these be called?

        context.synths.buffer_synthdefs[
            "timbralson"
        ] = r"""
{ |bufnum={{BUFNUM}}, f0=90, amp=0.1, rate=1 |
    var nch = {{NUM_CHANNELS}};
    var sines = SinOsc.ar(nch.collect{|i| f0*rate*(i+1)});
    var playbufs = PlayBuf.ar(nch, bufnum, BufRateScale.kr(bufnum)*rate, doneAction: 2);
    Out.ar(0, (sines * playbufs).sum * amp!2 )
}"""

        scn.SynthDef(
            "noise",
            r"""
{ |out=0, freq=2000, rq=0.02, amp=0.3, dur=1, pos=0 |
    Out.ar(out, Pan2.ar(
        BPF.ar(WhiteNoise.ar(10), freq, rq)
        * Line.kr(1, 0, dur, doneAction: 2).pow(4), pos, amp));
}""",
        ).add(server=backend.sc.server)
