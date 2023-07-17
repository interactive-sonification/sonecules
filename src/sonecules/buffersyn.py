from pya import Asig

from sonecules.base import Sonecule


class BufferSynth(Sonecule):
    synth_name: str

    def __init__(self, data, sr, context=None):
        super().__init__(context=context)
        if type(self).synth_name not in self.context.synths.buffer_synthdefs:
            raise NotImplementedError(
                "the selected Context does not offer an {self.synth_name} Synth"
            )
        self.dasig = Asig(data, sr)
        self.buf = self.context.buffers.from_data(self.dasig.sig, self.dasig.sr)
        self.synth = self.context.synths.from_buffer(
            self.buf, synth_name=type(self).synth_name
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

    def schedule(self, at=0, params=None, remove=True):
        if remove:
            self.remove()
        with self.context.at(time=at):
            self.synth.start(params=params)

        # could also offer schedule_from_to(at, until, params) or extend this
        # that changes the rate to match the buffer duration accordingly
        # this would then exclude rate from the viable params


class Audification(BufferSynth):
    synth_name = "playbuf"


class TimbralSon(BufferSynth):
    synth_name = "timbralson"
