from sonecules.base import Sonecule


class BufferSynth(Sonecule):
    synth_name: str

    def __init__(self, data, sr, context=None):
        super().__init__(context=context)
        if type(self).synth_name not in self.context.synths.buffer_synthdefs:
            raise NotImplementedError(
                "the selected Context does not offer an {self.synth_name} Synth"
            )
        # self.df = ...
        self._buf = self.context.buffers.from_data(data, sr)
        self._synth = self.context.synths.from_buffer(
            self._buf, synth_name=type(self).synth_name
        )
        self._synth.metadata["sonecule_id"] = self.sonecule_id

    def resampling(self, **kwargs):
        # TODO
        # get data from self.buf (not implemented) or save it in __init__
        # > data = self.buf ---- .data
        # create Asig
        # > asig = Asig(data, ...)
        # process using Asig instance...
        # > self.buf = self.context.buffer.from_asig(asig)
        ...

    def schedule(self, at=0, params=None):
        with self.context.at(time=at):
            self._synth.start(params=params)

        # could also offer schedule_from_to(at, until, params) or extend this
        # that changes the rate to match the buffer duration accordingly
        # this would then exclude rate from the viable params


class Audification(BufferSynth):
    synth_name = "playbuf"


class TimbralSon(BufferSynth):
    synth_name = "timbralson"
    # TODO we should perhaps seperate synth name from sonecule name
